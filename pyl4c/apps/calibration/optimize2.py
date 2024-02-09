import os
import yaml
import warnings
import numpy as np
import h5py
import pyl4c
from typing import Sequence
from matplotlib import pyplot
from scipy import signal
from pyl4c import pft_dominant
from pyl4c.data.fixtures import restore_bplut_flat
from pyl4c.science import vpd, par, rescale_smrz
from pyl4c.stats import linear_constraint, rmsd

L4C_DIR = os.path.dirname(pyl4c.__file__)
PFT_VALID = (1,2,3,4,5,6,7,8)

# This matplotlib setting prevents labels from overplotting
pyplot.rcParams['figure.constrained_layout.use'] = True


class CalibrationAPI(object):
    '''
    Convenience class for calibrating the L4C GPP and RECO models. Meant to
    be used with `fire.Fire()`. Uses:

        # Run the calibration for a specific PFT
        python optimize.py tune-gpp --pft=<pft>

        # Get access to the sampler (and debugger), after calibration is run
        python optimize.py tune-gpp --pft=<pft> --ipdb
    '''
    _required_parameters = {
        'GPP':  ['LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'],
        'RECO': ['CUE', 'tsoil', 'smsf0', 'smsf1'],
    }
    _required_drivers = {
        # Tsurf = Surface skin temperature; Tmin = Minimum daily temperature
        'GPP':  ['fPAR', 'PAR', 'Tmin', 'VPD', 'SMRZ', 'FT'],
        'RECO': ['SMSF', 'Tsoil']
    }

    def __init__(self, config = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                L4C_DIR, 'data/files/config_L4C_MCMC_calibration.yaml')
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.hdf5 = self.config['data']['file']

    def _clean(
            self, raw: Sequence, drivers: Sequence, protocol: str = 'GPP',
            num_std: int = 5):
        'Cleans up data values according to a prescribed protocol'
        if protocol == 'GPP':
            # Filter out observed GPP values when GPP is negative or when
            #   APAR < 0.1 g C m-2 day-1
            apar = np.nanmean(drivers['fPAR'], axis = -1) * drivers['PAR']
            cleaned = np.where(
                apar < 0.1, np.nan, np.where(raw < 0, np.nan, raw))
            return np.apply_along_axis(
                lambda x: np.where(
                    x > (num_std * np.nanstd(x)), np.nan, x), 0, cleaned)

    def _filter(self, raw: Sequence, size: int):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        return raw # Or, revert to the raw data

    def _get_params(self, pft, model):
        # Filter the parameters to just those for the PFT of interest
        params_dict = restore_bplut_flat(self.config['BPLUT'])
        return dict([
            (k, params_dict[k].ravel()[pft])
            for k in self._required_parameters[model]
        ])

    def _load_gpp_data(self, pft, blacklist, filter_length):
        'Load the required datasets for GPP, for a single PFT'
        with h5py.File(self.hdf5, 'r') as hdf:
            sites = hdf['site_id'][:]
            if hasattr(sites[0], 'decode'):
                sites = list(map(lambda x: x.decode('utf-8'), sites))
            # Get dominant PFT
            pft_map = pft_dominant(hdf['state/PFT'][:], sites)
            # Blacklist validation sites
            pft_mask = np.logical_and(
                np.in1d(pft_map, pft), ~np.in1d(sites, blacklist))
            drivers = dict()
            field_map = self.config['data']['fields']
            for field in self._required_drivers['GPP']:
                # Try reading the field exactly as described in config file
                if field in field_map:
                    if field_map[field] in hdf:
                        # Preserve 1-km subgrid for fPAR
                        if field == 'fPAR':
                            drivers[field] = hdf[field_map[field]][:,pft_mask,:]
                        else:
                            drivers[field] = hdf[field_map[field]][:,pft_mask]
                elif field == 'PAR':
                    if 'SWGDN' not in field_map:
                        raise ValueError(f"Could not find PAR or SWGDN data")
                    drivers[field] = par(hdf[field_map['SWGDN']][:,pft_mask])
                elif field == 'VPD':
                    qv2m = hdf[field_map['QV2M']][:,pft_mask]
                    ps   = hdf[field_map['PS']][:,pft_mask]
                    t2m  = hdf[field_map['T2M']][:,pft_mask]
                    drivers[field] = vpd(qv2m, ps, t2m)
                elif field == 'SMRZ':
                    smrz = hdf[field_map['SMRZ0']][:,pft_mask]
                    smrz_min = smrz.min(axis = 0)
                    drivers[field] = rescale_smrz(smrz, smrz_min)
                elif field == 'FT':
                    tsurf = hdf[field_map['Tsurf']][:,pft_mask]
                    # Classify soil as frozen (FT=0) or unfrozen (FT=1) based
                    #   on threshold freezing point of water
                    drivers[field] = np.where(tsurf <= 273.15, 0, 1)

            # Check units on fPAR
            if np.nanmax(drivers['fPAR'][:]) > 10:
                drivers['fPAR'] /= 100
            assert len(set(self._required_drivers['GPP'])\
                .difference(set(drivers.keys()))) == 0,\
                'Did not find all required drivers for the GPP model!'

            # If RMSE is used, then we want to pay attention to weighting
            weights = None
            if 'weights' in hdf.keys():
                weights = hdf['weights'][pft_mask][np.newaxis,:]\
                    .repeat(t2m.shape[0], axis = 0)
            else:
                print('WARNING - "weights" not found in HDF5 file!')
            if 'GPP' not in hdf.keys():
                with h5py.File(
                        self.config['data']['supplemental_file'], 'r') as _hdf:
                    tower_gpp = _hdf['GPP'][:][:,pft_mask]
            else:
                tower_gpp = hdf['GPP'][:][:,pft_mask]

        # Check that driver data do not contain NaNs
        for field in drivers.keys():
            if field == 'fPAR':
                continue # The 1-km subgrid may have NaNs
            assert not np.isnan(drivers[field]).any(),\
                f'Driver dataset "{field}" contains NaNs'
        # Clean observations, then mask out driver data where the are no
        #   observations
        tower_gpp = self._filter(tower_gpp, filter_length)
        tower_gpp = self._clean(tower_gpp, drivers, protocol = 'GPP')
        tower_gpp_flat = tower_gpp[~np.isnan(tower_gpp)]
        # Subset all datasets to just the valid observation site-days
        if weights is not None:
            weights = weights[~np.isnan(tower_gpp)]
        drivers_flat = dict()
        for field in drivers.keys():
            drivers_flat[field] = drivers[field][~np.isnan(tower_gpp)]
        return (drivers, drivers_flat, tower_gpp, tower_gpp_flat, weights)

    def _plot(self, pft: int, model: str):
        'Configures the sampler and backend'
        params_dict = self._get_params(pft, model)
        backend = self.config['optimization']['backend_template'].format(
            model = model, pft = pft)
        sampler = L4CStochasticSampler(
            self.config, getattr(L4CStochasticSampler, f'_{model.lower()}'),
            params_dict, backend = backend)
        return (sampler, backend)

    @staticmethod
    def e_mult(params, tmin, vpd, smrz, ft):
        # Calculate E_mult based on current parameters
        f_tmin = linear_constraint(params[1], params[2])
        f_vpd  = linear_constraint(params[3], params[4], 'reversed')
        f_smrz = linear_constraint(params[5], params[6])
        f_ft   = linear_constraint(params[7], 1.0, 'binary')
        e = f_tmin(tmin) * f_vpd(vpd) * f_smrz(smrz) * f_ft(ft)
        return e[...,np.newaxis]

    @staticmethod
    def gpp(params, apar, tmin, vpd, smrz, ft):
        # Calculate GPP based on the provided BPLUT parameters
        return apar * params[0] * self.e_mult(params, tmin, vpd, smrz, ft)

    def tune_gpp(
            self, pft: int, filter_length: int = 2, plot: str = None,
            ipdb: bool = False, save_fig: bool = False):
        '''
        Run the L4C GPP calibration.

        - For GPP data: Removes observations where GPP < 0 or where APAR is
            < 0.1 MJ m-2 day-1

        Parameters
        ----------
        pft : int
            The Plant Functional Type (PFT) to calibrate
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data
        plot : str or None
            Plot either: the "trace" for a previous calibration run; an
            "exemplar", or single time series showing tower observations and
            simulations using new and old parameters; a "scatter" plot
            showing simulations, using new and old parameters, against
            observations, with RMSE; or the "posterior" plot, an HDI plot
            of the posterior distribution(s). If None, calibration will
            proceed (calibration is not performed if plotting).
        ipdb : bool
            True to drop the user into an ipdb prompt, prior to and instead of
            running calibration
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        '''
        def residuals(params, drivers, observed, weights):
            apar, tmin, vpd, smrz, ft = drivers
            # Objective function: Difference between tower GPP and L4C GPP
            gpp0 = self.gpp(params, apar, tmin, vpd, smrz, ft).mean(axis = 2)
            diff = np.subtract(observed, gpp0)
            # Multiply by the tower weights
            return (weights * diff)[~np.isnan(diff)]

        assert pft in PFT_VALID, f'Invalid PFT: {pft}'
        params_dict = self._get_params(pft, 'GPP')
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']
        objective = self.config['optimization']['objective'].lower()

        print('Loading driver datasets...')
        drivers, drivers_flat, tower_gpp, tower_gpp_flat, weights =\
            self._load_gpp_data(pft, blacklist, filter_length)


if __name__ == '__main__':
    import fire
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
