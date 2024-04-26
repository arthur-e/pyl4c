'''
To calibrate the GPP or RECO model for a specific PFT:

    python optimize.py pft <pft> tune-gpp
    python optimize.py pft <pft> tune-reco

To plot a specific PFT's (optimized) response to a driver:

    python optimize.py pft <pft> plot-gpp <driver>
    python optimize.py pft <pft> plot-reco <driver>

Where:

- For `plot-gpp`, `<driver>` is one of: `smrz`, `vpd`, `tmin`.
- For `plot-reco`, `<driver>` is one of: `smsf`, `tsoil`.

To plot the steady-state SOC versus IGBP SOC pit measurements, and then
select potential new decay rates:

    python optimize.py pft <pft> tune-soc

To get the goodness-of-fit statistics for the updated parameters:

    python optimize.py pft <pft> score GPP
    python optimize.py pft <pft> score RECO

To view the current (potentially updated) BPLUT:

    python optimize.py bplut show

    # View parameter values for <param> across ALL PFTs
    python optimize.py bplut show None <param>

    # View a PFT's values across ALL parameters
    python optimize.py bplut show <param> None
'''

import datetime
import os
import yaml
import warnings
import numpy as np
import h5py
import pyl4c
from typing import Sequence
from functools import partial
from matplotlib import pyplot
from scipy import signal
from pyl4c import pft_dominant, suppress_warnings
from pyl4c.data.fixtures import PFT, restore_bplut, restore_bplut_flat
from pyl4c.science import vpd, par, rescale_smrz, arrhenius, climatology365, soc_analytical_spinup, soc_numerical_spinup
from pyl4c.stats import linear_constraint, rmsd
from pyl4c.apps.calibration import BPLUT, GenericOptimization, cbar, report_fit_stats

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
    _driver_bounds = {'apar': (2, np.inf)}
    _metadata = {
        'tmin': {'units': 'deg K'},
        'vpd': {'units': 'Pa'},
        'smrz': {'units': '%'},
        'smsf': {'units': '%'},
        'tsoil': {'units': 'deg K'},
    }
    _required_parameters = {
        'GPP':  ['LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'],
        'RECO': ['CUE', 'tsoil', 'smsf0', 'smsf1'],
        'SOC':  ['decay_rates0', 'decay_rates1', 'decay_rates2']
    }
    _required_drivers = {
        # Tsurf = Surface skin temperature; Tmin = Minimum daily temperature
        'GPP':  ['fPAR', 'PAR', 'Tmin', 'VPD', 'SMRZ', 'FT'],
        'RECO': ['Tsoil', 'SMSF']
    }

    def __init__(self, config: str = None, pft: int = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                L4C_DIR, 'data/files/config_L4C_calibration.yaml')
        print(f'Using configuration file: {config_file}')
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        if pft is not None:
            assert pft in PFT_VALID, f'Invalid PFT: {pft}'
            self._pft = pft
        self.hdf5 = self.config['data']['file']
        self.bplut = BPLUT(
            restore_bplut(self.config['BPLUT']),
            hdf5_path = self.config['optimization']['backend'])

    def _bounds(self, init_params, bounds, model, fixed = None):
        'Defines bounds; optionally "fixes" parameters by fixing bounds'
        params = init_params
        lower = []
        upper = []
        # Then, set the bounds; for free parameters, this is what was given;
        #   for fixed parameters, this is the fixed value plus/minus some
        #   tolerance
        for i, p in enumerate(self._required_parameters[model]):
            # This is a parameter to be optimized; use default bounds
            lower.append(bounds[p][0])
            upper.append(bounds[p][1])
            if fixed is not None:
                if p in fixed:
                    if fixed[p] is not None:
                        lower.pop()
                        upper.pop()
                        lower.append(fixed[p] - 1e-3)
                        upper.append(fixed[p] + 1e-3)
        return (np.array(lower), np.array(upper))

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
        elif protocol == 'RECO':
            # Remove negative values
            return np.where(raw < 0, np.nan, raw)

    def _filter(self, raw: Sequence, size: int):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        return raw # Or, revert to the raw data

    def _get_params(self, model: str):
        # Filter the parameters to just those for the PFT of interest
        return self.bplut.flat(self._pft, self._required_parameters[model.upper()])

    def _load_gpp_data(self, filter_length):
        'Load the required datasets for GPP, for a single PFT'
        blacklist = self.config['data']['sites_blacklisted']
        with h5py.File(self.hdf5, 'r') as hdf:
            n_steps = hdf['time'].shape[0]
            sites = hdf['site_id'][:]
            if hasattr(sites[0], 'decode'):
                sites = list(map(lambda x: x.decode('utf-8'), sites))
            # Get dominant PFT
            pft_map = pft_dominant(hdf['state/PFT'][:], sites)
            # Blacklist validation sites
            pft_mask = np.logical_and(
                np.in1d(pft_map, self._pft), ~np.in1d(sites, blacklist))
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
                    .repeat(n_steps, axis = 0)
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
        drivers_flat = list()
        # For eveything other than fPAR, add a trailing axis to the flat view;
        #   this will enable datasets to line up with fPAR's 1-km subgrid
        for field in drivers.keys():
            flat = drivers[field][~np.isnan(tower_gpp)]
            drivers_flat.append(flat[:,np.newaxis] if field != 'fPAR' else flat)
        return (drivers, drivers_flat, tower_gpp, tower_gpp_flat, weights)

    def _load_reco_data(self, filter_length):
        'Load the required datasets for RECO, for a single PFT'
        blacklist = self.config['data']['sites_blacklisted']
        with h5py.File(self.hdf5, 'r') as hdf:
            n_steps = hdf['time'].shape[0]
            sites = hdf['site_id'][:]
            if hasattr(sites[0], 'decode'):
                sites = list(map(lambda x: x.decode('utf-8'), sites))
            # Get dominant PFT
            pft_map = pft_dominant(hdf['state/PFT'][:], sites)
            # Blacklist validation sites
            pft_mask = np.logical_and(
                np.in1d(pft_map, self._pft), ~np.in1d(sites, blacklist))
            drivers = dict()
            field_map = self.config['data']['fields']
            for field in self._required_drivers['RECO']:
                # Try reading the field exactly as described in config file
                if field in field_map:
                    if field_map[field] in hdf:
                        drivers[field] = hdf[field_map[field]][:,pft_mask]

            # If RMSE is used, then we want to pay attention to weighting
            weights = None
            if 'weights' in hdf.keys():
                weights = hdf['weights'][pft_mask][np.newaxis,:]\
                    .repeat(n_steps, axis = 0)
            else:
                print('WARNING - "weights" not found in HDF5 file!')
            if 'RECO' not in hdf.keys():
                with h5py.File(
                        self.config['data']['supplemental_file'], 'r') as _hdf:
                    tower_reco = _hdf['RECO'][:][:,pft_mask]
            else:
                tower_reco = hdf['RECO'][:][:,pft_mask]

        # Check that driver data do not contain NaNs
        for field in drivers.keys():
            assert not np.isnan(drivers[field]).any(),\
                f'Driver dataset "{field}" contains NaNs'
        # Clean observations, then mask out driver data where the are no
        #   observations
        tower_reco = self._filter(tower_reco, filter_length)
        tower_reco = self._clean(tower_reco, drivers, protocol = 'RECO')
        drivers = [drivers[k] for k in self._required_drivers['RECO']]
        return (drivers, tower_reco, weights)

    def _report(self, old_params, new_params, model, prec = 2):
        'Prints a report on the updated (optimized) parameters'
        labels = self._required_parameters[model.upper()]
        pad = max(len(l) for l in labels) + 1
        fmt_string = '-- {:<%d} {:>%d} [{:>%d}]' % (pad, 5 + prec, 7 + prec)
        print('%s parameters report, %s (PFT %d):' % (
            f'{model.upper()} Optimization', PFT[self._pft][0], self._pft))
        print((' {:>%d} {:>%d}' % (8 + pad + prec, 8 + prec))\
            .format('NEW', 'INITIAL'))
        for i, label in enumerate(labels):
            new = ('%%.%df' % prec) % new_params[i] if new_params[i] is not None else ''
            old = ('%%.%df' % prec) % old_params[i]
            print(fmt_string.format(('%s:' % label), new, old))

    @staticmethod
    def e_mult(params, tmin, vpd, smrz, ft):
        # Calculate E_mult based on current parameters
        f_tmin = linear_constraint(params[1], params[2])
        f_vpd  = linear_constraint(params[3], params[4], 'reversed')
        f_smrz = linear_constraint(params[5], params[6])
        f_ft   = linear_constraint(params[7], 1.0, 'binary')
        return f_tmin(tmin) * f_vpd(vpd) * f_smrz(smrz) * f_ft(ft)

    @staticmethod
    def k_mult(params, tsoil, smsf):
        # Calculate K_mult based on current parameters
        f_tsoil = partial(arrhenius, beta0 = params[1])
        f_smsf  = linear_constraint(params[2], params[3])
        return f_tsoil(tsoil) * f_smsf(smsf)

    @staticmethod
    def gpp(params, fpar, par, tmin, vpd, smrz, ft):
        # Calculate GPP based on the provided BPLUT parameters
        apar = fpar * par
        return apar * params[0] *\
            CalibrationAPI.e_mult(params, tmin, vpd, smrz, ft)

    @staticmethod
    def reco(params, tower_reco, tower_gpp, tsoil, smsf, q_rh, q_k):
        # Calculate RH as (RECO - RA) or (RECO - (faut * GPP))
        ra = ((1 - params[0]) * tower_gpp)
        rh = tower_reco - ra
        rh = np.where(rh < 0, 0, rh) # Mask out negative RH values
        kmult0 = CalibrationAPI.k_mult(params, tsoil, smsf)
        cbar0 = cbar(rh, kmult0, q_rh, q_k)
        return ra + (kmult0 * cbar0)

    def pft(self, pft):
        '''
        Sets the PFT class for the next calibration step.

        Parameters
        ----------
        pft : int
            The PFT class to use in calibration

        Returns
        -------
        CLI
        '''
        assert pft in range(1, 9), 'Unrecognized PFT class'
        self._pft = pft
        return self

    def plot_gpp(
            self, driver, filter_length = 2, coefs = None,
            xlim = None, ylim = None, alpha = 0.1, marker = '.'):
        '''
        Using the current or optimized BPLUT coefficients, plots the GPP ramp
        function for a given driver. NOTE: Values where APAR < 2.0 are not
        shown.

        Parameters
        ----------
        driver : str
            Name of the driver to plot on the horizontal axis
        filter_length : int
        coefs : list or tuple or numpy.ndarray
            (Optional) array-like, Instead of using what's in the BPLUT,
            specify the exact parameters, e.g., [tmin0, tmin1]
        xlim : list or tuple
            (Optional) A 2-element sequence: The x-axis limits
        ylim : list or tuple
            (Optional) A 2-element sequence: The x-axis limits
        alpha : float
            (Optional) The alpha value (Default: 0.1)
        marker : str
            (Optional) The marker symbol (Default: ".")
        '''
        @suppress_warnings
        def empirical_lue(apar, gpp):
            # Mask low APAR values
            lower, _ = self._driver_bounds.get('apar', (0, None))
            apar = np.where(apar < lower, np.nan, apar)
            # Calculate empirical light-use efficiency: GPP/APAR
            return np.where(apar > 0, np.divide(gpp, apar), 0)

        np.seterr(invalid = 'ignore')
        # Read in GPP and APAR data
        assert driver.lower() in ('tmin', 'vpd', 'smrz'),\
            'Requested driver "%s" cannot be plotted for GPP' % driver
        if coefs is not None:
            assert hasattr(coefs, 'index') and not hasattr(coefs, 'title'),\
            "Argument --coefs expects a list [values,] with NO spaces"
        coefs0 = [ # Original coefficients
            self.bplut[driver.lower()][i][self._pft]
            for i in (0, 1)
        ]

        # Load APAR and tower GPP data
        driver_data, _, tower_gpp, _, _ = self._load_gpp_data(filter_length)
        apar = np.nanmean(driver_data['fPAR'], axis = -1) * driver_data['PAR']
        # Based on the bounds, create an empirical ramp function that
        #   spans the range of the driver
        bounds = [
            self.config['optimization']['bounds'][f'{driver.lower()}{i}'][i]
            for i in (0, 1) # i.e., min(tmin0) and max(tmin1)
        ]
        domain = np.arange(bounds[0], bounds[1], 0.1)
        ramp_func_original = linear_constraint(
            *coefs0, 'reversed' if driver.lower() == 'vpd' else None)
        if coefs is not None:
            ramp_func = linear_constraint(
                *coefs, 'reversed' if driver.lower() == 'vpd' else None)
        if driver.lower() == 'vpd':
            x0 = driver_data['VPD']
        elif driver.lower() == 'tmin':
            x0 = driver_data['Tmin']
        elif driver.lower() == 'smrz':
            x0 = driver_data['SMRZ']

        # Update plotting parameters
        lue = empirical_lue(apar, tower_gpp)
        # Mask out negative LUE values and values with APAR<2
        pyplot.scatter(x0, np.where(
            np.logical_or(lue == 0, apar < 2), np.nan, lue),
            alpha = alpha, marker = marker)
        # Plot the original ramp function (black line)
        pyplot.plot(domain, ramp_func_original(domain) *\
            self.bplut['LUE'][:,self._pft], 'k-')
        # Optionally, plot a proposed ramp function (red line)
        if coefs is not None:
            pyplot.plot(domain, ramp_func(domain) *\
                self.bplut['LUE'][:,self._pft], 'r-')
        pyplot.xlabel('%s (%s)' % (driver, self._metadata[driver]['units']))
        pyplot.ylabel('GPP/APAR (g C MJ-1 d-1)')
        if xlim is not None:
            pyplot.xlim(xlim[0], xlim[1])
        if ylim is not None:
            pyplot.ylim(ylim[0], ylim[1])
        pyplot.title(
            '%s (PFT %d): GPP Response to "%s"' % (
                PFT[self._pft][0], self._pft, driver))
        pyplot.show()

    def plot_reco(
            self, driver, filter_length: int = 2, coefs = None, q_rh = 75,
            q_k = 50, xlim = None, ylim = None, alpha = 0.1, marker = '.'):
        '''
        Using the current or optimized BPLUT coefficients, plots the RECO ramp
        function for a given driver. The ramp function is shown on a plot of
        RH/Cbar, which is equivalent to Kmult (as Cbar is an upper quantile of
        the RH/Kmult distribution).

        Parameters
        ----------
        driver : str
            Name of the driver to plot on the horizontal axis
        filter_length : int
        coefs : list or tuple or numpy.ndarray
            (Optional) array-like, Instead of using what's in the BPLUT,
            specify the exact parameters, e.g., `[tmin0, tmin1]`
        q_rh : int
            Additional arguments to `pyl4c.apps.calibration.cbar()`
        q_k : int
            Additional arguments to `pyl4c.apps.calibration.cbar()`
        ylim : list or tuple
            (Optional) A 2-element sequence: The x-axis limits
        alpha : float
            (Optional) The alpha value (Default: 0.1)
        marker : str
            (Optional) The marker symbol (Default: ".")
        '''
        xlabels = {
            'smsf': 'Surface Soil Moisture',
            'tsoil': 'Soil Temperature'
        }
        np.seterr(invalid = 'ignore')
        assert driver.lower() in ('tsoil', 'smsf'),\
            'Requested driver "%s" cannot be plotted for RECO' % driver

        if coefs is not None:
            assert hasattr(coefs, 'index') and not hasattr(coefs, 'title'),\
            "Argument --coefs expects a list [values,] with NO spaces"
        drivers, reco, _ = self._load_reco_data(filter_length)
        _, _, gpp, _, _ = self._load_gpp_data(filter_length)
        tsoil, smsf = drivers
        # Calculate k_mult based on original parameters
        f_smsf = linear_constraint(*self.bplut['smsf'][:,self._pft])
        k_mult = f_smsf(smsf) * arrhenius(tsoil, self.bplut['tsoil'][0,self._pft])
        # Calculate RH as (RECO - RA)
        rh = reco - ((1 - self.bplut['CUE'][0,self._pft]) * gpp)
        # Set negative RH values to zero
        rh = np.where(suppress_warnings(np.less)(rh, 0), 0, rh)
        cbar0 = suppress_warnings(cbar)(rh, k_mult, q_rh, q_k)
        gpp = reco = None

        # Update plotting parameters
        pyplot.scatter( # Plot RH/Cbar against either Tsoil or SMSF
            tsoil if driver == 'tsoil' else smsf,
            suppress_warnings(np.divide)(rh, cbar0),
            alpha = alpha, marker = marker)

        if driver == 'tsoil':
            domain = np.arange(tsoil.min(), tsoil.max(), 0.1)
            pyplot.plot(domain,
                arrhenius(domain, self.bplut['tsoil'][0,self._pft]), 'k-')
        elif driver == 'smsf':
            domain = np.arange(smsf.min(), smsf.max(), 0.001)
            pyplot.plot(domain, f_smsf(domain).ravel(), 'k-')

        if coefs is not None:
            if driver == 'tsoil':
                pyplot.plot(domain, arrhenius(domain, *coefs), 'r-')
            elif driver == 'smsf':
                pyplot.plot(domain, linear_constraint(*coefs)(domain), 'r-')

        pyplot.xlabel('%s (%s)' % (
            xlabels[driver], self._metadata[driver]['units']))
        pyplot.ylabel('RH/Cbar')
        if xlim is not None:
            pyplot.xlim(xlim[0], xlim[1])
        if ylim is not None:
            pyplot.ylim(ylim[0], ylim[1])
        pyplot.title(
            '%s (PFT %d): RECO Response to "%s"' % (
                PFT[self._pft][0], self._pft,
                'SMSF' if driver.lower() == 'smsf' else 'Tsoil'))
        pyplot.show()

    def score(
            self, model: str, filter_length: int = 2,
            q_rh: int = 75, q_k: int = 50):
        '''
        Scores the current model (for a specific PFT) based on the parameters
        in the BPLUT.

        Parameters
        ----------
        model : str
            One of: "GPP" or "RECO"
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data
        '''
        gpp_drivers, gpp_drivers_flat, tower_gpp, tower_gpp_flat, weights =\
            self._load_gpp_data(filter_length)
        if model.upper() == 'GPP':
            observed = tower_gpp_flat
        elif model.upper() == 'RECO':
            reco_drivers, tower_reco, weights =\
                self._load_reco_data(filter_length)
            observed = tower_reco
        # Print the parameters table
        old_params = restore_bplut_flat(self.config['BPLUT'])
        old_params = [
            old_params[k][:,self._pft]
            for k in self._required_parameters[model.upper()]
        ]
        params = self._get_params(model.upper())
        self._report(old_params, params, model.upper())
        # Get goodness-of-fit statistics
        if model.upper() == 'GPP':
            predicted = self.gpp(params, *gpp_drivers_flat).mean(axis = -1)
        elif model.upper() == 'RECO':
            predicted = self.reco(
                params, tower_reco, tower_gpp, *reco_drivers,
                q_rh = q_rh, q_k = q_k)
        r2, rmse_score, ub_rmse, bias = report_fit_stats(
            observed, predicted, weights, verbose = False)
        print(f'{model.upper()} model goodness-of-fit statistics:')
        print(('-- {:<13} {:>5}').format('R-squared:', '%.3f' % r2))
        print(('-- {:<13} {:>5}').format('RMSE:', '%.2f' % rmse_score))
        print(('-- {:<13} {:>5}').format('Unbised RMSE:', '%.2f' % ub_rmse))
        print(('-- {:<13} {:>5}').format('Bias:', '%.2f' % bias))

    def set(self, parameter, value):
        '''
        Sets the named parameter to the given value for the specified PFT
        class. This updates the initial parameters, affecting any subsequent
        optimization.

        Parameters
        ----------
        parameter : str
            Name of the parameter to bet set
        value : int or float
            Value of the named parameter to be set

        Returns
        -------
        CLI
        '''
        # Update the BPLUT in memory but NOT the file BPLUT (this is temporary)
        self.bplut.update(self._pft, (value,), (parameter,), flush = False)
        return self

    def tune_gpp(
            self, filter_length: int = 2, optimize: bool = True,
            use_nlopt: bool = True):
        '''
        Run the L4C GPP calibration.

        - For GPP data: Removes observations where GPP < 0 or where APAR is
            < 0.1 MJ m-2 day-1

        Parameters
        ----------
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data
        optimize : bool
            True to run the optimization (Default) or False if you just want
            the goodness-of-fit to be reported
        use_nlopt : bool
            True to use `nlopt` for optimization (Default)
        '''
        def residuals(params, drivers, observed, weights):
            gpp0 = self.gpp(params, *drivers).mean(axis = -1)
            diff = np.subtract(observed, gpp0)
            # Objective function: Difference between tower GPP and L4C GPP,
            #   multiplied by the tower weights
            return (weights * diff)[~np.isnan(diff)]

        init_params = self._get_params('GPP')
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']

        print('Loading driver datasets...')
        drivers, drivers_flat, tower_gpp, tower_gpp_flat, weights =\
            self._load_gpp_data(filter_length)
        print(f'NOTE: Counts of (days, towers) are: {tower_gpp.shape}')

        # Configure the optimization, get bounds for the parameter search
        fixed = None
        if self.config['optimization']['fixed'] is not None:
            fixed = dict([ # Get any fixed parameters as {param: fixed_value}
                (k, v[self._pft])
                for k, v in self.config['optimization']['fixed'].items()
            ])
        step_size_global = [
            self.config['optimization']['step_size'][p]
            for p in self._required_parameters['GPP']
        ]
        bounds_dict = self.config['optimization']['bounds']
        bounds = self._bounds(init_params, bounds_dict, 'GPP', fixed)
        trials = self.config['optimization']['trials']
        params = [] # Optimized parameters
        params0 = [] # Initial (random) parameters
        scores = []
        # Will be a (100, P) space, where P is the number of parameters
        param_space = np.linspace(bounds[0], bounds[1], 100)
        for t in range(0, trials):
            # If multiple trials, randomize the initial parameter values
            #   and score the model in each trial
            if trials > 1:
                p = param_space.shape[1] # Number of parameters
                idx = np.random.randint(0, param_space.shape[0], p)
                init_params = param_space[idx,np.arange(0, p)]
                params0.append(init_params)
            # If we're optimizing (with any library), define the bounds and
            #   the objective function
            if optimize:
                bounds = self._bounds(init_params, bounds_dict, 'GPP', fixed)
                # Set initial value to a fixed value if specified
                if fixed is not None:
                    for key, value in fixed.items():
                        if value is not None and key in self._required_parameters['GPP']:
                            init_params[self._required_parameters['GPP'].index(key)] = value
                objective = partial(
                    residuals, drivers = drivers_flat,
                    observed = tower_gpp_flat, weights = weights)
            # Apply constrained, non-linear least-squares optimization, using
            #   either SciPy or NLOPT
            if optimize and not use_nlopt:
                solution = solve_least_squares(
                    objective, init_params,
                    labels = self.required_parameters['GPP'],
                    bounds = bounds, loss = 'arctan')
                fitted = solution.x.tolist()
                print(solution.message)
            elif optimize and use_nlopt:
                opt = GenericOptimization(
                    objective, bounds, step_size = step_size_global)
                fitted = opt.solve(init_params)
            else:
                fitted = [None for i in range(0, len(init_params))]
            # Record the found solution and its goodness-of-fit score
            params.append(fitted)
            predicted = self.gpp(
                fitted if optimize else init_params,
                *drivers_flat).mean(axis = -1)
            _, rmse_score, _, _ = report_fit_stats(
                tower_gpp_flat, predicted, weights, verbose = False)
            print('[%s/%s] RMSE score of last trial: %.3f' % (
                str(t + 1).zfill(2), str(trials).zfill(2), rmse_score))
            scores.append(rmse_score)

        # Select the fit params with the best score
        if trials > 1:
            fitted = params[np.argmin(scores)]
            init_params = params0[np.argmin(scores)]
        # Generate and print a report, update the BPLUT parameters
        self._report(init_params, fitted, 'GPP')
        if optimize:
            self.bplut.update(
                self._pft, fitted, self._required_parameters['GPP'])

    def tune_reco(
            self, filter_length: int = 2, q_rh: int = 75, q_k: int = 50,
            optimize: bool = True, use_nlopt: bool = True):
        '''
        Optimizes RECO. The 9-km mean L4C RECO is fit to the tower-observed
        RECO using constrained, non-linear least-squares optimization.
        Considerations:

        - Negative RH values (i.e., NPP > RECO) are set to zero.

        Parameters
        ----------
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data
        optimize : bool
            False to only report parameters and their fit statistics instead
            of optimizing (Default: True)
        use_nlopt : bool
            True to use the nlopt library for optimization (Default: True)
        '''
        def residuals(
                params, drivers, observed_reco, observed_gpp, weights,
                q_rh, q_k):
            # Objective function: Difference between tower RECO and L4C RECO
            reco0 = self.reco(
                params, observed_reco, observed_gpp, *drivers, q_rh, q_k)
            diff = np.subtract(observed_reco, reco0)
            missing = np.logical_or(np.isnan(observed_reco), np.isnan(reco0))
            # Multiply by the tower weights
            return (weights * diff)[~missing]

        assert q_rh >= 0 and q_rh <= 100 and q_k >= 0 and q_k <= 100,\
            'Invalid setting for "q_rh" or "q_k" parameters'

        init_params = self._get_params('RECO')
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']

        print('Loading driver datasets...')
        _, _, tower_gpp, _, _ = self._load_gpp_data(filter_length)
        drivers, tower_reco, weights = self._load_reco_data(filter_length)
        print(f'NOTE: Counts of (days, towers) are: {tower_reco.shape}')

        # Configure the optimization, get bounds for the parameter search
        fixed = None
        if self.config['optimization']['fixed'] is not None:
            fixed = dict([ # Get any fixed parameters as {param: fixed_value}
                (k, v[self._pft])
                for k, v in self.config['optimization']['fixed'].items()
            ])
        step_size_global = [
            self.config['optimization']['step_size'][p]
            for p in self._required_parameters['RECO']
        ]
        bounds_dict = self.config['optimization']['bounds']
        bounds = self._bounds(init_params, bounds_dict, 'RECO', fixed)
        trials = self.config['optimization']['trials']
        params = [] # Optimized parameters
        params0 = [] # Initial (random) parameters
        scores = []
        # Will be a (100, P) space, where P is the number of parameters
        param_space = np.linspace(bounds[0], bounds[1], 100)
        for t in range(0, trials):
            # If multiple trials, randomize the initial parameter values
            #   and score the model in each trial
            if trials > 1:
                p = param_space.shape[1] # Number of parameters
                idx = np.random.randint(0, param_space.shape[0], p)
                init_params = param_space[idx,np.arange(0, p)]
                params0.append(init_params)
            # If we're optimizing (with any library), define the bounds and
            #   the objective function
            if optimize:
                bounds = self._bounds(init_params, bounds_dict, 'RECO', fixed)
                # Set initial value to a fixed value if specified
                for key, value in fixed.items():
                    if value is not None and key in self._required_parameters['RECO']:
                        init_params[self._required_parameters['RECO'].index(key)] = value
                objective = partial(
                    residuals, drivers = drivers, weights = weights,
                    observed_gpp = tower_gpp, observed_reco = tower_reco,
                    q_rh = q_rh, q_k = q_k)
            # Apply constrained, non-linear least-squares optimization, using
            #   either SciPy or NLOPT
            if optimize and not use_nlopt:
                solution = solve_least_squares(
                    objective, init_params,
                    labels = self.required_parameters['RECO'],
                    bounds = bounds, loss = 'arctan')
                fitted = solution.x.tolist()
                print(solution.message)
            elif optimize and use_nlopt:
                opt = GenericOptimization(
                    objective, bounds, step_size = step_size_global)
                fitted = opt.solve(init_params)
            else:
                fitted = [None for i in range(0, len(init_params))]
            # Record the found solution and its goodness-of-fit score
            params.append(fitted)
            predicted = self.reco(
                fitted if optimize else init_params, tower_reco, tower_gpp,
                *drivers, q_rh = q_rh, q_k = q_k)
            _, rmse_score, _, _ = report_fit_stats(
                tower_reco, predicted, weights, verbose = False)
            print('[%s/%s] RMSE score of last trial: %.3f' % (
                str(t + 1).zfill(2), str(trials).zfill(2), rmse_score))
            scores.append(rmse_score)

        # Select the fit params with the best score
        if trials > 1:
            fitted = params[np.argmin(scores)]
            init_params = params0[np.argmin(scores)]
        # Generate and print a report, update the BPLUT parameters
        self._report(init_params, fitted, 'RECO')
        if optimize:
            self.bplut.update(
                self._pft, fitted, self._required_parameters['RECO'])

    def tune_soc(self, filter_length: int = 2):
        '''
        Starts interactive calibration procedure for the soil organic carbon
        (SOC) decay parameters for a given PFT.

        Parameters
        ----------
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data
        '''
        print('Loading driver datasets...')
        drivers_for_gpp, _, tower_gpp, _, _ = self._load_gpp_data(
            filter_length)
        drivers_for_reco, tower_reco, weights = self._load_reco_data(
            filter_length)

        # Load the date series (for computing a climatology) as well as the
        #   map of dominant PFTs (for selecting IGBP sites)
        blacklist = self.config['data']['sites_blacklisted']
        with h5py.File(self.hdf5, 'r') as hdf:
            dates = [datetime.date(*d) for d in hdf['time'][:].tolist()]
            sites = hdf['site_id'][:]
            if hasattr(sites[0], 'decode'):
                sites = list(map(lambda x: x.decode('utf-8'), sites))
            pft_map = pft_dominant(hdf['state/PFT'][:], sites)
            pft_mask = np.logical_and(
                np.in1d(pft_map, self._pft), ~np.in1d(sites, blacklist))

        # Load IGBP SOC pit measurements, convert from [kg C m-2] to [g C m-2]
        with h5py.File(self.config['data']['supplemental_file'], 'r') as hdf:
            igbp_soc = hdf['SOC'][:]
        igbp_soc = igbp_soc[pft_mask]
        igbp_soc[igbp_soc < 0] = np.nan

        # Calculate GPP based on the updated parameters
        init_params = restore_bplut_flat(self.config['BPLUT'])
        params_gpp = self._get_params('GPP')
        params_reco = self._get_params('RECO')
        gpp = self.gpp(params_gpp, *[
            drivers_for_gpp[d] if d != 'fPAR' else np.nanmean(drivers_for_gpp[d], axis = -1)
            for d in self._required_drivers['GPP']
        ])

        # Calculate a 365-day climatology of NPP
        cue = params_reco[self._required_parameters['RECO'].index('CUE')]
        npp = gpp * cue
        # Make the time axis (currently 0) be the trailing axis
        npp_clim = climatology365(npp.swapaxes(0, 1), dates)
        # Calculate litterfall
        litter = npp_clim.sum(axis = 0) / 365
        # Calculate a 365-day climatology of Kmult
        kmult = self.k_mult(params_reco, *drivers_for_reco)
        # Make the time axis (currently 0) be the trailing axis
        kmult_clim = climatology365(kmult.swapaxes(0, 1), dates)
        sigma = npp_clim.sum(axis = 0) / kmult_clim.sum(axis = 0)

        # Inferred steady-state storage
        fmet = init_params['f_metabolic'][:,self._pft][0]
        fstr = init_params['f_structural'][:,self._pft][0]
        decay_rates = self._get_params('SOC')
        decay_rates = decay_rates[:,np.newaxis]
        # Begin user-interaction loop to manually calibrate decay rates
        prev = None
        while True:
            init_soc = soc_analytical_spinup(
                litter, kmult_clim, fmet, fstr, decay_rates)
            soc, _ = soc_numerical_spinup(
                np.stack(init_soc), litter, kmult_clim, fmet, fstr, decay_rates,
                verbose = True)
            soc = np.stack(soc).sum(axis = 0)
            _, ax = pyplot.subplots(figsize = (6,6))
            ax.plot([0, 1], [0, 1], transform = ax.transAxes, linestyle = 'dotted')
            if prev is not None:
                pyplot.plot(igbp_soc / 1e3, prev / 1e3, 'o', c = 'gray', alpha = 0.3)
            try:
                pyplot.plot(igbp_soc / 1e3, soc / 1e3, 'o', alpha = 0.6)
            except:
                import ipdb
                ipdb.set_trace()#FIXME
            pyplot.xlabel('IGBP SOC (kg m$^{-2}$)')
            pyplot.ylabel('Modeled Equilibrium SOC (kg m$^{-2}$)')
            pyplot.show()
            # Calculate correlation coefficient
            mask = np.isnan(igbp_soc)
            r = np.corrcoef(igbp_soc[~mask], soc[~mask])[0,1]
            rmse = rmsd(igbp_soc[~mask], soc[~mask])
            print(f'Current metabolic rate (r={r.round(3)}, RMSE={round(rmse, 1)}):')
            print('%.5f\n' % decay_rates[0])
            proposal = input('New metabolic rate [Q to quit]:\n')
            if proposal == 'Q':
                break
            value = float(proposal)
            # NOTE: The "structural" and "recalcitrant" pool decay rates
            #   here should be the actual decay rates, i.e., the "metabolic"
            #   rate scaled by fixed constants
            decay_rates = np.array([
                value, value * 0.4, value * 0.0093
            ]).reshape((3, 1))
            prev = soc.copy()
        print(f'Updated BPLUT decay rates for PFT={self._pft}')
        self.bplut.update(
            self._pft, decay_rates.ravel(),
            ['decay_rates0', 'decay_rates1', 'decay_rates2'])


if __name__ == '__main__':
    import fire
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
