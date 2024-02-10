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
from pyl4c.science import vpd, par, rescale_smrz
from pyl4c.stats import linear_constraint, rmsd
from pyl4c.apps.calibration import BPLUT, GenericOptimization, report_fit_stats

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
    }
    _required_drivers = {
        # Tsurf = Surface skin temperature; Tmin = Minimum daily temperature
        'GPP':  ['fPAR', 'PAR', 'Tmin', 'VPD', 'SMRZ', 'FT'],
        'RECO': ['SMSF', 'Tsoil']
    }

    def __init__(self, config = None, pft = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                L4C_DIR, 'data/files/config_L4C_calibration.yaml')
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
        if fixed is not None:
            params = [ # If the given parameter is in "fixed", restrict bounds
                None if self._required_parameters[model][i] in fixed else init_params[i]
                for i in range(0, len(self._required_parameters[model]))
            ]
        lower = []
        upper = []
        for i, p in enumerate(self._required_parameters[model]):
            # This is a parameter to be optimized; use default bounds
            if p is not None:
                lower.append(bounds[p][0])
                upper.append(bounds[p][1])
            else:
                lower.append(init_params[i] - 1e-3)
                upper.append(init_params[i] + 1e-3)
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

    def _filter(self, raw: Sequence, size: int):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        return raw # Or, revert to the raw data

    def _get_params(self, model):
        # Filter the parameters to just those for the PFT of interest
        params_dict = restore_bplut_flat(self.config['BPLUT'])
        return dict([
            (k, params_dict[k].ravel()[self._pft])
            for k in self._required_parameters[model]
        ])

    def _load_gpp_data(self, filter_length):
        'Load the required datasets for GPP, for a single PFT'
        blacklist = self.config['data']['sites_blacklisted']
        with h5py.File(self.hdf5, 'r') as hdf:
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
        drivers_flat = list()
        # For eveything other than fPAR, add a trailing axis to the flat view;
        #   this will enable datasets to line up with fPAR's 1-km subgrid
        for field in drivers.keys():
            flat = drivers[field][~np.isnan(tower_gpp)]
            drivers_flat.append(flat[:,np.newaxis] if field != 'fPAR' else flat)
        return (drivers, drivers_flat, tower_gpp, tower_gpp_flat, weights)

    def _report(self, old_params, new_params, labels, title, prec = 2):
        'Prints a report on the updated (optimized) parameters'
        pad = max(len(l) for l in labels) + 1
        fmt_string = '-- {:<%d} {:>%d} [{:>%d}]' % (pad, 5 + prec, 7 + prec)
        print('%s parameters report, %s (PFT %d):' % (
            title, PFT[self._pft][0], self._pft))
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
    def gpp(params, fpar, par, tmin, vpd, smrz, ft):
        # Calculate GPP based on the provided BPLUT parameters
        apar = fpar * par
        return apar * params[0] *\
            CalibrationAPI.e_mult(params, tmin, vpd, smrz, ft)

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
        ramp_func_original = linear_constraint(*coefs0)
        if coefs is not None:
            ramp_func = linear_constraint(*coefs)
        if driver.lower() == 'vpd':
            x0 = driver_data['VPD']
            ramp_func_original = linear_constraint(*coefs0, 'reversed')
            ramp_func = linear_constraint(*coefs, 'reversed')
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
            use_nlopt: bool = True, ipdb: bool = False, save_fig: bool = False):
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
        ipdb : bool
            True to drop the user into an ipdb prompt, prior to and instead of
            running calibration
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        '''
        def residuals(params, drivers, observed, weights):
            gpp0 = self.gpp(params, *drivers).mean(axis = -1)
            diff = np.subtract(observed, gpp0)
            # Objective function: Difference between tower GPP and L4C GPP,
            #   multiplied by the tower weights
            return (weights * diff)[~np.isnan(diff)]

        params_dict = self._get_params('GPP')
        init_params = [
            params_dict[p] for p in self._required_parameters['GPP']
        ]
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']
        objective = self.config['optimization']['objective'].lower()

        print('Loading driver datasets...')
        drivers, drivers_flat, tower_gpp, tower_gpp_flat, weights =\
            self._load_gpp_data(filter_length)

        # Configure the optimization, get bounds for the parameter search
        bounds_dict = self.config['optimization']['bounds']
        fixed = self.config['optimization']['fixed']
        trials = self.config['optimization']['trials']
        step_size_global = [
            self.config['optimization']['step_size'][p]
            for p in self._required_parameters['GPP']
        ]
        bounds = self._bounds(init_params, bounds_dict, 'GPP', fixed)
        params = [] # Optimized parameters
        params0 = [] # Initial (random) parameters
        scores = []
        param_space = np.linspace(bounds[0], bounds[1], 100)
        for t in range(0, trials):
            # If multiple trials, randomize the initial parameter values
            #   and score the model in each trial
            if trials > 1:
                p = param_space.shape[1] # Number of parameters
                idx = np.random.randint(0, param_space.shape[0], p)
                init_params = param_space[idx,np.arange(0, p)]
                params0.append(init_params)
            if optimize:
                objective = partial(
                    residuals, drivers = drivers_flat,
                    observed = tower_gpp_flat, weights = weights)
            if optimize and not use_nlopt:
                # Apply constrained, non-linear least-squares optimization
                bounds = self._bounds(init_params, bounds_dict, 'GPP', fixed)
                solution = solve_least_squares(
                    objective, init_params,
                    labels = self.required_parameters['GPP'],
                    bounds = bounds, loss = 'arctan')
                fitted = solution.x.tolist()
                print(solution.message)
            elif optimize and use_nlopt:
                opt = GenericOptimization(
                    objective, bounds,step_size = step_size_global)
                fitted = opt.solve(init_params)
            else:
                fitted = [None for i in range(0, len(init_params))]
            # Record the found solution and its goodness-of-fit score
            params.append(fitted)
            _, rmse_score, _, _ = report_fit_stats(
                tower_gpp_flat,
                self.gpp(fitted if optimize else init_params, *drivers_flat)\
                    .mean(axis = -1), weights, verbose = False)
            print('[%s/%s] RMSE score of last trial: %.3f' % (
                str(t + 1).zfill(2), str(trials).zfill(2), rmse_score))
            scores.append(rmse_score)

        # Select the fit params with the best score
        if trials > 1:
            fitted = params[np.argmin(scores)]
            init_params = params0[np.argmin(scores)]
        # Generate and print a report, update the BPLUT parameters
        self._report(
            init_params, fitted, self._required_parameters['GPP'],
            'GPP Optimization')
        if optimize:
            self.bplut.update(
                self._pft, fitted, self._required_parameters['GPP'])



if __name__ == '__main__':
    import fire
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
