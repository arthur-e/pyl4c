'''
Extensions to SMAP L4C (and calibration) to support soil respiration
enhancements related to improved soil hydrology modeling.
'''

import os
import pickle
import warnings
import h5py
import numpy as np
import matplotlib
from functools import partial
from matplotlib import pyplot
from pyl4c import suppress_warnings
from pyl4c.apps.calibration.main import CLI, CONFIG
from pyl4c.science import arrhenius
from pyl4c.stats import linear_constraint
from pyl4c.apps.calibration import GenericOptimization, BPLUT, report_fit_stats, solve_least_squares
from pyl4c.data.fixtures import restore_bplut

# All DEPTHS must be positive
DEPTHS = np.array((0.05, 0.15, 0.35, 0.75, 1.5, 3.0))\
    .reshape((6,1)) # meters
# Constrained optimization bounds
OPT_BOUNDS = {
    'reco_z': ( # CUE, tsoil, smsf0, smsf1, k_depth_decay
        np.array((0.2,   1,  -30,  25, 0.3)),
        np.array((0.7, 800, 24.9, 100, 1.0))),
    'reco_z_power': ( # CUE, tsoil, smsf0, smsf1, z_tau_a, z_tau_b
        np.array((0.2,   1,  -30,  25, 0.01, 0.01)),
        np.array((0.7, 800, 24.9, 100, 1.00, 1.00))),
    # After Davidson et al. (2012)...
    #   Median d_gas in completely dry soil conditions (soil VWC < 5th
    #       percentile): 3.82
    'reco_o2_limit': ( # CUE, tsoil, smsf0, smsf1, k_depth_decay, d_gas, km_oxy
        np.array((0.2,   1,  -30,  25, 0.05, 3, 0.01)),
        np.array((0.7, 800, 24.9, 100, 1.50, 5, 0.15))),
}
NEW_PARAMETERS = ('k_depth_decay', 'd_gas', 'km_oxy', 'z_tau_a', 'z_tau_b')
L4C_PARAMETERS = list(BPLUT._labels)
L4C_PARAMETERS.extend(NEW_PARAMETERS)


class StratifiedSoilCalibrationCLI(CLI):
    '''
    Command line interface for calibrating L4C with a vertically stratified
    soil organic carbon (SOC) model.

    Get started by creating a scratch dataset:

        python hydrology.py setup

    Optionally, filter the tower GPP and/or RECO time series:

        python hydrology.py filter-all gpp <window_size>
        python hydrology.py filter-all reco <window_size>

    To optimize the RECO parameters for the model with vertically resolved
    soil organic carbon, specify a PFT class:

        python hydrology.py pft <pft> tune-reco

    To optimize the RECO parameters for the vertically resolved model that
    also includes an O2 diffusion limitation:

        python hydrology.py pft <pft> tune-reco-o2-limit
    '''
    _model_name = 'reco_z'
    _parameters = {
        'gpp': (
            'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'),
        'reco_z': (
            'CUE', 'tsoil', 'smsf0', 'smsf1', 'k_depth_decay'),
        'reco_z_power': (
            'CUE', 'tsoil', 'smsf0', 'smsf1', 'z_tau_a', 'z_tau_b'),
        'reco_o2_limit': (
            'CUE', 'tsoil', 'smsf0', 'smsf1', 'k_depth_decay', 'd_gas', 'km_oxy'),
    }
    _path_to_temp_profile = '/home/arthur/Downloads/L4C_experiments/L4C-Phenology/L4_SM_gph_NRv8-3_profile_at_356_tower_sites.h5'
    _path_to_sm_profile = '/home/arthur/Downloads/L4C_experiments/L4C-Phenology/L4_C_NRv8-3_soil_moisture_profiles_simulated_at_356_tower_sites.h5'

    def __init__(
            self, config = CONFIG, pft = None, start = None, end = None,
            debug = True, use_legacy_pft = True, n_layers = 6):
        super().__init__(
            config = config, pft = pft, start = start, end = end,
            debug = debug, use_legacy_pft = use_legacy_pft)
        self.depths = DEPTHS[0:n_layers]
        print('Working with layer depths: %s' % ', '.join(map(lambda v: '%.3f' % v, self.depths)))
        self.n_layers = n_layers
        # (Re-)creates the BPLUT store using the correct (expanded) list of
        #   parameter labels
        self._init_bplut(labels = L4C_PARAMETERS)

    @suppress_warnings
    def _configure(self, q_rh, q_k, fixed, model = 'reco_z'):
        'Loads driver data, sets starting parameters for RECO calibration'
        assert self._is_setup, 'Must run setup first'
        assert q_rh >= 0 and q_rh <= 100 and q_k >= 0 and q_k <= 100,\
            'Invalid setting for "q_rh" or "q_k" parameters'
        params = self._parameters[model]
        if fixed is not None:
            assert all(p in params for p in fixed),\
                'Arguments to "fixed" should be in: [%s]' % ', '.join(params)
        init_params = self.bplut.flat(self._pft, params)
        # Read in data, with optional subsetting of the time axis
        t0 = self._time_start if self._time_start is not None else 0
        t1 = self._time_end if self._time_end is not None else self._nsteps
        self._drivers = []
        # Open the soil moisture, temperature driver datasets
        with h5py.File(self._path_to_temp_profile, 'r') as hdf:
            # Calculate extent of soil layers, given bedrock depth
            bedrock = hdf['LAND_MODEL_CONSTANTS/depth_to_bedrock_m'][self._sites]
            self._layer_mask = self.depths < bedrock
            self._porosity = hdf['LAND_MODEL_CONSTANTS/porosity'][self._sites]
            temp_profile = []
            temp_profile.append(hdf['L4SM_DAILY_MEAN/surface_temp'][t0:t1,self._sites])
            for i in range(1, self.n_layers):
                temp_profile.append(
                    hdf['L4SM_DAILY_MEAN/soil_temp_layer%d' % i][t0:t1,self._sites])
            temp_profile = np.stack(temp_profile)
            self._drivers.append(temp_profile)
        # Convert soil VWC to wetness (%)
        if os.path.basename(self._path_to_sm_profile).split('.').pop() == 'h5':
            with h5py.File(self._path_to_sm_profile, 'r') as hdf:
                soil_m = 100 * np.divide(
                    hdf['soil_moisture_vwc'][:,t0:t1,self._sites], self._porosity)
                # Clip f(SM) response, as wetness values might be unrealistic
                #   given problems in ice-filled soil layers
                soil_m[soil_m > 100] = 100
                self._drivers.append(soil_m)
                # Add soil moisture in VWC
                if model == 'reco_o2_limit':
                    self._drivers.append(hdf['soil_moisture_vwc'][:,t0:t1,self._sites])
        elif os.path.basename(self._path_to_sm_profile).split('.').pop() == 'pickle':
            with open(self._path_to_sm_profile, 'rb') as file:
                profiles = pickle.load(file)
                soil_m = 100 * np.divide(
                    profiles[:,t0:t1,self._sites], self._porosity)
                # Clip f(SM) response, as wetness values might be unrealistic
                #   given problems in ice-filled soil layers
                soil_m[soil_m > 100] = 100
                self._drivers.append(soil_m)
                # Add soil moisture in VWC
                if model == 'reco_o2_limit':
                    self._drivers.append(profiles[:,t0:t1,self._sites])
        # Read in the tower flux data and site weights
        with h5py.File(self._path_to_scratch, 'r') as hdf:
            self._gpp_tower = hdf['tower/GPP'][t0:t1,self._sites,:].mean(axis = 2)
            self._reco_tower = hdf['tower/RECO'][t0:t1,self._sites,:].mean(axis = 2)
            self._site_weights = hdf['site_weights'][:,self._sites]
        return init_params

    def _concentration_O2(self, d_gas, soil_vwc):
        air_frac_O2 = 0.2095 # Liters of O2 per liter of air (20.95%)
        return d_gas * air_frac_O2 * np.power(self._porosity - soil_vwc, 4/3)

    def _k_mult(self, params):
        'Calculate K_mult based on current parameters'
        tsoil, sm = self._drivers
        # Note that f_z() is NOT included here, because we do not want
        #   cbar() to decline with depth
        f_tsoil = partial(arrhenius, beta0 = params[1])
        f_sm = linear_constraint(params[2], params[3])
        return f_tsoil(tsoil) * f_sm(sm)

    @suppress_warnings
    def _reco(self, params, q_rh, q_k):
        'Modeled ecosystem respiration (RECO) based on current parameters'
        # Calculate RH as (RECO - RA) or (RECO - (faut * GPP));
        ra = ((1 - params[0]) * self._gpp_tower)
        rh0 = self._reco_tower - ra
        rh0 = np.where(rh0 < 0, 0, rh0) # Mask out negative RH values
        # Compute Cbar with globals "q_rh" and "q_k"
        kmult0 = self._k_mult(params)
        cbar0 = cbar(rh0, kmult0, q_rh, q_k)
        # Extinction rate of heterotrophic respiration with depth, due to
        #   factors OTHER THAN temperature, moisture (Koven et al. 2013)
        f_z = np.exp(-np.abs(self.depths) / params[4]) *\
            np.ones(cbar0.shape)
        # Set RH from layers below the bedrock depth to zero
        rh = (kmult0 * f_z[:,None,:] * cbar0[:,None,:]).swapaxes(1, 2)
        rh[~self._layer_mask] = 0
        reco0 = ra + rh.swapaxes(1, 2).sum(axis = 0)
        return reco0

    def _tune(
            self, fit, residuals, init_params, fixed_params, step_sizes, trials,
            optimize, nlopt):
        '''
        Runs the optimization.

        Parameters
        ----------
        fit : function
            The function that returns fit values, given parameters
        residuals : function
            The function that returns residuals, given parameters
        init_params : tuple or list or numpy.ndarray
        fixed_params : tuple or list or numpy.ndarray
        step_sizes : tuple or list or numpy.ndarray
        trials : int
        optimize : bool
        nlopt : bool
        '''
        # Get bounds for the parameter search
        bounds = self._bounds(
            init_params, self._model_name, fixed_params, bounds = OPT_BOUNDS)
        params = []
        params0 = []
        scores = []
        param_space = np.linspace(bounds[0], bounds[1], 100)
        assert not np.isnan(init_params).any(),\
            'One or more NaNs were provided as "init_params"'
        for t in range(0, trials):
            # If multiple trials, randomize the initial parameter values
            #   and score the model in each trial
            if optimize and trials > 1:
                p = param_space.shape[1] # Number of parameters
                idx = np.random.randint(0, param_space.shape[0], p)
                init_params = param_space[idx,np.arange(0, p)]
                params0.append(init_params)
            if optimize and not nlopt:
                # Apply constrained, non-linear least-squares optimization
                #   NOTE: arctan loss function doesn't work well here
                if 'loss' in kwargs.keys():
                    print('NOTE: Overriding "loss" function specification')
                kwargs.update({'loss': 'linear'})
                solution = solve_least_squares(
                    residuals, init_params,
                    labels = self._parameters[self._model_name],
                    bounds = bounds, **kwargs)
                fitted = solution.x.tolist()
                message = solution.message
            elif optimize and nlopt:
                opt = GenericOptimization(
                    residuals, bounds, step_size = step_sizes)
                try:
                    fitted = opt.solve(init_params)
                except RuntimeError:
                    params.append(None)
                    scores.append(np.inf)
                    print('Error in objective function; restarting...')
                    continue # Try again!
                message = 'Success'
            else:
                fitted = [None for i in range(0, len(init_params))]
                break # Do not iterate through trials if not optimizing
            # Record the found solution and its goodness-of-fit score
            params.append(fitted)
            pred = fit(fitted if optimize else init_params)
            _, rmse_score, _, _ = self._report_fit(
                self._reco_tower, pred, self._site_weights, verbose = False)
            print('[%s/%s] RMSE score of last trial: %.3f' % (
                str(t + 1).zfill(2), str(trials).zfill(2), rmse_score))
            scores.append(rmse_score)

        # Select the fit params with the best score
        if optimize and trials > 1:
            fitted = params[np.argmin(scores)]
            init_params = params0[np.argmin(scores)]
        # Generate and print a report, update the BPLUT parameters
        self._report(
            init_params, fitted, self._parameters[self._model_name],
            'RECO Optimization')
        pred = fit(fitted if optimize else init_params)
        self._report_fit(self._reco_tower, pred, self._site_weights)
        if optimize:
            user_prompt = input('Update parameters for PFT=%d? [Y/n] ' % self._pft)
            do_write = user_prompt == 'Y'
            if do_write:
                print('Updating parameters for PFT=%d...' % self._pft)
                self.bplut.update(self._pft, fitted,
                self._parameters[self._model_name])

    def plot_reco(
            self, driver, model = 'reco_z', q_rh = 75, q_k = 50,
            by_depth = True, ylim = None, **kwargs):
        '''
        Plots both the soil moisture (wetness) ramp function and the O2
        diffusion limitation curve (a function of soil volumetric water
        content). As the model is vertically stratified but the EC flux tower
        data have a single value for each site-day observation, we take the
        mean soil moisture and mean Cbar values; hence, the plot shows the
        predicted response to environmental conditions for the "average" soil
        layer.

        Parameters
        ----------
        driver : str
            Name of environmental driver to plot
        model : str
            Name of the RECO model to use
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
        by_depth : bool
            True to plot the RH/Cbar quantity separately for each soil layer
        ylim : tuple or list or None
            Sequence of 2 values, the lower and upper limits for the vertical
            axis of the plot
        '''
        palette = ['#6e016b', '#88419d', '#8c6bb1', '#8c96c6', '#9ebcda']
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rc('font', **{
            'weight' : 'bold',
            'size'   : 14
        })
        params = self._configure(q_rh, q_k, None, model = model)
        assert not np.isnan(params).any(),\
            'Some required parameters are undefined; run calibration first'
        if model == 'reco_z':
            tsoil, sm = self._drivers
            f_sm = linear_constraint(*self.bplut['smsf'][:,self._pft])
            k_mult = f_sm(sm) * arrhenius(tsoil, self.bplut['tsoil'][0,self._pft])
        else:
            tsoil, sm, vwc = self._drivers
            f_sm = linear_constraint(*self.bplut['smsf'][:,self._pft])
            conc_O2 = self._concentration_O2(
                self._parameters[self._model_name].index('d_gas'), vwc)
            mm_O2 = conc_O2 / (params[-1] + conc_O2) # km_oxy param
            k_mult = arrhenius(tsoil, self.bplut['tsoil'][0,self._pft]) *\
                np.nanmin(np.stack((f_sm(sm), mm_O2)), axis = 0)

        # Calculate RH as (RECO - RA)
        rh0 = self._reco_tower - ((1 - self.bplut['CUE'][0,self._pft]) * self._gpp_tower)
        # Set negative RH values to zero
        rh0 = np.where(suppress_warnings(np.less)(rh0, 0), 0, rh0)
        cbar0 = suppress_warnings(cbar)(rh0, k_mult, q_rh, q_k)
        # Update plotting parameters
        kwargs.setdefault('alpha', 0.08)
        kwargs.setdefault('marker', '.')
        div = suppress_warnings(np.divide) # Quiet divide
        if driver == 'vwc':
            if by_depth:
                for z in range(0, vwc.shape[0]):
                    points = pyplot.scatter(
                        vwc[z], np.where(
                            cbar0[z] == 0, np.nan, div(rh0, cbar0[z])),
                        color = palette[z], label = '%.2f m' % self.depths[z],
                        **kwargs)
                    points.set_alpha(1)
            else:
                pyplot.scatter(
                    vwc.mean(axis = 0), div(rh0, np.nanmean(cbar0, axis = 0)),
                    **kwargs)
            ramp = self._ramp(sm, 'smsf')
            domain, y = ramp
            pyplot.plot(domain / 100, y, 'k-', label = 'Substrate Limit')
            conc_O2 = params[5] * 0.2095 *\
                np.power(0.8 - np.multiply(domain / 100, 0.8), 4/3)
            pyplot.plot(
                domain / 100, conc_O2 / (params[6] + conc_O2), 'r-',
                label = 'O2 Diffusion Limit')
            pyplot.xlabel(r'Soil Volumetric Water Content $(m^3 m^{-3})$')
        elif driver in ('smsf', 'sm'):
            if by_depth:
                for z in range(0, sm.shape[0]):
                    points = pyplot.scatter(
                        sm[z], np.where(
                            cbar0[z] == 0, np.nan, div(rh0, cbar0[z])),
                        color = palette[z], label = '%.2f m' % self.depths[z],
                        **kwargs)
                    points.set_alpha(1)
            else:
                pyplot.scatter(
                    sm.mean(axis = 0), div(rh0, np.nanmean(cbar0, axis = 0)),
                    **kwargs)
            pyplot.plot(*self._ramp(sm, 'smsf'), 'k-')
            pyplot.xlabel('Soil Moisture Wetness (\%)')
        elif driver == 'tsoil':
            if by_depth:
                for z in range(0, tsoil.shape[0]):
                    points = pyplot.scatter(
                        tsoil[z], div(rh0, cbar0[z]), color = palette[z],
                        label = '%.2f m' % self.depths[z], **kwargs)
                    points.set_alpha(1)
            else:
                pyplot.scatter(
                    tsoil.mean(axis = 0), div(rh0, np.nanmean(cbar0, axis = 0)),
                    **kwargs)
            domain = np.arange(tsoil.min(), tsoil.max(), 0.1)
            pyplot.plot(domain,
                arrhenius(domain, self.bplut['tsoil'][0,self._pft]), 'k-')
            pyplot.xlabel('Soil Temperature (deg K)')
        else:
            raise NotImplementedError(
                'Can only plot the following drivers: "vwc", "sm", "tsoil"')
        if by_depth:
            pyplot.legend(markerscale = 2)
        pyplot.ylabel(r'$R_H$/$\bar{C}$')
        pyplot.title(r'Average $K_{mult}$ Response in PFT=%d' % self._pft)
        if ylim is not None:
            pyplot.ylim(*ylim)
        pyplot.show()

    def plot_reco_time_series(
            self, model = 'reco_z', q_rh = 75, q_k = 50, q_nan = 20,
            ylim = None, **kwargs):
        '''
        Plots the observed tower RECO time series and the predicted (modeled)
        RECO time series, based on the current calibrated parameters. One of
        the tower sites with a long record is randomly chosen.

        Parameters
        ----------
        model : str
            Name of the RECO model to use
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
        q_nan : int
            The percentile of total NaN count by site, above which a site will
            not be plotted
        ylim : tuple or list or None
            Sequence of 2 values, the lower and upper limits for the vertical
            axis of the plot
        '''
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rc('font', **{
            'weight' : 'bold',
            'size'   : 14
        })
        params = self._configure(q_rh, q_k, None, model = model)
        assert not np.isnan(params).any(),\
            'Some required parameters are undefined; run calibration first'
        # Randomly pick a site based on low missing-ness
        missing = np.apply_along_axis(lambda x: np.isnan(x).sum(), 0, self._reco_tower)
        idx = np.argwhere(missing <= np.percentile(missing, q_nan)).ravel()
        idx = idx[int(np.random.sample(1) * idx.size)]
        with h5py.File(self._path_to_temp_profile, 'r') as hdf:
            site_name = hdf['site_id'][self._sites][idx]
        # Get predicted, observed values
        predicted = self._reco(params, q_rh, q_k)
        # Update plotting parameters
        kwargs.setdefault('alpha', 0.6)
        pyplot.figure(figsize = (12, 6))
        pyplot.plot(self._reco_tower[:,idx], 'k-', **kwargs)
        pyplot.plot(predicted[:,idx], 'r-', **kwargs)
        pyplot.ylabel(r'RECO $(g\,C\,m^{-2}\,d^{-1})$')
        pyplot.title(r'Site "%s", PFT=%d' % (site_name, self._pft))
        if ylim is not None:
            pyplot.ylim(*ylim)
        pyplot.show()

    def tune_reco(
            self, q_rh = 75, q_k = 50, fixed = None, optimize = True,
            nlopt = True, trials = 1, **kwargs):
        '''
        Optimizes RECO for the vertically stratified SOC model.
        Considerations:

        1. Negative RH values (i.e., NPP > RECO) are set to zero.

        Parameters
        ----------
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
        fixed : tuple or list
            Zero or more parameters whose values should be fixed
            (i.e, NOT optimized)
        optimize : bool
            False to only report parameters and their fit statistics instead
            of optimizing (Default: True)
        nlopt : bool
            True to use the nlopt library for optimization (Default: True)
        trials : int
            Number of searches of the parameter space to perform; if >1,
            initial parameters are randomized for each trial
        **kwargs
            Any number of additional keyword arguments to
            scipy.optimize.least_squares()
        '''
        def residuals(params, q_rh, q_k):
            # Objective function: Difference between tower RECO and L4C RECO
            reco0 = self._reco(params, q_rh, q_k)
            diff = np.subtract(self._reco_tower, reco0)
            # Multiply by the tower weights
            return (self._site_weights * diff)[np.isfinite(diff)]

        self._model_name = 'reco_z'
        init_params = self._configure(
            q_rh, q_k, fixed, model = self._model_name)
        # NaNs may be present because not every site has soil moisture
        #   estimates in every layer (bedrock may be present in soil column)
        assert np.nanmax(self._drivers[1]) <= 100
        assert np.nanmin(self._drivers[1]) >= 0
        # Set defaults where needed
        if np.isnan(init_params).any():
            init_params[np.isnan(init_params)] = np.array(
                [0.4, 270, 10, 50, 0.3])[np.isnan(init_params)]
        self._tune( # Step sizes specified here
            partial(self._reco, q_rh = q_rh, q_k = q_k),
            partial(residuals, q_rh = q_rh, q_k = q_k), init_params, fixed,
            (0.01, 1, 0.1, 0.1, 0.005), trials, optimize, nlopt)

    def tune_reco_power(
            self, q_rh = 75, q_k = 50, fixed = None, optimize = True,
            nlopt = True, trials = 1, **kwargs):
        '''
        Optimizes RECO for the vertically stratified SOC model with a
        power-law RH extinction function. Considerations:

        1. Negative RH values (i.e., NPP > RECO) are set to zero.

        Parameters
        ----------
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
        fixed : tuple or list
            Zero or more parameters whose values should be fixed
            (i.e, NOT optimized)
        optimize : bool
            False to only report parameters and their fit statistics instead
            of optimizing (Default: True)
        nlopt : bool
            True to use the nlopt library for optimization (Default: True)
        trials : int
            Number of searches of the parameter space to perform; if >1,
            initial parameters are randomized for each trial
        **kwargs
            Any number of additional keyword arguments to
            scipy.optimize.least_squares()
        '''
        @suppress_warnings
        def reco(params, q_rh, q_k):
            # Calculate RH as (RECO - RA) or (RECO - (faut * GPP));
            ra = ((1 - params[0]) * self._gpp_tower)
            rh0 = self._reco_tower - ra
            rh0 = np.where(rh0 < 0, 0, rh0) # Mask out negative RH values
            # Compute Cbar with globals "q_rh" and "q_k"
            kmult0 = self._k_mult(params)
            cbar0 = cbar(rh0, kmult0, q_rh, q_k)
            # Extinction rate of heterotrophic respiration with depth
            #   factors OTHER THAN temperature, moisture
            f_z = params[4] * np.power(np.abs(self.depths), -params[5]) *\
                np.ones(cbar0.shape)
            # Set RH from layers below the bedrock depth to zero
            rh = (kmult0 * f_z[:,None,:] * cbar0[:,None,:]).swapaxes(1, 2)
            rh[~self._layer_mask] = 0
            reco0 = ra + rh.swapaxes(1, 2).sum(axis = 0)
            return reco0

        def residuals(params, q_rh, q_k):
            # Objective function: Difference between tower RECO and L4C RECO
            reco0 = reco(params, q_rh, q_k)
            diff = np.subtract(self._reco_tower, reco0)
            # Multiply by the tower weights
            return (self._site_weights * diff)[np.isfinite(diff)]

        self._model_name = 'reco_z_power'
        init_params = self._configure(
            q_rh, q_k, fixed, model = self._model_name)
        # NaNs may be present because not every site has soil moisture
        #   estimates in every layer (bedrock may be present in soil column)
        assert np.nanmax(self._drivers[1]) <= 100
        assert np.nanmin(self._drivers[1]) >= 0
        # Set defaults where needed
        if np.isnan(init_params).any():
            init_params[np.isnan(init_params)] = np.array(
                [0.4, 270, 10, 50, 1, 0.5])[np.isnan(init_params)]
        self._tune( # Step sizes specified here
            partial(self._reco, q_rh = q_rh, q_k = q_k),
            partial(residuals, q_rh = q_rh, q_k = q_k), init_params, fixed,
            (0.01, 1, 0.1, 0.1, 0.01, 0.01), trials, optimize, nlopt)

    def tune_reco_o2_limit(
            self, q_rh = 75, q_k = 50, fixed = None, optimize = True,
            nlopt = True, trials = 1, infer_diff_coefs = True, **kwargs):
        '''
        Optimizes RECO for the vertically stratified SOC model.
        Considerations:

        1. Negative RH values (i.e., NPP > RECO) are set to zero.

        Parameters
        ----------
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
        fixed : tuple or list
            Zero or more parameters whose values should be fixed
            (i.e, NOT optimized)
        optimize : bool
            False to only report parameters and their fit statistics instead
            of optimizing (Default: True)
        nlopt : bool
            True to use the nlopt library for optimization (Default: True)
        trials : int
            Number of searches of the parameter space to perform; if >1,
            initial parameters are randomized for each trial
        infer_diff_coefs : bool
            True to infer the diffusion coefficients based on characteristic
            soil moisture conditions (Default: True); if False, coefficients
            will be fit to the tower respiration data
        **kwargs
            Any number of additional keyword arguments to
            scipy.optimize.least_squares()
        '''
        def k_mult(params):
            # Calculate K_mult* based on current parameters
            #   *K_mult but including O2 diffusion limitation
            f_tsoil = partial(arrhenius, beta0 = params[1])
            f_sm  = linear_constraint(params[2], params[3])
            # Note that f_z() is NOT included here, because we do not want
            #   cbar() to decline with depth
            tsoil, sm, soil_vwc = self._drivers
            conc_O2 = self._concentration_O2(
                self._parameters[self._model_name].index('d_gas'), soil_vwc)
            mm_O2 = conc_O2 / (params[-1] + conc_O2) # km_oxy param
            return f_tsoil(tsoil) *\
                np.nanmin(np.stack((f_sm(sm), mm_O2)), axis = 0)

        @suppress_warnings
        def reco(params, q_rh, q_k):
            # Calculate RH as (RECO - RA) or (RECO - (faut * GPP));
            ra = ((1 - params[0]) * self._gpp_tower)
            rh0 = self._reco_tower - ra
            rh0 = np.where(rh0 < 0, 0, rh0) # Mask out negative RH values
            # Compute Cbar with globals "q_rh" and "q_k"
            kmult0 = k_mult(params)
            cbar0 = cbar(rh0, kmult0, q_rh, q_k)
            # Extinction rate of heterotrophic respiration with depth, due to
            #   factors OTHER THAN temperature, moisture (Koven et al. 2013)
            f_z = np.exp(-np.abs(self.depths) / params[
                self._parameters[self._model_name].index('k_depth_decay')
            ])
            # Set RH from layers below the bedrock depth to zero
            rh = (kmult0 * f_z[:,None,:] * cbar0[:,None,:]).swapaxes(1, 2)
            rh[~self._layer_mask] = 0
            reco0 = ra + rh.swapaxes(1, 2).sum(axis = 0)
            return reco0

        def residuals(params, q_rh, q_k):
            # Objective function: Difference between tower RECO and L4C RECO
            reco0 = reco(params, q_rh, q_k)
            diff = np.subtract(self._reco_tower, reco0)
            # Multiply by the tower weights
            return (self._site_weights * diff)[np.isfinite(diff)]

        self._model_name = 'reco_o2_limit'
        init_params = self._configure(
            q_rh, q_k, fixed, model = self._model_name)
        assert np.nanmax(self._drivers[1]) <= 100
        assert np.nanmin(self._drivers[1]) >= 0
        assert np.nanmax(self._drivers[2]) <= 1
        assert np.nanmax(self._drivers[2]) >= 0
        # Set defaults where needed
        if np.isnan(init_params).any():
            init_params[np.isnan(init_params)] = np.array(
                [0.4, 270, 10, 50, 0.3, 0.1, 0.002])[np.isnan(init_params)]

        # Optionally, instead of fitting d_gas and km_oxy, we can infer their
        #   values based on the soil moisture distribution
        if infer_diff_coefs:
            _, _, soil_vwc = self._drivers
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a = np.power(np.subtract(self._porosity, soil_vwc), 4/3)
            nt, ns = a.shape[-2:]
            soil_vwc_ptiles = np.percentile(
                soil_vwc.reshape((self.depths.size, nt * ns)), (5, 50),
                axis = 1)
            # d_gas when soil is completely dry assuming soil [O2] = atm. [O2]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                d_gas = np.median(
                    1 / a[soil_vwc <= soil_vwc_ptiles[0][:,None,None]])
                km_oxy = np.nanmedian(
                    d_gas * 0.2095 * np.power(
                        self._porosity - soil_vwc_ptiles[1][:,None,None], 4/3))
            print('Inferred coefficients: d_gas = %.2f; km_oxy = %.3f' % (d_gas, km_oxy))
            init_params[
                self._parameters[self._model_name].index('d_gas')] = d_gas
            init_params[
                self._parameters[self._model_name].index('km_oxy')] = km_oxy
            fixed = [] if fixed is None else fixed
            fixed = set(fixed).union(('d_gas', 'km_oxy'))

        self._tune( # Step sizes specified here
            partial(reco, q_rh = q_rh, q_k = q_k),
            partial(residuals, q_rh = q_rh, q_k = q_k), init_params, fixed,
            (0.01, 1, 0.1, 0.1, 0.01, 0, 0), trials, optimize, nlopt)


def cbar(rh, k_mult, q_rh = 75, q_k = 50):
    '''
    Customized Cbar calculation, with depth dependence.

    Parameters
    ----------
    rh : numpy.ndarray
        (N x T) array of tower inferred RH
    k_mult : numpy.ndarray
        (Z x N x T) array of Kmult, for each of Z depths
    q_rh
    q_k

    Returns
    -------
    (Z x N) array of steady-state SOC
    '''
    cutoff = np.apply_along_axis(np.percentile, 1, k_mult, q = q_k)
    # Where environmental conditions are above a cutoff, return the
    #   pseudo-steady state SOC
    cbar0 = np.where(
        k_mult >= cutoff[:,None,:],
        np.divide(rh, np.where(k_mult == 0, np.nan, k_mult)), np.nan)
    return np.nanpercentile(cbar0, q = q_rh, axis = 1)


if __name__ == '__main__':
    import fire
    fire.Fire(StratifiedSoilCalibrationCLI)
