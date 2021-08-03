'''
Extensions to SMAP L4C (and calibration) to support autotrophic respiration,
soil respiration, and litterfall phenologies.

    python phenology.py pft <pft> tune-reco-o2-limit

The "set" command and --fixed argument can be used together to, e.g., turn off
the O2 diffusion limit for a given PFT:

    # Set initial value of km_oxy = 0 and keep it there (along with d_gas)
    python phenology.py pft <pft> set km_oxy 0 tune-reco-limit
        --fixed=[d_gas,km_oxy]
'''

import h5py
import nlopt
import numpy as np
import matplotlib
from functools import partial
from matplotlib import pyplot
from pyl4c import suppress_warnings
from pyl4c.apps.calibration.main import CLI, CONFIG
from pyl4c.science import arrhenius
from pyl4c.stats import linear_constraint
from pyl4c.apps.calibration import GenericOptimization, BPLUT, cbar, report_fit_stats, solve_least_squares
from pyl4c.data.fixtures import restore_bplut

# Constrained optimization bounds
OPT_BOUNDS = {
    # After Davidson et al. (2012)...
    #   Median d_gas in completely dry soil conditions (soil VWC < 5th
    #       percentile): 3.82
    'reco_o2_limit': ( # CUE, tsoil, smsf0, smsf1, d_gas, km_oxy
        np.array((0.0,   1,    0,  25, 3, 0.01)),
        np.array((0.7, 800, 24.9, 100, 5, 0.15))),
    'reco_variable_cue': ( # CUE, tsoil, smsf0, smsf1, par0, par1
        np.array((0.0,   1,    0,  25,    0,  2.01)),
        np.array((0.7, 800, 24.9, 100, 1.99, 20))),
}
NEW_PARAMETERS = ('d_gas', 'km_oxy', 'par0', 'par1')
L4C_PARAMETERS = list(BPLUT._labels)
L4C_PARAMETERS.extend(NEW_PARAMETERS)

class PhenologyCalibrationCLI(CLI):
    '''
    Command line interface for calibrating L4C with various phenology
    mechanisms included.
    '''
    _parameters = {
        'gpp': (
            'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'),
        'reco': (
            'CUE', 'tsoil', 'smsf0', 'smsf1'),
        'reco_o2_limit': (
            'CUE', 'tsoil', 'smsf0', 'smsf1', 'd_gas', 'km_oxy'),
        'reco_variable_cue': (
            'CUE', 'tsoil', 'smsf0', 'smsf1', 'par0', 'par1')
    }

    def __init__(
            self, config = CONFIG, pft = None, start = None, end = None,
            debug = True, use_legacy_pft = True):
        super().__init__(
            config = config, pft = pft, start = start, end = end,
            debug = debug, use_legacy_pft = use_legacy_pft)
        # (Re-)creates the BPLUT store using the correct (expanded) list of
        #   parameter labels
        self._init_bplut(labels = L4C_PARAMETERS)

    def _configure(
            self, q_rh, q_k, fixed, driver_fields, model = 'reco_o2_limit'):
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
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            self._drivers = [
                hdf['drivers/%s' % field][t0:t1,self._sites]
                for field in driver_fields
            ]
        with h5py.File(self._path_to_scratch, 'r') as hdf:
            self._gpp_tower = hdf['tower/GPP'][t0:t1,self._sites,:].mean(axis = 2)
            self._reco_tower = hdf['tower/RECO'][t0:t1,self._sites,:].mean(axis = 2)
            self._site_weights = hdf['site_weights'][:,self._sites]
        # L4C drivers should have no NaNs, based on how they were sourced
        for arr in self._drivers:
            assert np.all(~np.isnan(arr)), 'Unexpected NaNs'
        return init_params

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
                fitted = opt.solve(init_params)
                message = 'Success'
            else:
                fitted = [None for i in range(0, len(init_params))]
                break # Do not iterate through trials if not optimizing
            # Record the found solution and its goodness-of-fit score
            params.append(fitted)
            _, rmse_score, _, _ = self._report_fit(
                self._reco_tower, fit(fitted if optimize else init_params),
                self._site_weights, verbose = False)
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
        self._report_fit(
            self._reco_tower, fit(fitted if optimize else init_params),
            self._site_weights)
        if optimize:
            user_prompt = input('Update parameters for PFT=%d? [Y/n] ' % self._pft)
            do_write = user_prompt == 'Y'
            if do_write:
                print('Updating parameters for PFT=%d...' % self._pft)
                self.bplut.update(self._pft, fitted,
                self._parameters[self._model_name])

    def plot_reco_o2_limit(
            self, q_rh = 75, q_k = 50, ylim = [-0.1, 3], **kwargs):
        '''
        Plots both the soil moisture (wetness) ramp function and the O2
        diffusion limitation curve (a function of soil volumetric water
        content).

        Parameters
        ----------
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
        ylim : tuple or list
            Sequence of 2 values, the lower and upper limits for the vertical
            axis of the plot
        '''
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rc('font', **{
            'weight' : 'bold',
            'size'   : 14
        })
        init_params = self._configure(
            q_rh, q_k, None, ('tsoil', 'smsf'), model = 'reco_o2_limit')
        assert not np.isnan(init_params).any(),\
            'Some required parameters are undefined; run calibration first'
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            porosity = hdf['state/porosity'][self._sites]
        tsoil, smsf = self._drivers
        soil_vwc = np.multiply(smsf / 100, porosity)
        self._drivers.append(soil_vwc)
        f_smsf = linear_constraint(*self.bplut['smsf'][:,self._pft])
        k_mult = f_smsf(smsf) * arrhenius(tsoil, self.bplut['tsoil'][0,self._pft])
        # Calculate RH as (RECO - RA)
        rh = self._reco_tower - ((1 - self.bplut['CUE'][0,self._pft]) * self._gpp_tower)
        # Set negative RH values to zero
        rh = np.where(suppress_warnings(np.less)(rh, 0), 0, rh)
        cbar0 = suppress_warnings(cbar)(rh, k_mult, q_rh, q_k)
        # Update plotting parameters
        kwargs.setdefault('alpha', 0.08)
        kwargs.setdefault('marker', '.')
        pyplot.scatter(smsf, suppress_warnings(np.divide)(rh, cbar0), **kwargs)
        pyplot.plot(*self._ramp(smsf, 'smsf'), 'k-', label = 'Substrate Diffusion')
        domain, _ = self._ramp(smsf, 'smsf')
        conc_O2 = init_params[4] * 0.2095 * np.power(0.8 - np.multiply(domain / 100, 0.8), 4/3)
        pyplot.plot(domain, conc_O2 / (init_params[5] + conc_O2), 'r-', label = 'O2 Diffusion')
        pyplot.xlabel('Surface Soil Moisture Wetness (\%)')
        pyplot.ylabel(r'$R_H$/$\bar{C}$')
        pyplot.title('RH Response in PFT=%d at Maximum Soil Porosity' % self._pft)
        pyplot.ylim(*ylim)
        pyplot.legend()
        pyplot.show()

    def tune_reco_o2_limit(
            self, q_rh = 75, q_k = 50, fixed = None, optimize = True,
            nlopt = True, trials = 1, infer_diff_coefs = True, **kwargs):
        '''
        Optimizes RECO for a model that includes an O2 diffusion limitation
        with Michaelis-Menten kinetics. The 9-km mean L4C RECO is fit to the
        tower-observed RECO using constrained, non-linear least-squares
        optimization.
        Considerations:
            1) Negative RH values (i.e., NPP > RECO) are set to zero.

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
            f_smsf  = linear_constraint(params[2], params[3])
            tsoil, smsf, soil_vwc, porosity = self._drivers
            conc_O2 = concentration_O2(params, soil_vwc, porosity)
            mm_O2 = conc_O2 / (params[5] + conc_O2) # km_oxy param
            return f_tsoil(tsoil) * np.min(
                np.stack((f_smsf(smsf), mm_O2)), axis = 0)

        def concentration_O2(params, soil_vwc, porosity):
            air_frac_O2 = 0.2095 # Liters of O2 per liter of air (20.95%)
            d_gas = params[4]
            return d_gas * air_frac_O2 * np.power(porosity - soil_vwc, 4/3)

        @suppress_warnings
        def reco(params):
            # Calculate RH as (RECO - RA) or (RECO - (faut * GPP));
            ra = ((1 - params[0]) * self._gpp_tower)
            rh = self._reco_tower - ra
            rh = np.where(rh < 0, 0, rh) # Mask out negative RH values
            # Compute Cbar with globals "q_rh" and "q_k"
            kmult0 = k_mult(params)
            cbar0 = cbar(rh, kmult0, q_rh, q_k)
            return ra + (kmult0 * cbar0)

        def residuals(params):
            # Objective function: Difference between tower RECO and L4C RECO
            reco0 = reco(params)
            diff = np.subtract(self._reco_tower, reco0)
            # Multiply by the tower weights
            return (self._site_weights * diff)[np.isfinite(diff)]

        self._model_name = 'reco_o2_limit'
        if infer_diff_coefs and fixed is not None:
            raise ValueError('Cannot set both --infer-diff-coefs and --fixed')
        init_params = self._configure(
            q_rh, q_k, fixed, ('tsoil', 'smsf'), model = self._model_name)
        # Set defaults where needed
        if np.isnan(init_params).any():
            init_params[np.isnan(init_params)] = np.array(
                [0.4, 270, 10, 50, 3.78, 0.105])[np.isnan(init_params)]
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            porosity = hdf['state/porosity'][self._sites]
        # NOTE: Converting from "wetness" to volumetric water content (VWC)
        _, smsf = self._drivers
        soil_vwc = np.multiply(smsf / 100, porosity)
        self._drivers.extend((soil_vwc, porosity))

        # Optionally, instead of fitting d_gas and km_oxy, we can infer their
        #   values based on the soil moisture distribution
        if infer_diff_coefs:
            a = np.power(np.subtract(porosity, soil_vwc), 4/3)
            soil_vwc_ptiles = np.percentile(soil_vwc, (5, 50))
            # d_gas when soil is completely dry assuming soil [O2] = atm. [O2]
            d_gas = np.median(1 / a[soil_vwc <= soil_vwc_ptiles[0]])
            km_oxy = np.median(
                d_gas * 0.2095 * np.power(porosity - soil_vwc_ptiles[1], 4/3))
            print('Inferred coefficients: d_gas = %.2f; km_oxy = %.3f' % (d_gas, km_oxy))
            init_params[4] = d_gas
            init_params[5] = km_oxy
            fixed = [] if fixed is None else fixed
            fixed = set(fixed).union(('d_gas', 'km_oxy'))
        self._tune( # Step sizes specified here
            reco, residuals, init_params, fixed, (0.01, 1, 0.1, 0.1, 0.1, 0.002),
            trials, optimize, nlopt)

    def plot_variable_cue(
            self, q_rh = 75, q_k = 50, ylim=[-0.1, 1.1], **kwargs):
        '''
        Plots empirical carbon-use efficiency (CUE) against photosynthetically
        active radiation (PAR).

        Parameters
        ----------
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
        ylim : tuple or list
            Sequence of 2 values, the lower and upper limits for the vertical
            axis of the plot
        '''
        @suppress_warnings
        def empirical_cue(ra, gpp, pclip = (1, 99)):
            cue0 = (1 - np.divide(ra, np.where(gpp > 0, gpp, np.nan)))
            cue0[cue0 < np.nanpercentile(cue0, pclip[0])] = np.nan
            cue0[cue0 > np.nanpercentile(cue0, pclip[1])] = np.nan
            return cue0

        with h5py.File(self._path_to_drivers, 'r') as hdf:
            soc = hdf['state/soil_organic_carbon'][:,self._sites,:].mean(axis = 2)

        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rc('font', **{
            'weight' : 'bold',
            'size'   : 14
        })
        # Update plotting parameters
        kwargs.setdefault('alpha', 0.08)
        kwargs.setdefault('marker', '.')
        init_params = self._configure(
            q_rh, q_k, None, ('tsoil', 'smsf', 'par'), model = 'reco_variable_cue')
        assert not np.isnan(init_params).any(),\
            'Some required parameters are undefined; run calibration first'
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            porosity = hdf['state/porosity'][self._sites]
        tsoil, smsf, par = self._drivers
        f_par = linear_constraint(*self.bplut['par'][:,self._pft].tolist())
        # Generate (fixed) TSOIL and SMSF ramp functions
        f_tsoil = partial(arrhenius, beta0 = self.bplut['tsoil'][0,self._pft])
        f_smsf  = linear_constraint(*self.bplut['smsf'][:,self._pft].tolist())
        kmult = f_tsoil(tsoil) * f_smsf(smsf)
        rh = np.stack([
            kmult * soc[i,...] * self.bplut['decay_rates'][i,self._pft]
            for i in range(0, 3)
        ], axis = 0)
        rh[1,...] = rh[1,...] * (1 - self.bplut['f_structural'][0,self._pft])
        ra = self._reco_tower - rh.sum(axis = 0)
        ra = np.where(suppress_warnings(np.less)(ra, 0), np.nan, ra)
        kwargs.setdefault('alpha', 0.05)
        kwargs.setdefault('marker', '.')
        pyplot.scatter(par, empirical_cue(ra, self._gpp_tower), **kwargs)
        pyplot.xlabel('Photosynthetically Active Radiation (PAR)')
        pyplot.ylabel('Empirical CUE (1 - $R_A$/GPP)')
        pyplot.title('PFT=%d' % self._pft)
        domain = np.arange(np.nanmin(par), np.nanmax(par), 0.1)
        ramp = self._constrain(domain, 'par')
        pyplot.plot(domain, self.bplut['CUE'][0,self._pft] * ramp, 'k-')
        pyplot.ylim(*ylim)
        pyplot.show()

    def tune_variable_cue(
            self, q_rh = 75, q_k = 50, fixed = None, optimize = True,
            nlopt = True, trials = 1, **kwargs):
        '''
        Optimizes RECO for a model that includes the Kok effect (CUE varies
        linearly with PAR). The 9-km mean L4C RECO is fit to the
        tower-observed RECO using constrained, non-linear least-squares
        optimization.
        Considerations:
            1) Negative RH values (i.e., NPP > RECO) are set to zero.

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
        def k_mult(params):
            # Calculate K_mult based on current parameters
            f_tsoil = partial(arrhenius, beta0 = params[1])
            f_smsf  = linear_constraint(params[2], params[3])
            tsoil, smsf, _ = self._drivers
            return f_tsoil(tsoil) * f_smsf(smsf)

        @suppress_warnings
        def reco(params):
            f_par = linear_constraint(params[-2], params[-1])
            _, _, par = self._drivers
            ra = ((1 - (f_par(par) * params[0])) * self._gpp_tower)
            # Calculate RH as (RECO - RA) or (RECO - (faut * GPP));
            rh = self._reco_tower - ra
            rh = np.where(rh < 0, 0, rh) # Mask out negative RH values
            # Compute Cbar with globals "q_rh" and "q_k"
            kmult0 = k_mult(params)
            cbar0 = cbar(rh, kmult0, q_rh, q_k)
            return ra + (kmult0 * cbar0)

        def residuals(params):
            # Objective function: Difference between tower RECO and L4C RECO
            reco0 = reco(params)
            diff = np.subtract(self._reco_tower, reco0)
            # Multiply by the tower weights
            return (self._site_weights * diff)[np.isfinite(diff)]

        self._model_name = 'reco_variable_cue'
        init_params = self._configure(
            q_rh, q_k, fixed, ('tsoil', 'smsf', 'par'), model = self._model_name)
        # Set defaults where needed
        if np.isnan(init_params).any():
            init_params[np.isnan(init_params)] = np.array(
                [0.4, 270, 10, 50, 1, 5])[np.isnan(init_params)]
        self._tune( # Step sizes specified here
            reco, residuals, init_params, fixed, (0.01, 1, 0.1, 0.1, 0.02, 0.02),
            trials, optimize, nlopt)


if __name__ == '__main__':
    import fire
    fire.Fire(PhenologyCalibrationCLI)
