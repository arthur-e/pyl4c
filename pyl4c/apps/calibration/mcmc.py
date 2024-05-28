'''
Calibration of L4C using Markov Chain Monte Carlo (MCMC). Example use:

    python mcmc.py pft <pft> tune-gpp --config=<config_file>
    python mcmc.py pft <pft> tune-reco --config=<config_file>
'''

import datetime
import yaml
import os
import warnings
import numpy as np
import h5py
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import pyl4c
from functools import partial
from multiprocessing import get_context
from numbers import Number
from typing import Callable, Sequence
from matplotlib import pyplot
from textwrap import wrap
from scipy import signal
from pyl4c import pft_dominant
from pyl4c.data.fixtures import restore_bplut_flat
from pyl4c.science import vpd, par, rescale_smrz, arrhenius
from pyl4c.stats import linear_constraint, rmsd
from pyl4c.apps.calibration import cbar

L4C_DIR = os.path.dirname(pyl4c.__file__)
PFT_VALID = (1,2,3,4,5,6,7,8)

# This matplotlib setting prevents labels from overplotting
pyplot.rcParams['figure.constrained_layout.use'] = True


class BlackBoxLikelihood(pt.Op):
    '''
    A custom Theano operator that calculates the "likelihood" of model
    parameters; it takes a vector of values (the parameters that define our
    model) and returns a single "scalar" value (the log-likelihood).

    Parameters
    ----------
    model : Callable
        An arbitrary "black box" function that takes two arguments: the
        model parameters ("params") and the forcing data ("x")
    observed : numpy.ndarray
        The "observed" data that our log-likelihood function takes in
    x : numpy.ndarray or None
        The forcing data (input drivers) that our model requires, or None
        if no driver data are required
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    objective : str
        Name of the objective (or "loss") function to use, one of
        ('rmsd', 'gaussian', 'kge'); defaults to "rmsd"
    '''
    itypes = [pt.dvector] # Expects a vector of parameter values when called
    otypes = [pt.dscalar] # Outputs a single scalar value (the log likelihood)

    def __init__(
            self, model: Callable, observed: Sequence, x: Sequence = None,
            weights: Sequence = None, objective: str = 'rmsd'):
        '''
        Initialise the Op with various things that our log-likelihood function
        requires. The observed data ("observed") and drivers ("x") must be
        stored on the instance so the Theano Op can work seamlessly.
        '''
        self.model = model
        self.observed = observed
        self.x = x
        self.weights = weights
        if objective in ('rmsd', 'rmse'):
            self._loglik = self.loglik
        elif objective == 'gaussian':
            self._loglik = self.loglik_gaussian
        elif objective == 'kge':
            self._loglik = self.loglik_kge
        else:
            raise ValueError('Unknown "objective" function specified')

    def loglik(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        '''
        Pseudo-log likelihood, based on the root-mean squared deviation
        (RMSD). The sign of the RMSD is forced to be negative so as to allow
        for maximization of this objective function.

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The (negative) root-mean squared deviation (RMSD) between the
            predicted and observed values
        '''
        predicted = self.model(params, *x)
        if self.weights is not None:
            return -np.sqrt(
                np.nanmean(((predicted - observed) * self.weights) ** 2))
        return -np.sqrt(np.nanmean(((predicted - observed)) ** 2))

    def loglik_gaussian(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        '''
        Gaussian log-likelihood, assuming independent, identically distributed
        observations.

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The (negative) log-likelihood
        '''
        predicted = self.model(params, *x)
        sigma = params[-1]
        # Gaussian log-likelihood;
        # -\frac{N}{2}\,\mathrm{log}(2\pi\hat{\sigma}^2)
        #   - \frac{1}{2\hat{\sigma}^2} \sum (\hat{y} - y)^2
        return -0.5 * np.log(2 * np.pi * sigma**2) - (0.5 / sigma**2) *\
            np.nansum((predicted - observed)**2)

    def loglik_kge(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        r'''
        Kling-Gupta efficiency.

        $$
        KGE = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}
        $$

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The Kling-Gupta efficiency
        '''
        predicted = self.model(params, *x)
        r = np.corrcoef(predicted, observed)[0, 1]
        alpha = np.std(predicted) / np.std(observed)
        beta = np.sum(predicted) / np.sum(observed)
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    def perform(self, node, inputs, outputs):
        '''
        The method that is used when calling the Op.

        Parameters
        ----------
        node
        inputs : Sequence
        outputs : Sequence
        '''
        (params,) = inputs
        logl = self._loglik(params, self.observed, self.x)
        outputs[0][0] = np.array(logl) # Output the log-likelihood


class AbstractSampler(object):
    '''
    Generic algorithm for fitting a model to data based on observed values
    similar to what we can produce with our model. Not intended to be called
    directly.
    '''

    def get_posterior(self, thin: int = 1) -> np.ndarray:
        '''
        Returns a stacked posterior array, with optional thinning, combining
        all the chains together.

        Parameters
        ----------
        thin : int

        Returns
        -------
        numpy.ndarray
        '''
        trace = az.from_netcdf(self.backend)
        return np.stack([ # i.e., get every ith element, each chain
            trace['posterior'][p].values[:,::thin].ravel()
            for p in self.required_parameters[self.name]
        ], axis = -1)

    def get_trace(
            self, thin: int = None, burn: int = None
        ) -> az.data.inference_data.InferenceData:
        '''
        Extracts the trace from the backend data store.

        Parameters
        ----------
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        '''
        trace = az.from_netcdf(self.backend)
        if thin is None and burn is None:
            return trace
        return trace.sel(draw = slice(burn, None, thin))

    def plot_autocorr(self, thin: int = None, burn: int = None, **kwargs):
        '''
        Auto-correlation plot for an MCMC sample.

        Parameters
        ----------
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        **kwargs
            Additional keyword arguments to `arviz.plot_autocorr()`.
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        kwargs.setdefault('combined', True)
        if thin is None:
            az.plot_autocorr(trace, **kwargs)
        else:
            burn = 0 if burn is None else burn
            az.plot_autocorr(
                trace.sel(draw = slice(burn, None, thin))['posterior'],
                **kwargs)
        pyplot.show()

    def plot_forest(self, thin: int = None, burn: int = None, **kwargs):
        '''
        Forest plot for an MCMC sample.

        Parameters
        ----------
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        **kwargs
            Additional keyword arguments to `arviz.plot_forest()`.

        In particular:

        - `hdi_prob`: A float indicating the highest density interval (HDF) to
            plot
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        if thin is None:
            az.plot_forest(trace, **kwargs)
        else:
            burn = 0 if burn is None else burn
            az.plot_forest(
                trace.sel(draw = slice(burn, None, thin))['posterior'],
                **kwargs)
        pyplot.show()

    def plot_pair(self, **kwargs):
        '''
        Paired variables plot for an MCMC sample.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to `arviz.plot_pair()`.
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        az.plot_pair(trace, **kwargs)
        pyplot.show()

    def plot_posterior(self, **kwargs):
        '''
        Plots the posterior distribution for an MCMC sample.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to `arviz.plot_posterior()`.
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        az.plot_posterior(trace, **kwargs)
        pyplot.show()


class StochasticSampler(AbstractSampler):
    '''
    A Markov Chain-Monte Carlo (MCMC) sampler for an arbitrary model. The
    specific sampler used is the Differential Evolution (DE) MCMC algorithm
    described by Ter Braak (2008), though the implementation is specific to
    the PyMC3 library.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    model : Callable or None
        The function to call (with driver data and parameters); this function
        should have a Sequence of model parameters as its first argument and
        then each driver dataset as a following positional argument; it should
        require no external state. If `None`, will look for a static method
        defined on this class called `_name()` where `name` is the model name
        defined in the configuration file.
    params_dict : dict or None
        Dictionary of model parameters, to be used as initial values and as
        the basis for constructing a new dictionary of optimized parameters
    backend : str or None
        Path to a netCDF4 file backend (Default: None)
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    '''
    def __init__(
            self, config: dict, model: Callable = None,
            params_dict: dict = None, backend: str = None,
            weights: Sequence = None):
        self.backend = backend
        self.config = config
        self.model = model
        self.name = config['name']
        self.params = params_dict
        self.weights = weights
        assert os.path.exists(os.path.dirname(backend))

    def run(
            self, observed: Sequence, drivers: Sequence,
            draws = 1000, chains = 3, tune = 'lambda',
            scaling: float = 1e-3, prior: dict = dict(),
            check_shape: bool = False, save_fig: bool = False,
            var_names: Sequence = None) -> None:
        '''
        Fits the model using DE-MCMCz approach. `tune="lambda"` (default) is
        recommended; lambda is related to the scale of the jumps learned from
        other chains, but epsilon ("scaling") controls the scale directly.
        Using a larger value for `scaling` (Default: 1e-3) will produce larger
        jumps and may directly address "sticky" chains.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function
        draws : int
            Number of samples to draw (on each chain); defaults to 1000
        chains : int
            Number of chains; defaults to 3
        tune : str or None
            Which hyperparameter to tune: Defaults to 'lambda', but can also
            be 'scaling' or None.
        scaling : float
            Initial scale factor for epsilon (Default: 1e-3)
        prior : dict
        check_shape : bool
            True to require that input driver datasets have the same shape as
            the observed values (Default: False)
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        var_names : Sequence
            One or more variable names to show in the plot
        '''
        assert not check_shape or drivers[0].shape == observed.shape,\
            'Driver data should have the same shape as the "observed" data'
        assert len(drivers) == len(self.required_drivers[self.name]),\
            'Did not receive expected number of driver datasets!'
        assert tune in ('lambda', 'scaling') or tune is None
        # Update prior assumptions
        self.prior.update(prior)
        # Generate an initial goodness-of-fit score
        if self.params is not None:
            predicted = self.model([
                self.params[p] for p in self.required_parameters[self.name]
            ], *drivers)
        if self.weights is not None:
            score = np.sqrt(
                np.nanmean(((predicted - observed) * self.weights) ** 2))
        else:
            score = np.sqrt(np.nanmean(((predicted - observed)) ** 2))
        print('-- RMSD at the initial point: %.3f' % score)
        print('Compiling model...')
        try:
            compiler = getattr(self, 'compile_%s_model' % self.name.lower())
        except AttributeError:
            raise AttributeError('''Could not find a compiler for model named
            "%s"; make sure that a function "compile_%s_model()" is defined on
             this class''' % (self.name, self.name.lower()))
        with compiler(observed, drivers) as model:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                step_func = pm.DEMetropolisZ(tune = tune, scaling = scaling)
                trace = pm.sample(
                    draws = draws, step = step_func, cores = chains,
                    chains = chains, idata_kwargs = {'log_likelihood': True})
            if self.backend is not None:
                print('Writing results to file...')
                trace.to_netcdf(self.backend)
            if var_names is None:
                az.plot_trace(trace, var_names = ['~log_likelihood'])
            else:
                az.plot_trace(trace, var_names = var_names)
            if save_fig:
                pyplot.savefig('.'.join(self.backend.split('.')[:-1]) + '.png')
            else:
                pyplot.show()


class L4CStochasticSampler(StochasticSampler):
    '''
    A Markov Chain-Monte Carlo (MCMC) sampler for L4C. The specific sampler
    used is the Differential Evolution (DE) MCMC algorithm described by
    Ter Braak (2008), though the implementation is specific to the PyMC3
    library.

    Considerations:

    1. Tower GPP is censored when values are < 0 or when APAR is
        < 0.1 MJ m-2 d-1.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    model : Callable or None
        The function to call (with driver data and parameters); this function
        should have a Sequence of model parameters as its first argument and
        then each driver dataset as a following positional argument; it should
        require no external state. If `None`, will look for a static method
        defined on this class called `_name()` where `name` is the model name
        defined in the configuration file.
    params_dict : dict or None
        Dictionary of model parameters, to be used as initial values and as
        the basis for constructing a new dictionary of optimized parameters
    backend : str or None
        Path to a NetCDF4 file backend (Default: None)
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    '''
    required_parameters = {
        'GPP':  ['LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'],
        'RECO': ['CUE', 'tsoil', 'smsf0', 'smsf1'],
    }
    required_drivers = {
        # Tsurf = Surface skin temperature; Tmin = Minimum daily temperature
        'GPP':  ['fPAR', 'PAR', 'Tmin', 'VPD', 'SMRZ', 'FT'],
        'RECO': ['tower_RECO', 'tower_GPP', 'SMSF', 'Tsoil', 'q_rh', 'q_k']
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model is None:
            self.model = getattr(self, f'_{self.name}')
        # Set the model's prior distribution assumptions
        self.prior = dict()
        if self.params is not None:
            for key in self.required_parameters[self.name]:
                # NOTE: This is the default only for LUE_max; other parameters,
                #   with Uniform distributions, don't use anything here
                self.prior.setdefault(key, {
                    'mu': self.params[key],
                    'sigma': 2e-4
                })

    @staticmethod
    def _gpp(params, fpar, par, tmin, vpd, smrz, ft):
        'L4C GPP function'
        # Calculate E_mult based on current parameters:
        #   'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'
        try:
            f_tmin = linear_constraint(params[1], params[2])
        except AssertionError:
            raise AssertionError(f"FAILED in linear_constraint() with parameters: {params[1:3]}")
        try:
            f_vpd  = linear_constraint(params[3], params[4], 'reversed')
        except AssertionError:
            raise AssertionError(f"FAILED in linear_constraint(form = 'reversed') with parameters: {params[3:5]}")
        try:
            f_smrz = linear_constraint(params[5], params[6])
        except AssertionError:
            raise AssertionError(f"FAILED in linear_constraint() with parameters: {params[5:7]}")
        f_ft   = linear_constraint(params[7], 1.0, 'binary')
        e_mult = f_tmin(tmin) * f_vpd(vpd) * f_smrz(smrz) * f_ft(ft)
        # Calculate GPP based on the provided BPLUT parameters
        return fpar * par * params[0] * e_mult

    @staticmethod
    def _gpp2(params, fpar, par, tmin, vpd, smrz, ft):
        '''
        L4C GPP function, with alternate parameters, where the ramp functions
        are defined in terms of the left edge (lowest value) and the width of
        the ramp function interval.
        '''
        # Calculate E_mult based on current parameters:
        #   'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'
        f_tmin = linear_constraint(params[1], params[1] + params[2])
        f_vpd  = linear_constraint(params[3], params[3] + params[4], 'reversed')
        f_smrz = linear_constraint(params[5], params[5] + params[6])
        f_ft   = linear_constraint(params[7], 1.0, 'binary')
        e_mult = f_tmin(tmin) * f_vpd(vpd) * f_smrz(smrz) * f_ft(ft)
        # Calculate GPP based on the provided BPLUT parameters
        return fpar * par * params[0] * e_mult

    @staticmethod
    def _reco(params, tower_reco, tower_gpp, smsf, tsoil, q_rh, q_k):
        '''
        L4C RECO function, with standard parameters.
        '''
        # Calculate RH as (RECO - RA) or (RECO - (faut * GPP))
        ra = ((1 - params[0]) * tower_gpp)
        rh = tower_reco - ra
        rh = np.where(rh < 0, 0, rh) # Mask out negative RH values
        f_tsoil = partial(arrhenius, beta0 = params[1])
        f_smsf = linear_constraint(params[2], params[3])
        kmult0 = f_tsoil(tsoil) * f_smsf(smsf)
        cbar0 = cbar(rh, kmult0, q_rh, q_k)
        return ra + (kmult0 * cbar0)

    def compile_gpp_model(
            self, observed: Sequence, drivers: Sequence) -> pm.Model:
        '''
        Creates a new GPP model based on the prior distribution. Model can be
        re-compiled multiple times, e.g., for cross validation.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function

        Returns
        -------
        pm.Model
        '''
        # Define the objective/ likelihood function
        log_likelihood = BlackBoxLikelihood(
            self.model, observed, x = drivers, weights = self.weights,
            objective = self.config['optimization']['objective'].lower())
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            # NOTE: LUE is unbounded on the right side
            LUE = pm.TruncatedNormal('LUE', **self.prior['LUE'])
            tmin0 = pm.Uniform('tmin0', **self.prior['tmin0'])
            tmin1 = pm.Uniform('tmin1', **self.prior['tmin1'])
            vpd0 = pm.Uniform('vpd0', **self.prior['vpd0'])
            vpd1 = pm.Uniform('vpd1', **self.prior['vpd1'])
            # NOTE: Fixing lower-bound on SMRZ at zero
            smrz0 = pm.Uniform('smrz0', **self.prior['smrz0'])
            smrz1 = pm.Uniform('smrz1', **self.prior['smrz1'])
            ft0 = pm.Uniform('ft0', **self.prior['ft0'])
            # Convert model parameters to a tensor vector
            params_list = [LUE, tmin0, tmin1, vpd0, vpd1, smrz0, smrz1, ft0]
            params = pt.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        return model

    def compile_reco_model(
            self, observed: Sequence, drivers: Sequence) -> pm.Model:
        '''
        Creates a new RECO model based on the prior distribution. Model can be
        re-compiled multiple times, e.g., for cross validation.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function

        Returns
        -------
        pm.Model
        '''
        log_likelihood = BlackBoxLikelihood(
            self.model, observed, x = drivers, weights = self.weights,
            objective = self.config['optimization']['objective'].lower())
        with pm.Model() as model:
            # (Stochstic) Priors for unknown model parameters
            CUE = pm.Beta('CUE', **self.prior['CUE'])
            tsoil = pm.Uniform('tsoil', **self.prior['tsoil'])
            smsf0 = 0.0 # NOTE: Left edge fixed at 0.0
            smsf1 = pm.Uniform('smsf1', **self.prior['smsf1'])
            # Convert model parameters to a tensor vector
            params_list = [CUE, tsoil, smsf0, smsf1]
            params = pt.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        return model


class CalibrationAPI(object):
    '''
    Convenience class for calibrating the L4C GPP and RECO models. Meant to
    be used with `fire.Fire()`. Uses:

        # Run the calibration for a specific PFT
        python mcmc.py tune-gpp <pft>

        # Get access to the sampler (and debugger), after calibration is run
        python mcmc.py tune-gpp <pft> --ipdb
    '''
    def __init__(self, config: str = None, pft: int = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                L4C_DIR, 'data/files/config_L4C_MCMC_calibration.yaml')
        print(f'Using configuration file: {config_file}')
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        if pft is not None:
            assert pft in PFT_VALID, f'Invalid PFT: {pft}'
            self._pft = pft
        self.hdf5 = self.config['data']['file']

    def _clean(
            self, raw: Sequence, drivers: Sequence, protocol: str = 'GPP',
            num_std: int = 5):
        'Cleans up data values according to a prescribed protocol'
        if protocol == 'GPP':
            # Filter out observed GPP values when GPP is negative or when
            #   APAR < 0.1 g C m-2 day-1
            apar = drivers['fPAR'] * drivers['PAR']
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

    def _get_params(self, model):
        # Filter the parameters to just those for the PFT of interest
        params_dict = restore_bplut_flat(self.config['BPLUT'])
        return dict([
            (k, params_dict[k].ravel()[self._pft])
            for k in L4CStochasticSampler.required_parameters[model]
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
            for field in L4CStochasticSampler.required_drivers['GPP']:
                # Try reading the field exactly as described in config file
                if field in field_map:
                    if field_map[field] in hdf:
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

            # Check units on fPAR, average sub-grid heterogeneity
            if np.nanmax(drivers['fPAR'][:]) > 10:
                drivers['fPAR'] /= 100
            if drivers['fPAR'].ndim == 3 and drivers['fPAR'].shape[-1] == 81:
                drivers['fPAR'] = np.nanmean(drivers[f'fPAR'], axis = -1)
            assert len(set(L4CStochasticSampler.required_drivers['GPP'])\
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
            for field in ('SMSF', 'Tsoil'):
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
        drivers = [drivers[k] for k in ('SMSF', 'Tsoil')]
        return (drivers, tower_reco, weights)

    def _preplot(self, model: str):
        'Loads the data and parameters required for plotting'
        params_dict = self._get_params(self._pft, model)
        backend = self.config['optimization']['backend_template'].format(
            model = model, pft = self._pft)
        sampler = L4CStochasticSampler(
            self.config, getattr(L4CStochasticSampler, f'_{model.lower()}'),
            params_dict, backend = backend)
        return (sampler, backend)

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

    def plot_autocorr(self, model: str, **kwargs):
        'Plots the autocorrelation in the trace for each parameter'
        sampler, backend = self._preplot(self._pft, model)
        sampler.plot_autocorr(**kwargs)

    def plot_posterior(self, model: str, **kwargs):
        'Plots the posterior density for each parameter'
        sampler, backend = self._preplot(self._pft, model)
        sampler.plot_posterior()

    def plot_trace(self, model: str, **kwargs):
        'Plots the trace for each parameter'
        sampler, backend = self._preplot(self._pft, model)
        trace = sampler.get_trace(
            burn = kwargs.get('burn', None), thin = kwargs.get('thin'))
        az.plot_trace(trace, kwargs.get('var_names', None))
        pyplot.show()

    def tune_gpp(
            self, filter_length: int = 2, plot: str = None,
            ipdb: bool = False, save_fig: bool = False, **kwargs):
        '''
        Run the L4C GPP calibration.

        - For GPP data: Removes observations where GPP < 0 or where APAR is
            < 0.1 MJ m-2 day-1

        Parameters
        ----------
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
        **kwargs
            Additional keyword arguments passed to
            `L4CStochasticSampler.run()`
        '''
        assert self._pft in PFT_VALID, f'Invalid PFT: {self._pft}'
        # IMPORTANT: Set the "name" property of the configuration file;
        #   this is used by StochasticSampler classes to figure out how
        #   to compile the model
        self.config['name'] = 'GPP'
        # Pass configuration parameters to L4CStochasticSampler.run()
        for key in ('chains', 'draws', 'tune', 'scaling'):
            if key in self.config['optimization'].keys() and not key in kwargs.keys():
                kwargs[key] = self.config['optimization'][key]
        params_dict = self._get_params('GPP')
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']
        objective = self.config['optimization']['objective'].lower()

        print('Loading driver datasets...')
        drivers, drivers_flat, tower_gpp, tower_gpp_flat, weights =\
            self._load_gpp_data(filter_length)

        print('Initializing sampler...')
        backend = self.config['optimization']['backend_template'].format(
            model = 'GPP', pft = self._pft)
        model_func = getattr( # e.g., L4CStochasticSampler._gpp
            L4CStochasticSampler,
            self.config['optimization']['function']['GPP'])
        sampler = L4CStochasticSampler(
            self.config, model_func, params_dict,
            backend = backend, weights = weights)

        # Get (informative) priors for just those parameters that have them
        with open(self.config['optimization']['prior'], 'r') as file:
            prior = yaml.safe_load(file)
        prior_params = filter(
            lambda p: p in prior.keys(), sampler.required_parameters['GPP'])
        prior = dict([
            (p, dict([(k, v[self._pft]) for k, v in prior[p].items()]))
            for p in prior_params
        ])

        # For diganostics or plotting representative sites
        if ipdb or plot in ('exemplar', 'scatter'):
            # Get the tower with the most available data
            idx = np.apply_along_axis(
                lambda x: x.size - np.isnan(x).sum(), 0, tower_gpp).argmax()
            # The BPLUT has a different representation of ramp function
            #   parameters: translate "param1" into ("param1" - "param0")
            params0 = [
                params_dict[k] if k[-1] != '1' else params_dict[k] - params_dict[k.replace('1', '0')]
                for k in sampler.required_parameters['GPP']
            ]
            # Get proposed (new) parameters, if provided
            params1 = []
            for k, key in enumerate(sampler.required_parameters['GPP']):
                if key in kwargs.keys():
                    params1.append(kwargs[key])
                else:
                    params1.append(params0[k])
            if plot == 'exemplar':
                _drivers = [
                    drivers[k][:,idx] for k in sampler.required_drivers['GPP']
                ]
                gpp0 = model_func(params0, *_drivers)
                gpp1 = model_func(params1, *_drivers)
                pyplot.plot(tower_gpp[:,idx], 'g-', label = 'Tower GPP')
                pyplot.plot(gpp0, 'r-', alpha = 0.5, label = 'Old Simulation')
                pyplot.plot(gpp1, 'k-', label = 'New Simulation')
            elif plot == 'scatter':
                tidx = np.random.randint( # Random 20% sample
                    0, tower_gpp.size, size = tower_gpp.size // 5)
                _drivers = [
                    drivers[k].ravel()[tidx] for k in sampler.required_drivers['GPP']
                ]
                _obs = tower_gpp.ravel()[tidx]
                gpp0 = model_func(params0, *_drivers)
                gpp1 = model_func(params1, *_drivers)
                # Calculate (parameters of) trend lines
                mask = np.isnan(_obs)
                a0, b0 = np.polyfit(_obs[~mask], gpp0[~mask], deg = 1)
                a1, b1 = np.polyfit(_obs[~mask], gpp1[~mask], deg = 1)
                # Create a scatter plot
                fig = pyplot.figure(figsize = (6, 6))
                pyplot.scatter(
                    _obs, gpp0, s = 2, c = 'k', alpha = 0.2,
                    label = 'Old (RMSE=%.2f, r=%.2f)' % (
                        rmsd(_obs, gpp0), np.corrcoef(_obs[~mask], gpp0[~mask])[0,1]))
                pyplot.plot(_obs, a0 * _obs + b0, 'k-', alpha = 0.8)
                pyplot.scatter(
                    _obs, gpp1, s = 2, c = 'r', alpha = 0.2,
                    label = 'New (RMSE=%.2f, r=%.2f)' % (
                        rmsd(_obs, gpp1), np.corrcoef(_obs[~mask], gpp1[~mask])[0,1]))
                pyplot.plot(_obs, a1 * _obs + b1, 'r-', alpha = 0.8)
                ax = fig.get_axes()
                ax[0].set_aspect(1)
                ax[0].plot([0, 1], [0, 1],
                    transform = ax[0].transAxes, linestyle = 'dashed',
                    c = 'k', alpha = 0.5)
                pyplot.xlabel('Observed')
                pyplot.ylabel('Predicted')
                pyplot.title('\n'.join(wrap(f'PFT {self._pft} with: ' + ', '.join(list(map(
                    lambda x: f'{x[0]}={x[1]}', zip(
                    sampler.required_parameters['GPP'], params1)))))))
            pyplot.legend()
            pyplot.show()
            # For diagnostics
            if ipdb:
                trace = sampler.get_trace(
                    burn = kwargs.get('burn', None), thin = kwargs.get('thin'))
                import ipdb
                ipdb.set_trace()
            return

        # Set var_names to tell ArviZ to plot only the free parameters; i.e.,
        #   those with priors
        var_names = list(filter(
            lambda x: x in prior.keys(), sampler.required_parameters['GPP']))
        # Convert drivers from a dict to a sequence, then run sampler
        drivers_flat = [drivers_flat[d] for d in sampler.required_drivers['GPP']]
        # Remove any kwargs that don't belong
        for k in list(kwargs.keys()):
            if k not in ('chains', 'draws', 'tune', 'scaling', 'save_fig', 'var_names'):
                del kwargs[k]
        sampler.run(
            tower_gpp_flat, drivers_flat, prior = prior, save_fig = save_fig,
            **kwargs)

    def tune_reco(
            self, filter_length: int = 2, q_rh: int = 75, q_k: int = 50,
            plot: str = None, ipdb: bool = False, save_fig: bool = False,
            **kwargs):
        '''
        Run the L4C RECO calibration.

        - Negative RH values (i.e., NPP > RECO) are set to zero.

        Parameters
        ----------
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data
        q_rh : int
            The percentile of RH/Kmult to use in calculating Cbar
        q_k : int
            The percentile of Kmult below which RH/Kmult values are masked
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
        **kwargs
            Additional keyword arguments passed to
            `L4CStochasticSampler.run()`
        '''
        assert self._pft in PFT_VALID, f'Invalid PFT: {self._pft}'
        # IMPORTANT: Set the "name" property of the configuration file;
        #   this is used by StochasticSampler classes to figure out how
        #   to compile the model
        self.config['name'] = 'RECO'
        # Pass configuration parameters to MOD17StochasticSampler.run()
        for key in ('chains', 'draws', 'tune', 'scaling'):
            if key in self.config['optimization'].keys() and not key in kwargs.keys():
                kwargs[key] = self.config['optimization'][key]
        params_dict = self._get_params('RECO')
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']
        objective = self.config['optimization']['objective'].lower()

        print('Loading driver datasets...')
        _, _, tower_gpp, _, _ = self._load_gpp_data(filter_length)
        drivers, tower_reco, weights = self._load_reco_data(filter_length)
        # For simplicity and consistency with StochasticSampler.run(), the
        #   observed data and hyperparamters become part of the "driver" data
        drivers = [tower_reco, tower_gpp, *drivers, q_rh, q_k]

        print('Initializing sampler...')
        backend = self.config['optimization']['backend_template'].format(
            model = 'RECO', pft = self._pft)
        model_func = getattr( # e.g., L4CStochasticSampler._reco
            L4CStochasticSampler,
            self.config['optimization']['function']['RECO'])
        sampler = L4CStochasticSampler(
            self.config, model_func, params_dict,
            backend = backend, weights = weights)

        # Get (informative) priors for just those parameters that have them
        with open(self.config['optimization']['prior'], 'r') as file:
            prior = yaml.safe_load(file)
        prior_params = filter(
            lambda p: p in prior.keys(), sampler.required_parameters['RECO'])
        prior = dict([
            (p, dict([(k, v[self._pft]) for k, v in prior[p].items()]))
            for p in prior_params
        ])

        # For diagnostics
        if ipdb:
            trace = sampler.get_trace(
                burn = kwargs.get('burn', None), thin = kwargs.get('thin'))
            import ipdb
            ipdb.set_trace()

        # Set var_names to tell ArviZ to plot only the free parameters; i.e.,
        #   those with priors
        var_names = list(filter(
            lambda x: x in prior.keys(), sampler.required_parameters['RECO']))
        # Remove any kwargs that don't belong
        for k in list(kwargs.keys()):
            if k not in ('chains', 'draws', 'tune', 'scaling', 'save_fig', 'var_names'):
                del kwargs[k]
        sampler.run(
            tower_reco, drivers, prior = prior, save_fig = save_fig, **kwargs)


if __name__ == '__main__':
    import fire
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
