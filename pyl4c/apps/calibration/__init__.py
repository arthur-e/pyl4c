'''
Essential functions and classes for calibration. See, in particular:

- `OPT_BOUNDS`, which has lower and upper bounds on calibration parameters

**You must create a configuration JSON file before calibrating L4C.** There
is a template available in the directory:

    pyl4c/data/fixtures/files

The optimization routines here are best used with multiple `--trials`, i.e.,
multiple, random initial parameter values. This can help avoiding falling into
a local optimum when a better solution set is available. However, if you are
repeatedly seeing the message:

    Error in objective function; restarting...

Most likely, the bounds on your parameters are producing initial parameter
values that are totally unreasonable.

Note that calibration of L4C (i.e., updating the BPLUT) can be performed using
the command-line interface in `pyl4c.apps.calibration.main`, for example:

    # Build the scratch dataset needed for calibration
    python main.py setup

    # You can get a preview of what filtering the data would look like...
    python main.py pft <pft_number> filter-preview gpp <window_size>
    python main.py pft <pft_number> filter-preview reco <window_size>

    # Optionally, filter the tower datasets to remove spurious spikes
    python main.py filter-all gpp <window_size>
    python main.py filter-all reco <window_size>

    # Run the GPP calibration for a given Plant Functional Type (PFT)
    python main.py pft <pft_number> tune-gpp

    # Run the RECO calibration for a given Plant Functional Type (PFT)
    python main.py pft <pft_number> tune-reco

    # Finally, to dump the updated BPLUT into a CSV or Python pickle file...
    python main.py bplut pickle <output_path> --version-id=<version_id>

    # Get help on any command with --help, for example:
    python main.py plot-gpp --help
'''

import os
import pickle
import nlopt
import netCDF4 # Necessary to import this before HDF5 due to a bug
import h5py
import numpy as np
from collections import OrderedDict
from scipy import optimize
from pyl4c import suppress_warnings
from pyl4c.stats import detrend, rmsd, sum_of_squares
from pyl4c.science import k_mult

# Constrained optimization bounds
OPT_BOUNDS = {
    'gpp': ( # lue, tmin0, tmin1, vpd0, vpd1, smrz0, smrz1, ft0
        np.array((0.5, 230, 276,    0,  1501,  0,  30.1, 0.)), # Lower bound
        np.array((4.0, 275, 320, 1500, 10000, 30, 100,   1.))), # Upper bound
    'reco': ( # CUE, beta_tsoil, smsf0, smsf1
        np.array((0.0,   1,    0,  25)),
        np.array((0.7, 800, 24.9, 100)))
}


class BPLUT(object):
    '''
    Represents a Biome Properties Look-Up Table (BPLUT) with PFT classes along
    the rows and parameters along the columns.

    If initialized with a `params_dict`, these are the values of the BPLUT.
    If initialized with an `hdf5_path` but without a `params_dict`, the
    parameters are read-in from the HDF5 file.
    '''
    _labels = [
        'LUE', 'CUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1',
        'smsf0', 'smsf1', 'ft0', 'ft1', 'tsoil', 'decay_rates0',
        'decay_rates1', 'decay_rates2', 'f_metabolic', 'f_structural'
    ]
    _npft = 10 # Number of (possible) PFT classes
    _valid_pft = range(1, 9) # Canonical range of valid PFTs

    def __init__(
            self, params_dict = None, labels = None, hdf5_path = None,
            hdf5_group = 'BPLUT'):
        '''
        Parameters
        ----------
        params_dict : dict
            A BPLUT to initialize the new BPLUT
        labels : tuple or list
            Names of the parameters
        hdf5_path : str
            Path to an HDF5 file to use as a temporary store
        hdf5_group : str
            Field name with which to store data in HDF5 file
            (Default: "BPLUT")
        '''
        self.hdf5_group = hdf5_group
        self.hdf5_path = hdf5_path
        if labels is not None:
            print('WARNING: Parameter names ending with a number are assumed to be bounds on ramp functions!')
            self._labels = labels
        # Create an in-memory parameter dictionary
        empty = self._empty_dict(self.labels)
        if params_dict is None:
            init_data = empty # No prior BPLUT, use empty table
        else:
            init_data = params_dict.copy()
            # IMPORTANT: Make sure prior BPLUT has all the necessary params
            for key in empty.keys():
                init_data.setdefault(key, empty[key])
        # Optionally, maintain a file BPLUT
        if hdf5_path is not None:
            # Restore from the file data, filling in NaNs with initial
            if os.path.exists(hdf5_path):
                with h5py.File(hdf5_path, 'r') as hdf:
                    if hdf5_group in hdf.keys():
                        init_data = self.hdf5_restore(hdf, init_data = init_data)
            # Then, if file dataset doesn't exist, create a new one and store
            #   the initial data
            with h5py.File(hdf5_path, 'a') as hdf:
                self.hdf5_flush(hdf, data = init_data)
        self.data = init_data

    @property
    def data(self):
        'The parameters dictionary or `dict` instance'
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def labels(self):
        'Names of the free parameters'
        return self._labels

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def _empty_dict(self, labels = None, dtype = np.float32):
        '''
        Given sequence of labels, convert to legacy, human-readable dict,
        e.g.:

            { 'LUE': array([[ nan, 1.17, ..., nan ]]), ... }
        '''
        labels_dedupe = self._canonical(labels)
        result = dict()
        for name in labels_dedupe:
            size = len(list(filter(lambda x: x.startswith(name), labels)))
            result[name] = np.ones((size, self._npft), dtype = dtype) * np.nan
        return result

    def _canonical(self, labels = None):
        '''
        Returns a short list of labels, without duplicates. Specifically,
        a list of labels like, e.g., ("tmin0", "tmin1", "decay_rates1")
        becomes ("tmin", "decay_rates").

        Parameters
        ----------
        labels : tuple or list or None

        Returns
        -------
        list
        '''
        labels = self.labels if labels is None else labels
        # Remove numeric suffixes and de-duplicate the list
        return list(OrderedDict([
            (p.strip('0123456789'), 0) for p in labels
        ]).keys())

    def flat(self, pft, labels = None):
        '''
        Retrieves a flat list of parameters for a specific PFT.

        Parameters
        ----------
        pft : int
            Numeric code of the PFT for which to return parameter values
        labels : tuple or list
            (Optional) A sequence of parameter names desired, if not all;
            defaults to returning all parameters

        Returns
        -------
        numpy.ndarray
        '''
        labels_dedupe = self._canonical(labels)
        return np.hstack([
            self.data[p][:,pft].ravel() if p in self.data.keys() else np.nan
            for p in labels_dedupe
        ])

    def hdf5_flush(self, hdf, data = None):
        '''
        Writes the current BPLUT to an HDF5 file.

        Parameters
        ----------
        hdf : h5py.File
            HDF5 file open for writing
        data : dict
        '''
        assert hdf.mode != 'r', 'File not open for writing!'
        data = self.data if data is None else data
        if self.hdf5_group not in hdf.keys():
            hdf.create_group(self.hdf5_group)
        for key, value in data.items():
            if key.startswith('_'):
                continue # Skip keys that are not parameter names
            field = '%s/%s' % (self.hdf5_group, key)
            if key not in hdf[self.hdf5_group].keys():
                hdf.create_dataset(field, value.shape, np.float32, value)
            else:
                # Overwrite NaNs in the file data
                _value = hdf[field][:]
                hdf[field][:] = np.where(np.isnan(_value), value, _value)
        hdf.flush()

    def hdf5_restore(self, hdf, init_data = None, dtype = np.float32):
        '''
        Reads in the BPLUT table stored in the HDF5 file.

        Parameters
        ----------
        hdf : h5py.File
        init_data : dict
            Initital data; will be over-written by HDF5 file contents

        Returns
        -------
        dict
        '''
        data = dict() if init_data is None else init_data
        for key in hdf[self.hdf5_group].keys():
            # Update the in-memory BPLUT
            from_hdf5 = hdf[self.hdf5_group][key][:]
            data[key] = np.where(
                ~np.isnan(from_hdf5), from_hdf5, init_data.get(key)
            ).astype(dtype)
        return data

    def pickle(self, output_path, version_id = None):
        '''
        Writes the current BPLUT parameters, as a dictionary, to a pickle
        file.

        Parameters
        ----------
        output_path : str
            The output path for the pickle file (*.pickle)
        version_id : str
            (Optional) The version identifier for this BPLUT
        '''
        with open(output_path, 'wb') as file:
            output = self.data.copy()
            if version_id is not None:
                output['_version'] = version_id
            pickle.dump(output, file)

    def show(self, pft, param, precision = 2):
        '''
        Prints the current BPLUT parameters for a given PFT, the values of a
        given parameter for all PFTs, or the value of a specific PFT-parameter
        combination.

        Parameters
        ----------
        pft : int or None
            The PFT class
        param : str or None
            The name of the parameter
        precision : int
            Decimal precision to use for printing numbers
        '''
        assert not (pft is None and param is None),\
            'Either one or both must be specified: --pft or --param'
        set_of_labels = self._canonical() if param is None else [param]
        set_of_pfts = self._valid_pft if pft is None else [pft]
        for each in set_of_labels:
            assert each in self._canonical(), 'Unrecognized parameter: %s' % each
        for each in set_of_pfts:
            assert each in range(0, self._npft), 'PFT code out of range'
        pad = max(len(l) for l in set_of_labels) + 2
        fmt_string = '{:>%d} {:>%d}' % (pad, 5 + precision)
        for pft in set_of_pfts:
            print('BPLUT parameters for PFT %d:' % pft)
            for label in set_of_labels:
                param_values = self.data[label][:,pft]
                for i, value in enumerate(param_values):
                    # If there are multiple values for a parameter (group),
                    #   append a number to the end of the label
                    if len(param_values) > 1:
                        prefix = '%s%d:' % (label, i)
                    else:
                        prefix = '%s:' % label
                    print(
                        fmt_string.format(prefix, ('%%.%df' % precision) % value))

    def update(self, pft, values, labels, flush = True):
        '''
        Updates the BPLUT with the specified parameters for a single PFT.

        Parameters
        ----------
        pft : int
            The PFT class
        values : tuple or list
            Sequence of parameter values, one for each parameter named in
            `labels`
        labels : tuple or list
            Sequence of parameter names, one for each value in `values`
        flush : bool
            True to write the result to disk (attached HDF5 file storage)
            (Default: True)
        '''
        assert len(values) == len(labels),\
            'Vectors of values and parameter labels must have the same length'
        if flush:
            assert self.hdf5_path is not None,\
                'No HDF5 file storage is attached'
            hdf = h5py.File(self.hdf5_path, 'a')
        for i, name in enumerate(labels):
            abbrv = name.strip('0123456789')
            # In case parameter has multiple levels, like a ramp function
            #   (e.g., "smsf0" and "smsf1" are two rows)
            dupes = list(filter(lambda x: x.startswith(abbrv), self.labels))
            j = dupes.index(name) # If it has one level (e.g., "LUE"), j = 0
            self.data[abbrv][j,pft] = values[i]
            if flush:
                path = '%s/%s' % (self.hdf5_group, abbrv)
                hdf[path][:,pft] = self.data[abbrv][:,pft]
        if flush:
            hdf.flush()
            hdf.close()


class ModelParameters(OrderedDict):
    '''
    Convenience wrapper for an OrderedDict, allowing both vectorized and
    keyword access to model parameters.

    Parameters
    ----------
    group : str
        Name of this model parameters group, usually the name of the model or
        sub-model to which they belong
    *params : spotpy.parameter
        One or more parameters
    '''
    def __init__(self, group, *params):
        self._group = group
        # Create {name: spotpy.parameter, ...} dictionary
        super().__init__(**dict([(p.name, p) for p in params]))


class GenericOptimization(object):
    '''
    A more generic and expansive tool for optimization; includes many more
    algorithms for minimization/ maximization problems, including sequential
    quadratic programming (SQP), which is the default here and is closest to
    what is performed in Matlab's `fmincon`. Despite the similarity to `fmincon`,
    SQP will tend to deviate strongly from the initial parameters derived via
    fmincon. This solver is SLOW for gradient descent methods relative to
    `scipy.optimize.least_squares()`, because the gradient is calculated with
    a finite element approach.

        opt = GenericOptimization(residuals, OPT_BOUNDS['gpp'],
            step_size = (0.01, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.05))
        opt.solve(init_params)

    See: https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/

    Parameters
    ----------
    func : function
        Function to calculate the residuals
    bounds : list or tuple
        2-element sequence of (lower, upper) bounds where each element is an
        array
    method : str
        One of the nlopt algorithms
    step_size : list or tuple or numpy.ndarray
        Sequence of steps to take in gradient descent; not needed for
        derivative-free methods
    verbose : bool
        True to print all output to the screen
    '''
    def __init__(
            self, func, bounds, method: int = nlopt.LD_SLSQP,
            step_size = None, verbose = True):
        # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp
        assert isinstance(method, int), 'Did not recognize "method" argument'
        self._bounds = bounds
        self._method = method
        self._residuals = func
        self._step_size = step_size
        self._verbose = verbose

    def solve(self, init_params, ftol = 1e-8, xtol = 1e-8, maxeval = 500):
        '''
        Using the sum-of-squared errors (SSE) as the objective function,
        solves a minimization problem.

        Parameters
        ----------
        init_params : list or tuple or numpy.ndarray
            Sequence of starting parameters (or "initial guesses")
        ftol : float
        xtol : float
        maxeval : int
            Maximum number of objective function evaluations

        Returns
        -------
        numpy.ndarray
        '''
        @suppress_warnings
        def sse(x):
            return np.power(self._residuals(x), 2).sum()

        @suppress_warnings
        def objf(x, grad):
            if grad.size > 0:
                # Approximate the gradient using finite element method
                grad[...] = optimize.approx_fprime(
                    x, sse, self._step_size)
            return sse(x)

        opt = nlopt.opt(self._method, len(init_params))
        # https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#localsubsidiary-optimization-algorithm
        if self._method == nlopt.G_MLSL_LDS:
            opt.set_local_optimizer(
                nlopt.opt(nlopt.LN_COBYLA, len(init_params)))
        opt.set_min_objective(objf)
        opt.set_lower_bounds(self._bounds[0])
        opt.set_upper_bounds(self._bounds[1])
        opt.set_ftol_abs(ftol)
        opt.set_xtol_abs(xtol)
        opt.set_maxeval(maxeval)
        if self._verbose:
            print('Solving...')
        return opt.optimize(init_params)


def cbar(rh, k_mult, q_rh = 75, q_k = 50):
    '''
    Calculates "Cbar," the time-constant upper quantile of the RH/Kmult
    ratio. Where Kmult is >/= `q_k`, return the `q_rh` quantile of RH/Kmult;
    intended for T x N arrays where T is the number of time steps and
    N is the number of (flux tower) sites.

    Parameters
    ----------
    rh : numpy.ndarray
        (T x N) vector of heterotrophic respiration
    k_mult : numpy.ndarray
        (T x N) vector of Kmult
    q_rh : float
        Percentile of RH/Kmult to return
    q_k : float
        Percentile of Kmult below which RH/Kmult values are masked

    Returns
    -------
    numpy.float64
    '''
    cutoff = np.apply_along_axis(
        np.percentile, 0, k_mult, q = q_k).reshape((1, k_mult.shape[1]))
    return np.nanpercentile(
        np.where(k_mult >= cutoff,
            np.divide(rh, np.where(k_mult == 0, np.nan, k_mult)), np.nan),
        q = q_rh, axis = 0)


def reco(params, tsoil, smsf, reco_tower, gpp_tower, q_rh = 75, q_k = 50):
    '''
    Calculate empirical ecosystem respiration, RECO, based on current model
    parameters and the inferred soil organic carbon (SOC) storage; i.e., this
    calculation should be used in model calibration when SOC is not a priori
    known, see `pyl4c.apps.calibration.cbar()`. The expected model parameter
    names are "CUE" for the carbon use efficiency of plants.

    Parameters
    ----------
    params : dict
        A dict-like data structure with named model parameters
    tsoil : numpy.ndarray
        (T x N) vector of soil temperature (deg K), where T is the number of
        time steps, N the number of sites
    smsf : numpy.ndarray
        (T x N) vector of surface soil wetness (%), where T is the number of
        time steps, N the number of sites
    reco_tower : numpy.ndarray
        (T x N) vector of observed RECO from eddy covariance tower sites
    gpp_tower : numpy.ndarray
        (T x N) vector of observed GPP from eddy covariance tower sites
    q_rh : int
        The percentile of RH/Kmult to use in calculating Cbar
    q_k : int
        The percentile of Kmult below which RH/Kmult values are masked

    Returns
    -------
    numpy.ndarray
    '''
    # Calculate RH as (RECO - RA) or (RECO - (faut * GPP));
    #   globals "reco_tower", "gpp_tower"
    ra = ((1 - params['CUE']) * gpp_tower)
    rh = reco_tower - ra
    rh = np.where(rh < 0, 0, rh) # Mask out negative RH values
    # Compute Cbar with globals "q_rh" and "q_k"
    kmult0 = k_mult(params, tsoil, smsf)
    cbar0 = cbar(rh, kmult0, q_rh, q_k)
    return ra + (kmult0 * cbar0)


def report_fit_stats(obs, pred, weights = np.array([1]), verbose = True):
    '''
    Reports the RMSE, ubRMSE, and Bias for observed and predicted values.

    Parameters
    ----------
    obs : numpy.ndarray
        Vector of observed ("true") values
    pred : numpy.ndarray
        Vector of predicted values
    weights : numpy.ndarray
        (Optional) Vector of weights for each sample

    Returns
    -------
    tuple
        (R-squared, RMSE, ubRMSE, Bias)
    '''
    y = np.apply_along_axis(detrend, 0, obs, fill = True)
    yhat = np.apply_along_axis(detrend, 0, pred, fill = True)
    rmse = rmsd(obs, pred, weights = weights)
    ubrmse = rmsd(y, yhat, weights = weights)
    bias = np.nanmean(np.subtract(pred, obs))
    mask = np.logical_or(np.isnan(obs), np.isnan(pred))
    r_squared = 1 - np.divide(
        sum_of_squares(
            obs[~mask], pred[~mask], add_intercept = False, which = 'sse'),
        sum_of_squares(
            obs[~mask], pred[~mask], add_intercept = False, which = 'sst'))
    if verbose:
        print('Fit statistics:')
        print('--    R^2: %s' % ('%.3f' % r_squared).rjust(6))
        print('--   RMSE: %s' % ('%.3f' % rmse).rjust(6))
        print('-- ubRMSE: %s' % ('%.3f' % ubrmse).rjust(6))
        print('--   Bias: %s' % ('%.3f' % bias).rjust(6))
    return (r_squared, rmse, ubrmse, bias)


def solve_least_squares(func, init_params, labels, bounds, **kwargs):
    '''
    Apply constrained, non-linear least-squares optimization. Mostly a
    wrapper for `scipy.optimize.least_squares()`.

    Parameters
    ----------
    func : function
        Function to calculate the residuals
    init_params : list or tuple or numpy.ndarray
        Sequence of starting parameters (or "initial guesses")
    labels : list or tuple or numpy.ndarray
        Sequence of parameter names
    bounds : list or tuple
        2-element sequence of (lower, upper) bounds where each element is an
        array

    Returns
    -------
    scipy.optimize.OptimizeResult
    '''
    # Update the optimization settings; this loss function produces
    #   an estimate for the FT multiplier that is closest to prior
    kwargs.setdefault('loss', 'arctan')
    kwargs.setdefault('method', 'trf')
    kwargs.setdefault('max_nfev', 500)
    kwargs.setdefault('ftol', 1e-8)
    kwargs.setdefault('xtol', 1e-8)
    kwargs.setdefault('gtol', 1e-8)
    try:
        solution = optimize.least_squares(
            func, init_params, bounds = bounds, **kwargs)
    except ValueError:
        below = [
            labels[i]
            for i in np.argwhere(init_params < bounds[0]).flatten().tolist()
        ]
        above = [
            labels[i]
            for i in np.argwhere(init_params > bounds[1]).flatten().tolist()
        ]
        if np.isnan(init_params).any():
            raise ValueError(
                'Error in candidate parameter values; residual function probably returning NaNs')
        else:
            raise ValueError(
                '"Infeasibility" error; check lower bound on %s; upper bound on %s' % (
                '(None)' if len(below) == 0 else ', '.join(below),
                '(None)' if len(above) == 0 else ', '.join(above)))
    return solution
