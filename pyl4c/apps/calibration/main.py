'''
Calibration procedure for SMAP Level-4 Carbon (L4C). For a list of all
commands, type:

    python main.py

**You must create a configuration JSON file before calibrating L4C.** This
file tells the calibration tool where on your file system the data files
are located. There is a template available in the directory:

    pyl4c/data/fixtures/files

The scratch data must be set-up before doing anything else:

    python main.py setup

Some commands can be chained together, for example, setting the PFT class to
calibrate is REQUIRED before most other commands; it can be set in one of
two ways:

    python main.py --pft=<pft> <command>
    python main.py pft <pft> <command>

Generally, the workflow for calibrating a single PFT is as follows:

    python main.py setup
    python main.py pft <pft> filter-preview gpp <window_size>
    python main.py pft <pft> filter         gpp <window_size>
    python main.py pft <pft> plot-gpp <driver>
    python main.py pft <pft> tune-gpp
    python main.py pft <pft> filter-preview reco <window_size>
    python main.py pft <pft> filter         reco <window_size>
    python main.py pft <pft> plot-reco <driver>
    python main.py pft <pft> tune-reco

Can optionally filter flux tower data for all PFTs:

    python main.py setup --reset
    python main.py filter-all gpp <window_size>
    python main.py filter-all reco <window_size>

**See the docstring on the CLI in this module for more information.**

Possible improvements:

- Replace parameter vectors with a
    `pyl4c.apps.calibration.ModelParameters` instance.
- Tower HDF5 file has upper-case field names (e.g., "APAR") while
    driver HDF5 file has lower-case field names (e.g., "par")
'''

import csv
import datetime
import itertools
import json
import os
import pickle
import h5py
import numpy as np
import pyl4c
from collections import Counter
from functools import partial
from scipy import signal
from matplotlib import pyplot
from pyl4c import pft_selector, suppress_warnings
from pyl4c.data.fixtures import PFT, restore_bplut
from pyl4c.science import arrhenius, climatology365
from pyl4c.stats import linear_constraint, detrend, rmsd
from pyl4c.lib.cli import ProgressBar
from pyl4c.apps.calibration import BPLUT, OPT_BOUNDS, GenericOptimization, cbar, report_fit_stats, solve_least_squares

CONFIG = os.path.join(os.path.dirname(pyl4c.__file__), 'data/files/config_calibration.json')

class CLI(object):
    '''
    Command line interface for calibrating L4C.

    To plot the response function of, e.g., Tmin, with (optional) suggested
    parameters (lower, upper bounds):

        python main.py pft <pft> plot-gpp <driver> [xmin, xmax]

    To optimize all of the GPP parameters:

        python main.py pft <pft> tune-gpp

    To optimize some of the GPP parameters, keeping named parameters fixed:

        python main.py pft <pft> tune-gpp --fixed="(LUE,ft0)"

    To optimize GPP with the best match for previous calibrations:

        python main.py pft <pft> tune-gpp --end="2014-12-31"

    To optimize GPP, first setting the initial value of a parameter:

        python main.py pft <pft> set <param> <value> tune-gpp

    Parameters
    ----------
    config : str
        Path to a configuration JSON file, which describes the file paths
        for calibration datasets
    pft : int
        The numeric code of the PFT to calibrate
    start : int
        (Optional) The numeric index making the start of a time series subset
        to use; if not provided, entire time series (in dataset) is used
    end : int
        (Optional) The numeric index making the end of a time series subset
        to use; if not provided, entire time series (in dataset) is used
    debug : bool
    use_legacy_pft : bool
        (Optional) True to use the L4C Nature Run v7.2 "legacy" PFT map
        ("lc_dom") for the PFT class assignments of each pixel (Default: True)
    '''
    _driver_bounds = {'apar': (2, np.inf)}
    _metadata = {
        'tmin': {'units': 'deg K'},
        'vpd': {'units': 'Pa'},
        'smrz': {'units': '%'},
        'smsf': {'units': '%'},
        'tsoil': {'units': 'deg K'},
    }
    _parameters = {
        'gpp': (
            'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0'
        ),
        'reco': (
            'CUE', 'tsoil', 'smsf0', 'smsf1'
        )
    }
    _required_drivers = [
        'fpar', 'par', 'smrz', 'smsf', 'tsoil', 'tmin', 'vpd', 'tsurf'
    ]

    def __init__(
            self, config = CONFIG, pft = None, start = None, end = None,
            debug = True, use_legacy_pft = True):
        self._debug = debug
        self._nsites = 0
        self._nsteps = 0
        self._pft = pft
        self._pft_map = None
        self._site_weights = None
        self._time_end = None
        self._time_start = None
        self._use_legacy_pft = use_legacy_pft

        # Read in configuration file to determine file paths
        with open(config, 'r') as file:
            config_data = json.load(file)
        self._path_to_bplut = config_data['BPLUT_file']
        self._path_to_drivers = config_data['drivers_file']
        self._path_to_scratch = config_data['scratch_file']
        self._path_to_towers = config_data['towers_file']
        self._check() # Check for required keys in driver data

        # Allow users to specify start and end dates for the time series
        if start is not None or end is not None:
            with h5py.File(self._path_to_drivers, 'r') as hdf:
                time_series = [
                    datetime.datetime(y, m, d).strftime('%Y-%m-%d')
                    for y, m, d, _ in hdf['time'][:].tolist()
                ]
            if start is not None:
                self._time_start = time_series.index(start)
            if end is not None:
                self._time_end = time_series.index(end)

        # Creates the BPLUT store
        self._init_bplut()

        # Read in PFT map
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            self._nsites = hdf['state/PFT'].shape[0]
            self._nsteps = hdf['time'].shape[0]
            if self._use_legacy_pft:
                self._pft_map = hdf['legacy/lc_dom'][:].swapaxes(0, 1)
            else:
                self._pft_map = hdf['state/PFT'][:]

    @property
    def _is_setup(self):
        return os.path.exists(self._path_to_scratch)

    def _bounds(self, init_params, group, fixed = None, bounds = OPT_BOUNDS):
        'Defines bounds; optionally "fixes" parameters by fixing bounds'
        params = init_params
        if fixed is not None:
            params = [ # If the given parameter is in "fixed", restrict bounds
                None if self._parameters[group][i] in fixed else init_params[i]
                for i in range(0, len(self._parameters[group]))
            ]
        lower = []
        upper = []
        for i, p in enumerate(params):
            # This is a parameter to be optimized; use default bounds
            if p is not None:
                lower.append(bounds[group][0][i])
                upper.append(bounds[group][1][i])
            else:
                lower.append(init_params[i] - 1e-3)
                upper.append(init_params[i] + 1e-3)
        return (np.array(lower), np.array(upper))

    def _check(self):
        'Checks for all the required keys in the driver dataset'
        msg = 'Missing required driver: %s'
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            for key in self._required_drivers:
                assert key in hdf['drivers'].keys(), msg % key
            assert hdf['drivers/smsf'][:].max() > 1,\
                'SMSF data may not be in percent saturation units!'
            assert hdf['drivers/smrz'][:].max() > 1,\
                'SMRZ data may not be in percent saturation units!'

    @suppress_warnings
    def _climatology(self):
        'Computes a 365-day climatology for each driver'
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            time_series = [
                datetime.datetime(y, m, d)
                for y, m, d, _ in hdf['time'][:].tolist()
            ]
            with h5py.File(self._path_to_scratch, 'a') as target_hdf:
                for key in self._required_drivers:
                    if 'climatologies' in target_hdf.keys():
                        if key in target_hdf['climatologies'].keys():
                            continue # Skip to next

                    if self._debug:
                        print('Calculating %s climatology...' % key)
                    shp = (365, *hdf['drivers/%s' % key].shape[1:])
                    target_hdf.create_dataset(
                        'climatologies/%s' % key, shp,
                        dtype = hdf['drivers/%s' % key].dtype,
                        data = climatology365(
                            hdf['drivers/%s' % key][:], time_series))

    def _constrain(self, x, driver, coefs = None):
        'Converts a driver x into a multiple on [0, 1].'
        if driver == 'tsoil':
            return arrhenius(x, self.bplut['tsoil'][0,self._pft])
        # User can provide an M-element array-like
        if coefs is None:
            coefs = self.bplut[driver][:,self._pft].tolist()
        constraint = linear_constraint(coefs[0], coefs[1])
        if driver == 'vpd':
            # VPD mult. declines with increasing VPD, unlike other drivers
            constraint = linear_constraint(coefs[0], coefs[1], 'reversed')
        elif driver == 'ft':
            # FT has a binary response
            constraint = linear_constraint(coefs[0], coefs[1], 'binary')
        return constraint(x)

    def _filter(self, raw, size):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        else:
            return raw # Or, revert to the raw data

    def _init_bplut(self, labels = None):
        # Create the output BPLUT store; if there's already a BPLUT table
        #   in the scratch, it will be combined with the initial BPLUT,
        #   overwriting INTITIAL_BPLUT values in favor of the file BPLUT
        self.bplut = BPLUT(
            restore_bplut(self._path_to_bplut), labels = labels,
            hdf5_path = self._path_to_scratch)

    def _ramp(self, x, driver, step = 0.1, coefs = None):
        'Returns a ramp function over the domain of x; returns (domain, ramp)'
        domain = np.arange(np.nanmin(x), np.nanmax(x), step)
        ramp = self._constrain(domain, driver, coefs)
        return (domain, ramp)

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

    @suppress_warnings
    def _report_fit(self, obs, pred, weights, verbose = True):
        'Reports RMSE and unbiased RMSE'
        return report_fit_stats(obs, pred, weights, verbose = verbose)

    @property
    def _sites(self):
        '''
        For a given PFT class, returns the tower sites, as rank indices, that
        represent that PFT. Exceptions are made according to the L4C
        calibration protocol, e.g., sites with any amount of
        Deciduous Needleleaf (DNF) in their 1-km subgrid are considered to
        represent the DNF PFT class.
        '''
        return pft_selector(self._pft_map, self._pft)

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

    def filter(self, flux, size = 2):
        '''
        Filters the tower GPP or RECO flux data. Each time it is called, the
        raw (noisy) tower data is filtered; successive calls with the same
        arguments will only overwrite the filtered tower data.

        Parameters
        ----------
        flux : str
            "gpp" or "reco" for the corresponding tower dataset
        size : int
            Filter window size, in days
        '''
        assert any(self._sites), 'You must select a PFT class'
        with h5py.File(self._path_to_scratch, 'a') as hdf:
            assert flux.upper() in hdf['tower'].keys(),\
                'Field "tower/%s" not found' % flux.upper()
            # Tower fluxes are duplicated on the last axis, so pick one
            raw = hdf['tower/%s' % flux.upper()][:,self._sites,0]
            filtered = self._filter(raw, size)
            # Re-duplicate the 1-km subgrid on the last axis
            filtered = filtered[...,np.newaxis].repeat(81, axis = 2)
            hdf['tower/%s' % flux.upper()][:,self._sites,:] = filtered

    def filter_all(self, flux, size = 2):
        '''
        Filters the tower GPP or RECO flux data for ALL PFT classes.
        See also: filter().

        Parameters
        ----------
        flux : str
            "gpp" or "reco" for the corresponding tower dataset
        size : int
            Filter window size, in days
        '''
        with h5py.File(self._path_to_scratch, 'a') as hdf:
            assert flux.upper() in hdf['tower'].keys(),\
                'Field "tower/%s" not found' % flux.upper()
            # Tower fluxes are duplicated on the last axis, so pick one
            filtered = self._filter(
                hdf['tower/%s' % flux.upper()][...,0], size)
            # Re-duplicate the 1-km subgrid on the last axis
            filtered = filtered[...,np.newaxis].repeat(81, axis = 2)
            hdf['tower/%s' % flux.upper()][:] = filtered

    def filter_preview(self, flux, size = 2, seed = 9):
        '''
        Previews a filter for the tower GPP or RECO data.

        Parameters
        flux : str
            "gpp" or "reco" for the corresponding tower dataset
        size : int
            Filter window size, in days
        seed : int
            Random seed; change to select a different tower site
        '''
        assert self._sites.any(), 'You must select a PFT class'
        with h5py.File(self._path_to_scratch, 'r') as hdf:
            assert flux.upper() in hdf['tower'].keys(),\
                'Field "%s" not found' % flux.upper()
            x = hdf['tower/%s' % flux.upper()][:]

        # Get a random tower site matching the current PFT class
        np.random.seed(seed)
        idx = np.random.choice(np.argwhere(self._sites).flatten(), 1)[0]
        x0 = x[:,idx,0]
        x_filt = self._filter(x0, size)
        pyplot.plot(x0, c = 'lightsteelblue')
        pyplot.plot(x_filt, c = 'darkorange')
        pyplot.ylabel('Tower %s (g C m-2 day-1)' % flux.upper())
        pyplot.title('Window Size = %d days' % size)
        pyplot.show()

    def plot_gpp(
            self, driver, coefs = None, xlim = None, ylim = None, alpha = 0.1,
            marker = '.'):
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

        assert self._is_setup, 'Must run setup first'
        np.seterr(invalid = 'ignore')
        # Read in GPP and APAR data
        if coefs is not None:
            assert hasattr(coefs, 'index') and not hasattr(coefs, 'title'),\
                "Argument --coefs expects a list [values,] with NO spaces"
        assert driver in ('tmin', 'vpd', 'smrz'),\
            'Requested driver "%s" cannot be plotted for GPP' % driver
        with h5py.File(self._path_to_scratch, 'r') as hdf:
            gpp = hdf['tower/GPP'][:,self._sites,:].mean(axis = 2)
            apar = hdf['APAR'][:,self._sites,:].mean(axis = 2)

        with h5py.File(self._path_to_drivers, 'r') as hdf:
            # Get the user-specified driver data
            x = hdf['drivers/%s' % driver][:,self._sites]

        # Update plotting parameters
        lue = empirical_lue(apar, gpp)
        # Mask out negative LUE values and values with APAR<2
        pyplot.scatter(x, np.where(
            np.logical_or(lue == 0, apar < 2), np.nan, lue),
            alpha = alpha, marker = marker)
        a, b = self._ramp(x, driver)
        pyplot.plot(a, b * self.bplut['LUE'][:,self._pft], 'k-')
        if coefs is not None:
            pyplot.plot(*self._ramp(x, driver, coefs = coefs), 'r-')
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
            self, driver, coefs = None, q_rh = 75, q_k = 50, xlim = None,
            ylim = None, alpha = 0.1, marker = '.'):
        '''
        Using the current or optimized BPLUT coefficients, plots the RECO ramp
        function for a given driver. The ramp function is shown on a plot of
        RH/Cbar, which is equivalent to Kmult (as Cbar is an upper quantile of
        the RH/Kmult distribution).

        Parameters
        ----------
        driver : str
            Name of the driver to plot on the horizontal axis
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
        assert self._is_setup, 'Must run setup first'
        assert driver in ('tsoil', 'smsf'),\
            'Requested driver "%s" cannot be plotted for RECO' % driver
        np.seterr(invalid = 'ignore')
        with h5py.File(self._path_to_scratch, 'r') as hdf:
            gpp = hdf['tower/GPP'][:,self._sites,:].mean(axis = 2)
            reco = hdf['tower/RECO'][:,self._sites,:].mean(axis = 2)

        with h5py.File(self._path_to_drivers, 'r') as hdf:
            # Get the user-specified driver data
            tsoil = hdf['drivers/tsoil'][:,self._sites]
            smsf = hdf['drivers/smsf'][:,self._sites]

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
            pyplot.plot(*self._ramp(smsf, driver), 'k-')

        if coefs is not None:
            if driver == 'tsoil':
                pyplot.plot(domain, arrhenius(domain, *coefs), 'r-')
            elif driver == 'smsf':
                pyplot.plot(*self._ramp(smsf, driver, coefs = coefs), 'r-')

        pyplot.xlabel('%s (%s)' % (driver, self._metadata[driver]['units']))
        pyplot.ylabel('RH/Cbar')
        if xlim is not None:
            pyplot.xlim(xlim[0], xlim[1])
        if ylim is not None:
            pyplot.ylim(ylim[0], ylim[1])
        pyplot.title(
            '%s (PFT %d): RECO Response to "%s"' % (
                PFT[self._pft][0], self._pft, driver))
        pyplot.show()

    def reset(self):
        '''
        Clears the current BPLUT in the scratch data, wiping out any
        calibrated parameters. Subsequent calibration will then use the
        reference (file) BPLUT for initial values.
        '''
        with h5py.File(self._path_to_scratch, 'a') as hdf:
            del hdf['BPLUT']
        print('Deleted BPLUT in the scratch file')

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

    def setup(self, reset = False):
        '''
        Compute freeze/thaw, copy tower data, calculate site weights, and
        compute a 365-day climatology for each driver.

        Important considerations:
            1) Tower GPP and RECO values <0 are masked;
            2) Tower GPP is masked where APAR is <0.1 MJ m-2 d-1;
            3) Freeze/Thaw status is determiend as "frozen" (0) if the surface
               skin temperature is below 273.15 K, "thawed" (1) otherwise

        Parameters
        ----------
        reset : bool
            True to automatically delete the scratch file
        '''
        np.seterr(invalid = 'ignore')
        if reset:
            os.remove(self._path_to_scratch)
            print('Deleted scratch data.')

        # Open the scratch file
        target_hdf = h5py.File(self._path_to_scratch, 'a')
        # Copy tower flux datasets
        with h5py.File(self._path_to_towers, 'r') as hdf:
            for key in ('GPP', 'RECO', 'NEE'):
                arr = hdf[key][:]
                arr = arr[...,np.newaxis].repeat(81, axis = 2)
                # Mask out negative GPP and RECO values
                if key != 'NEE':
                    arr = np.where(np.less(arr, 0), np.nan, arr)
                target_hdf.create_dataset(
                    'tower/%s' % key, arr.shape, arr.dtype, arr)

        # Read in Tsurf, 9-km grid indices, and PAR + fPAR
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            tsurf = hdf['drivers/tsurf'][:]
            idx = hdf['coords/grid_9km_idx'][:].round(0).astype(np.int16)
            # Make conformable PAR array at 1-km scale
            par = hdf['drivers/par'][:]
            par = par[...,np.newaxis].repeat(81, axis = 2)
            apar = np.multiply(hdf['drivers/fpar'][:], par)
            par = None

        # Compute freeze/thaw
        target_hdf.create_dataset( # Frozen = 0, Thawed = 1
            'ft', tsurf.shape, tsurf.dtype, np.where(tsurf <= 273.15, 0, 1))
        tsurf = None
        # Find duplicate (shared) 9-km cells, calculate site weights
        uid = ['%d%d' % (r, c) for r, c in idx.tolist()]
        target_hdf.create_dataset('site_weights', (1, self._nsites),
            np.float16, np.array([(1.0 / uid.count(x)) for x in uid]))
        # Copy APAR; screen tower GPP where APAR is low
        gpp = target_hdf['tower/GPP'][:]
        target_hdf.create_dataset('APAR', apar.shape, apar.dtype, apar)
        # With APAR below 0.1, tower GPP is unstable
        target_hdf['tower/GPP'][:] = np.where(apar < 0.1, np.nan, gpp)
        apar = gpp = None
        # Close the scratch data
        target_hdf.close()
        # Compute climatology
        self._climatology()

    def tune_gpp(
            self, fixed = None, optimize = True, nlopt = True, trials = 1):
        '''
        Optimizes GPP. The 9-km mean L4C GPP is fit to the tower-observed GPP
        using constrained, non-linear least-squares optimization.

        Parameters
        ----------
        fixed : tuple or list
            Zero or more parameters whose values should be fixed
            (i.e, NOT optimized)
        optimize : bool
            False to only report parameters and their fit statistics instead
            of optimizing (Default: True)
        nlopt : bool
            True to use the nlopt library for optimization (Default: True)
        '''
        def e_mult(params):
            # Calculate E_mult based on current parameters
            f_tmin = linear_constraint(params[1], params[2])
            f_vpd  = linear_constraint(params[3], params[4], 'reversed')
            f_smrz = linear_constraint(params[5], params[6])
            f_ft   = linear_constraint(params[7], 1.0, 'binary')
            tmin, vpd, smrz, ft = drivers # Unpack global "drivers"
            e = f_tmin(tmin) * f_vpd(vpd) * f_smrz(smrz) * f_ft(ft)
            return e[...,np.newaxis]

        def gpp(params):
            # Calculate GPP based on the provided BPLUT parameters
            return apar * params[0] * e_mult(params) # Global "apar"

        def residuals(params):
            # Objective function: Difference between tower GPP and L4C GPP
            gpp0 = gpp(params).mean(axis = 2)
            diff = np.subtract(gpp_tower, gpp0) # Global "gpp_tower"
            # Multiply by the tower weights
            return (weights * diff)[~np.isnan(diff)] # Global "weights"

        assert self._is_setup, 'Must run setup first'
        # Compile the initial parameters for optimization, based on current
        #   BPLUT; they are: lue, tmin0, tmin1, vpd0, vpd1, smrz0, smrz1, ft0
        if fixed is not None:
            # In case a single parameter name is passed, wrap in a list
            fixed = [fixed] if hasattr(fixed, 'title') else fixed
            assert all(p in self._parameters['gpp'] for p in fixed),\
                'Arguments to "fixed" should be in: [%s]' % ', '.join(self._parameters['gpp'])
        init_params = [self.bplut['LUE'][0,self._pft]]
        for field in ('tmin', 'vpd', 'smrz', 'ft'):
            init_params.extend(self.bplut[field][:,self._pft].tolist())
        init_params.pop() # Remove FT upper limit (it is fixed at 1.0)

        # Read in data, with optional subsetting of the time axis
        t0 = self._time_start if self._time_start is not None else 0
        t1 = self._time_end if self._time_end is not None else self._nsteps
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            drivers = []
            for field in ('tmin', 'vpd', 'smrz'):
                drivers.append(hdf['drivers/%s' % field][t0:t1][:,self._sites])

        with h5py.File(self._path_to_scratch, 'r') as hdf:
            drivers.append(hdf['ft'][t0:t1][:,self._sites])
            apar = hdf['APAR'][t0:t1][:,self._sites,:]
            gpp_tower = hdf['tower/GPP'][t0:t1][:,self._sites,:].mean(axis = 2)
            weights = hdf['site_weights'][:][:,self._sites]

        # L4C drivers should have no NaNs, based on how they were sourced;
        #   BUT this is not the case for fPAR, which does have NaNs; not sure
        #   why I would have ever asserted it doesn't
        # assert np.all(~np.isnan(apar)), 'Unexpected NaNs in APAR'
        for arr in drivers:
            assert np.all(~np.isnan(arr)), 'Unexpected NaNs'

        # Get bounds for the parameter search
        bounds = self._bounds(init_params, 'gpp', fixed)
        params = []
        params0 = []
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
            if optimize and not nlopt:
                # Apply constrained, non-linear least-squares optimization
                print('Solving...')
                solution = solve_least_squares(
                    residuals, init_params, labels = self._parameters['gpp'],
                    bounds = self._bounds(init_params, 'gpp', fixed),
                    loss = 'arctan')
                fitted = solution.x.tolist()
                print(solution.message)
            elif optimize and nlopt:
                opt = GenericOptimization(residuals, bounds,
                    step_size = (0.01, 0.2, 0.2, 1, 1, 0.1, 0.1, 0.01))
                fitted = opt.solve(init_params)
            else:
                fitted = [None for i in range(0, len(init_params))]
            # Record the found solution and its goodness-of-fit score
            params.append(fitted)
            _, rmse_score, _, _ = self._report_fit(
                gpp_tower,
                gpp(fitted if optimize else init_params).mean(axis = 2),
                weights, verbose = False)
            print('[%s/%s] RMSE score of last trial: %.3f' % (
                str(t + 1).zfill(2), str(trials).zfill(2), rmse_score))
            scores.append(rmse_score)

        # Select the fit params with the best score
        if trials > 1:
            fitted = params[np.argmin(scores)]
            init_params = params0[np.argmin(scores)]
        # Generate and print a report, update the BPLUT parameters
        self._report(
            init_params, fitted, self._parameters['gpp'], 'GPP Optimization')
        self._report_fit(
            gpp_tower, gpp(fitted if optimize else init_params).mean(axis = 2),
            weights)
        if optimize:
            self.bplut.update(self._pft, fitted, self._parameters['gpp'])

    def tune_reco(
            self, q_rh = 75, q_k = 50, fixed = None, optimize = True,
            nlopt = True, trials = 1):
        '''
        Optimizes RECO. The 9-km mean L4C RECO is fit to the tower-observed
        RECO using constrained, non-linear least-squares optimization.
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
        '''
        def k_mult(params):
            # Calculate K_mult based on current parameters
            f_tsoil = partial(arrhenius, beta0 = params[1])
            f_smsf  = linear_constraint(params[2], params[3])
            tsoil, smsf = drivers # Unpack global "drivers"
            return f_tsoil(tsoil) * f_smsf(smsf)

        @suppress_warnings
        def reco(params):
            # Calculate RH as (RECO - RA) or (RECO - (faut * GPP));
            #   globals "reco_tower", "gpp_tower"
            ra = ((1 - params[0]) * gpp_tower)
            rh = reco_tower - ra
            rh = np.where(rh < 0, 0, rh) # Mask out negative RH values
            # Compute Cbar with globals "q_rh" and "q_k"
            kmult0 = k_mult(params)
            cbar0 = cbar(rh, kmult0, q_rh, q_k)
            return ra + (kmult0 * cbar0)

        def residuals(params):
            # Objective function: Difference between tower RECO and L4C RECO
            reco0 = reco(params)
            diff = np.subtract(reco_tower, reco0) # Global "reco_tower"
            missing = np.logical_or(np.isnan(reco_tower), np.isnan(reco0))
            # Multiply by the tower weights
            return (weights * diff)[~missing] # Global "weights"

        assert self._is_setup, 'Must run setup first'
        assert q_rh >= 0 and q_rh <= 100 and q_k >= 0 and q_k <= 100,\
            'Invalid setting for "q_rh" or "q_k" parameters'
        if fixed is not None:
            assert all(p in self._parameters['reco'] for p in fixed),\
                'Arguments to "fixed" should be in: [%s]' % ', '.join(self._parameters['reco'])
        init_params = [self.bplut['CUE'][0,self._pft]]
        for field in ('tsoil', 'smsf'):
            init_params.extend(self.bplut[field][:,self._pft].tolist())

        # Read in data, with optional subsetting of the time axis
        t0 = self._time_start if self._time_start is not None else 0
        t1 = self._time_end if self._time_end is not None else self._nsteps
        with h5py.File(self._path_to_drivers, 'r') as hdf:
            drivers = [
                hdf['drivers/%s' % field][t0:t1][:,self._sites]
                for field in ('tsoil', 'smsf')
            ]
        with h5py.File(self._path_to_scratch, 'r') as hdf:
            gpp_tower = hdf['tower/GPP'][t0:t1][:,self._sites,:].mean(axis = 2)
            reco_tower = hdf['tower/RECO'][t0:t1][:,self._sites,:].mean(axis = 2)
            weights = hdf['site_weights'][:,self._sites]

        # L4C drivers should have no NaNs, based on how they were sourced
        for arr in drivers:
            assert np.all(~np.isnan(arr)), 'Unexpected NaNs'

        # Get bounds for the parameter search
        bounds = self._bounds(init_params, 'reco', fixed)
        params = []
        params0 = []
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
            if optimize and not nlopt:
                # Apply constrained, non-linear least-squares optimization
                solution = solve_least_squares(
                    residuals, init_params, labels = self._parameters['reco'],
                    bounds = bounds, loss = 'arctan')
                fitted = solution.x.tolist()
                message = solution.message
            elif optimize and nlopt:
                opt = GenericOptimization(residuals, bounds,
                    step_size = (0.01, 1, 0.1, 0.1))
                fitted = opt.solve(init_params)
                message = 'Success'
            else:
                fitted = [None for i in range(0, len(init_params))]
            # Record the found solution and its goodness-of-fit score
            params.append(fitted)
            _, rmse_score, _, _ = self._report_fit(
                reco_tower, reco(fitted if optimize else init_params),
                weights, verbose = False)
            print('[%s/%s] RMSE score of last trial: %.3f' % (
                str(t + 1).zfill(2), str(trials).zfill(2), rmse_score))
            scores.append(rmse_score)

        # Select the fit params with the best score
        if trials > 1:
            fitted = params[np.argmin(scores)]
            init_params = params0[np.argmin(scores)]
        # Generate and print a report, update the BPLUT parameters
        self._report(
            init_params, fitted, self._parameters['reco'], 'RECO Optimization')
        self._report_fit(
            reco_tower, reco(fitted if optimize else init_params), weights)
        if optimize:
            self.bplut.update(self._pft, fitted, self._parameters['reco'])


if __name__ == '__main__':
    import fire
    fire.Fire(CLI)
