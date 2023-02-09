'''
The L4C Science model, for computing a total carbon budget for specific
geographic point locations. This is a "point" version, similar to the Matlab
calibration code, that runs L4C for individual cells; it is NOT intended to
run L4C daily for the global domain.

The `L4CForwardProcessPoint` has the same model logic as the L4C operational
algorithm that is run at Goddard Space Flight Center. However, it does not
produce "bit-to-bit" identical results as L4C Ops, chiefly because of
differences in the spatial reference system (SRS) definitions used for
extracting and compiling driver data, but also because the initial soil
organic carbon (SOC) data does not match. This mismatch is hard to account
for; we've tried multiple SOC restart files but none of them match what
Goddard is using. It's possible this is also due entirely to the SRS mismatch.

Possible optimizations:

- Pre-compute the product of APAR and optimal LUE.
- Will break with L4C MDL (Ops) code, but it's more memory efficient
    to assign initial SOC pool sizes to t=0 and makes very (very) little
    difference in the end.
'''

import datetime
import numpy as np
import h5py
from pyl4c import suppress_warnings
from pyl4c.data.fixtures import restore_bplut
from pyl4c.science import arrhenius
from pyl4c.stats import linear_constraint
from pyl4c.lib.cli import ProgressBar
from pyl4c.apps.l4c import L4CConstants, L4CDrivers, L4CState, report
from pyl4c.apps.l4c.io import L4CStreamingInputDataset

class L4CForwardProcessPoint(object):
    '''
    Outputs of L4C include both State variables (soil organic carbon) and Flux
    variables (NEE, GPP, RH).

    NOTE: In `debug = True` mode, the Emult and Kmult diagnostic parameters
    are disaggregated and the individual environmental constraint multipliers
    are stored. Outside of debug mode, only Emult, Tmult, and Wmult are
    reported.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    stream : bool
        True to use L4CStreamingInputDataset instead of reading in all driver
        data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    DAYS_PER_YEAR = 365
    BOUNDS = {
        'apar': [0, np.inf],
        'vpd': [np.nan, np.nan],
        'ft': [0, 1],
        'smrz': [0, 100],
        'smsf': [0, 100],
        'tmin': [-200, 400],
        'tsoil': [-200, 400],
        'tsurf': [-200, 400],
    }
    CONSTANTS = {
        'tsoil_beta1': 66.02,
        'tsoil_beta2': 227.13,
        'tsurf_freeze-thaw_threshold_kelvin': 273.15
    }
    CONSTANTS_INDEX = (
        'CUE', 'LUE', 'f_metabolic', 'f_structural', 'decay_rates')
    # These MUST be listed in a fixed order...
    CONSTRAINTS_INDEX = ( # Constraints on GPP or RECO
        'tmin', 'vpd', 'ft', 'smrz', 'tsoil', 'smsf')
    DIAGNOSTICS_INDEX = (
        'f_tmin', 'f_vpd', 'f_ft', 'f_smrz', 'f_tsoil', 'f_smsf', 'apar')
    DRIVERS_INDEX = ('apar', 'tmin', 'vpd', 'ft', 'smrz', 'tsoil', 'smsf')
    FLUX_INDEX = ('gpp', 'rh', 'nee')
    STATE_INDEX = ('soc1', 'soc2', 'soc3', 'e_mult', 't_mult', 'w_mult')
    PFT_CODES = range(1, 10)
    REQUIRED_CONFIGURATION = ('bplut', 'inputs_file_path', 'site_count')

    def __init__(
            self, config, stream = True, use_legacy_pft = False,
            verbose = True, debug = False):
        self._check_configuration(config)
        self._bplut = config['bplut']
        self._config = config
        self._constraints = dict()
        self._debug = debug
        self._multilayer = False # Single soil layer
        self._streaming = stream
        self._t0 = 0 # Starting time index
        self._t1 = config['time_steps']
        self._time_idx = -1
        self._verbose = verbose
        self._use_legacy_pft = use_legacy_pft
        self.file_path = config['inputs_file_path']
        if verbose:
            print('NOTE: Running with BPLUT version %s' % self._bplut['_version'])
        with h5py.File(self.file_path, 'r') as hdf:
            self._setup(config, hdf)

    @property
    def config(self):
        return self._config

    @property
    def constants(self):
        return self._constants

    @property
    def drivers(self):
        return self._drivers # NOTE: No setter

    @property
    def fluxes(self):
        return self._fluxes

    @fluxes.setter
    def fluxes(self, new):
        self._fluxes.data = new

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new):
        self._state.data = new

    def _arrhenius(self, tsoil):
        'The Arrhenius equation for response of enzymes to (soil) temperature'
        beta1, beta2 = (self.CONSTANTS['tsoil_beta%d' % i] for i in (1, 2))
        return arrhenius(tsoil, self._constraints['tsoil'], beta1, beta2)

    def _load_constants(self):
        'Load arrays of constants for each PFT class, to speed up computation'
        for label in self.CONSTANTS_INDEX:
            self.constants.add(self._bplut[label], label)

    def _load_constraints(self, pft, drivers_constrained):
        'Creates [3 x N x 81] array of lower, upper bounds for ramp functions'
        # The third axis is the range (max - min), or difference between the
        #   first and second axes
        by_pft = pft.ravel()
        # Set invalid PFT codes to PFT 0
        by_pft = np.where(np.in1d(by_pft, self.PFT_CODES), by_pft, 0)
        shp = (2, self.config['site_count'], 81)

        # For each driver, propagate the BPLUT coefficients by PFT class
        for driver in drivers_constrained:
            # Tsoil requires a single coefficient; not lower, upper bounds
            if driver == 'tsoil':
                self._constraints['tsoil'] = np.apply_along_axis(
                    lambda p: self._bplut['tsoil'][0,p], 0, by_pft)\
                    .reshape((shp[1], shp[2]))
                continue

            # Get lower and upper bounds by PFT class
            self._constraints[driver] = np.apply_along_axis(
                lambda p: self._bplut[driver][:,p], 0, by_pft).reshape(shp)

    def _load_drivers(self, hdf, keys):
        'Load driver variables all at once from the HDF'
        new_drivers = []
        for key in keys:
            if key not in ('apar', 'ft'):
                assert key in hdf['drivers'].keys(),\
                    'Required driver "%s" not found in inputs HDF5' % key

            if key == 'apar':
                for k in ('fpar', 'par'):
                    assert k in hdf['drivers'].keys(),\
                        'Required driver "%s" not found in inputs HDF5' % k
                par = hdf['drivers/par'][self._t0:,...]
                par = par.reshape((*par.shape, 1)).repeat(81, axis = 2)
                # APAR is the product of fPAR and PAR
                new_drivers.append(
                    np.multiply(hdf['drivers/fpar'][self._t0:,...], par))
                continue

            if key == 'ft':
                # Use "ft" or "tsurf" depending on what's available to obtain
                #   a freeze-thaw record
                if 'ft' in hdf['drivers'].keys():
                    self._print('Using existing freeze-thaw driver data instead of "tsurf"')
                    new_drivers.append(hdf['drivers/ft'][self._t0:,...])
                    continue
                elif 'tsurf' in hdf['drivers'].keys():
                    k = self.CONSTANTS['tsurf_freeze-thaw_threshold_kelvin']
                    self._print('Calculating freeze-thaw condition using "tsurf" and cutoff of %f degrees K' % k)
                    new_drivers.append( # Frozen = 0, Thawed = 1
                        np.where(
                            hdf['drivers/tsurf'][self._t0:,...] <= k, 0, 1))
                    continue
                else:
                    raise ValueError('No freeze-thaw driver data found')

            # In all other cases
            new_drivers.append(hdf['drivers/%s' % key][self._t0:,...])

        return new_drivers

    @suppress_warnings
    def _load_state(self, hdf, keys):
        '''
        Load state variables (soil organic carbon) all at once from the HDF.
        An array large enough to hold model state is created ("new_state"),
        but also the initial state ("init_state") is created.
        '''
        shp = (1, self.config['time_steps'], self.config['site_count'], 81)
        init_state = []
        new_state = []
        for p, key in enumerate(keys):
            # Create an empty state array, allocated T time steps
            new_state.append(np.full(shp, np.nan))
            if key.startswith('soc'):
                arr = hdf['state/soil_organic_carbon'][p,...]
                # Filter out any NoData, which (should) only correspond to
                #   1-km subgrid pixels that are outside the PFT range [1, 8]
                init_state.append(
                    np.where(arr < 0, np.nan, arr).reshape((1, 1, *shp[2:])))
            else:
                if key not in self.DIAGNOSTICS_INDEX:
                    init_state.append(np.full((1, 1, *shp[2:]), np.nan))
        return (init_state, new_state)

    def _setup(self, config, hdf):
        'Load point site PFTs, state data, driver data'
        # Get the starting time index, if specified
        if 'start' in config.keys():
            if config['start'] is not None:
                try:
                    ts0 = datetime.datetime.strptime(
                        config['start'], '%Y-%m-%dT%H:%M:%S')
                except ValueError:
                    ts0 = datetime.datetime.strptime(
                        config['start'], '%Y-%m-%d')
                self._t0 = np.argwhere(
                    np.logical_and(np.logical_and(
                        hdf['time'][:,0] == ts0.year,
                        hdf['time'][:,1] == ts0.month),
                        hdf['time'][:,2] == ts0.day)
                    ).flatten().tolist().pop()
        if 'end' in config.keys():
            if config['end'] is not None:
                try:
                    ts1 = datetime.datetime.strptime(
                        config['end'], '%Y-%m-%dT%H:%M:%S')
                except ValueError:
                    ts1 = datetime.datetime.strptime(
                        config['end'], '%Y-%m-%d')
                self._t1 = np.argwhere(
                    np.logical_and(np.logical_and(
                        hdf['time'][:,0] == ts1.year,
                        hdf['time'][:,1] == ts1.month),
                        hdf['time'][:,2] == ts1.day)
                    ).flatten().tolist().pop()

        # And check that the correct number of steps were specified
        assert config['time_steps'] == (hdf['time'].shape[0] - self._t0)\
            or config['time_steps'] == (self._t1 - self._t0),\
            'Parameter "time_steps" does not match the number of time steps suggested by "start" parameter and the "time" field'

        self._print('Accessing state and drivers data...')
        if self._use_legacy_pft:
            self._pft = hdf['legacy/lc_dom'][:].swapaxes(0, 1)
        else:
            self._pft = hdf['state/PFT'][:]

        # Initialize containers for various datasets
        self._setup_data_storage(config, hdf)

        # Calculate daily litterfall based on the annual NPP sum
        self.constants.add(
            hdf['state/npp_sum'][:] / self.DAYS_PER_YEAR, 'litterfall')

        # SET STATE
        self._print('Loading state...')
        keys = list(self.STATE_INDEX)
        if self._debug:
            keys.extend(self.DIAGNOSTICS_INDEX)
        init_state, new_state = self._load_state(hdf, keys)
        if len(init_state) > 0:
            self.state_initial = np.concatenate(init_state, axis = 0)
        if len(new_state) > 0:
            self.state = np.concatenate(new_state, axis = 0)
        self.state.labels = keys

        # SET DRIVERS
        if not self._streaming:
            self._print('Loading drivers...')
            self._drivers = L4CDrivers(
                config['site_count'],
                self._load_drivers(hdf, self.DRIVERS_INDEX),
                labels = self.DRIVERS_INDEX)

        # LOAD CONSTANTS
        self._print('Loading constants...')
        self._load_constants()

        # INITIALIZE CONSTRAINT FUNCTIONS
        self._print('Creating linear constraint functions...')
        self._load_constraints(self._pft, self.CONSTRAINTS_INDEX)

    def _setup_data_storage(self, config, hdf):
        'Initialize containers for various datasets'
        site_names = hdf['site_id'][:].tolist()
        shp = (config['time_steps'], config['site_count'], 81)
        self._constants = L4CConstants(
            config['site_count'], self._pft, self.PFT_CODES)
        self._drivers = None # We set this only once, below
        self._fluxes = L4CState(
            config['site_count'], np.full((len(self.FLUX_INDEX), *shp), np.nan),
            self.FLUX_INDEX, axis_labels = [None, None, site_names, None])
        self._state = L4CState(config['site_count'],
            axis_labels = [None, None, site_names, None])

    def _print(self, message):
        'Only print to screen if in "verbose" mode'
        if self._verbose:
            print(message)

    def _check_configuration(self, config):
        'Checks for the existing of certain keys in the configuration file'
        for key in self.REQUIRED_CONFIGURATION:
            assert key in config.keys(), '%s not found' % key

    def constrain(self, x, driver):
        '''
        Returns a linear interpolated multiple based on a ramp function.
        Equivalent to a vectorized version of:

            if x >= xmax:
                return 1
            if x <= xmin:
                return 0
            return (x - xmin) / (xmax - xmin)

        Parameters
        ----------
        x : float
            Observed value, generally an (N x 81) array or an array with an
            (N x 81) or (N x 1) sub-space (e.g., soil moisture values that
            are constant over the 1-km grid have no 81 unique values), so they
            will be broadcast over the 81-pixel sub-grid when combined with
            constraint parameter array
        driver : str
            Name of the driver

        Returns
        -------
        float
        '''
        if driver == 'tsoil':
            return self._arrhenius(x)

        coefs = self._constraints[driver]
        if driver == 'vpd':
            # VPD mult. declines with increasing VPD, unlike other drivers
            return linear_constraint(coefs[0], coefs[1], form = 'reversed')(x)

        if driver == 'ft':
            # FT has a binary response
            return linear_constraint(coefs[0], coefs[1], form = 'binary')(x)

        return linear_constraint(coefs[0], coefs[1])(x)

    def gpp(self, drivers):
        '''
        Calculate GPP for a single time step.

        Parameters
        ----------
        drivers : list or tuple
            Nested sequence of (driver: `numpy.ndarray`, label: `str`) pairs

        Returns
        -------
        numpy.ndarray
        '''
        # Extract APAR, translate other drivers into environ. constraints
        apar, f_tmin, f_vpd, f_ft, f_smrz = [
            self.constrain(driver, label) if label != 'apar' else driver
            for driver, label in drivers
        ]
        e_mult = f_tmin * f_vpd * f_ft * f_smrz
        if self._debug:
            return (apar * self.constants.LUE * e_mult,
                (f_tmin, f_vpd, f_ft, f_smrz))
        return (apar * self.constants.LUE * e_mult, e_mult)

    def rh(self, state, drivers):
        '''
        Calculate RH for a single time step.

        Parameters
        ----------
        state : numpy.ndarray
            `(3 x N x M)` array of current SOC state in each pool
        drivers : list or tuple
            Nested sequence of (driver: `numpy.ndarray`, label: `str`) pairs

        Returns
        -------
        numpy.ndarray
        '''
        # Translate Tsoil and SMSF into environmental constraints on RH
        f_tsoil, f_smsf = [
            self.constrain(driver, label) for driver, label in drivers
        ]
        k_mult = (f_tsoil * f_smsf)[np.newaxis,...].repeat(3, axis = 0)
        # NOTE: These are true decay rates for 2nd and 3rd pools, so it
        #   is straightforward to multiply them against SOC
        rh = k_mult * self.constants.decay_rates * state
        # "the adjustment...to account for material transferred into the
        #   slow pool during humification" (Jones et al. 2017 TGARS, p.5);
        #   note that this is a loss FROM the "medium" (structural) pool,
        #   see tcfModFunc.c Lines 54-55
        rh[1,...] = rh[1,...] * (1 - self.constants.f_structural)
        # T_mult, W_mult same for each pool
        return (rh, (f_tsoil, f_smsf))

    def soc(self, rh, t = None):
        '''
        Calculate change in SOC for a single time step.

        Parameters
        ----------
        rh : numpy.ndarray
            `(3 x N x M)` array of RH at the current time step
        t : int
            Current time step; useful in subclasses but not used in the
            operational L4C algorithm (not used here)

        Returns
        -------
        numpy.ndarray
        '''
        # Change in SOC according to diff. eq. in Jones et al. (2017)
        litter = self.constants.litterfall
        dc1 = (litter * self.constants.f_metabolic) - rh[0,...]
        dc2 = (litter * (1 - self.constants.f_metabolic)) - rh[1,...]
        dc3 = (self.constants.f_structural * rh[1,...]) - rh[2,...]
        return (dc1, dc2, dc3)

    def run(self, steps = None, fields_gpp = None, fields_rh = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: `None`)
        fields_gpp : list or tuple or None
            (Optional) Sequence of field names that are used to drive the
            GPP model
        fields_rh : list or tuple or None
            (Optional) Sequence of field names that are used to drive the
            RH model
        '''
        @suppress_warnings
        def step(t, fields_gpp, fields_rh):
            'Calculate fluxes, new states for the next time step t'
            if t == 0:
                # Retrieve initial SOC pool sizes
                state = self.state_initial[0:3,0,...]
            else:
                # Retrieve SOC in each pool from prior step
                state = self.state.data[0:3,t-1,...]
            # Calculate fluxes, new states
            gpp, e_mult = self.gpp(
                zip(hdf.index(t + self._t0, *fields_gpp), fields_gpp))
            rh, k_mult = self.rh(
                state, zip(hdf.index(t + self._t0, *fields_rh), fields_rh))
            npp = gpp * self.constants.CUE
            d_soc = self.soc(rh, t + self._t0)
            # Record fluxes for this time step
            self.fluxes.update('gpp', t, gpp)
            self.fluxes.update('rh', t, rh.sum(axis = 0))
            self.fluxes.update('nee', t, rh.sum(axis = 0) - npp)
            # Record diagnostics at this time step
            if self._debug:
                # In debug mode, e_mult is not a single quantity but multiple;
                #   the order of e_mult_fields MUST match the order of the
                #   return signature for gpp()
                e_mult_fields = ('f_tmin', 'f_vpd', 'f_ft', 'f_smrz')
                if all(f in self.DIAGNOSTICS_INDEX for f in e_mult_fields):
                    for e, key in enumerate(self.DIAGNOSTICS_INDEX):
                        if key in e_mult_fields:
                            idx = e_mult_fields.index(key)
                            self.state.update(key, t, e_mult[idx])
                # In debug mode, k_mult is not a single quantity but multiple;
                #   the order of k_mult_fields MUST match the order of the
                #   return signature for rh()
                k_mult_fields = ('f_tsoil', 'f_smsf')
                if all(f in self.DIAGNOSTICS_INDEX for f in k_mult_fields):
                    for k, key in enumerate(self.DIAGNOSTICS_INDEX):
                        if key in k_mult_fields:
                            idx = k_mult_fields.index(key)
                            self.state.update(key, t, k_mult[idx])
                # IMPORTANT: Put e_mult back together again
                e_mult = e_mult[0] * e_mult[1] * e_mult[2] * e_mult[3]
                if 'apar' in self.DIAGNOSTICS_INDEX:
                    self.state.update( # Back-calculating APAR as GPP/(Emult*LUE)
                        'apar', t, np.divide(gpp,
                            np.multiply(e_mult, self.constants.LUE)))

            # Treat Kmult and Emult differently (i.e., break out the former
            #   but not the latter) because that is what L4C Ops does
            if 'e_mult' in self.STATE_INDEX:
                self.state.update('e_mult', t, e_mult)
            if 't_mult' in self.STATE_INDEX:
                self.state.update('t_mult', t, k_mult[0])
            if 'w_mult' in self.STATE_INDEX:
                self.state.update('w_mult', t, k_mult[1])
            # Update the SOC state
            for p in range(1, 4):
                if t == 0:
                    # At time t=0, we have no state to advance, so update
                    #   the state at t=0 based on the initial state
                    delta = np.add(self.state_initial[p-1,0,...], d_soc[p-1])
                    self.state.update(
                        'soc%d' % p, t, delta, bounds = (0, np.inf))
                else:
                    self.state.advance(
                        'soc%d' % p, t, d_soc[p-1], bounds = (0, np.inf))

        if fields_gpp is None:
            fields_gpp = ['apar', 'tmin', 'vpd', 'ft', 'smrz']
        if fields_rh is None:
            fields_rh = ['tsoil', 'smsf']
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS) as hdf:
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(t, fields_gpp, fields_rh)
                    self._time_idx += 1
                    progress.update(t)


if __name__ == '__main__':
    # Report on a potential driver dataset with: python main.py data.h5
    import sys
    from pyl4c.data.fixtures import BPLUT

    # Example configuration for all 356 sites in the post-launch period
    config = {
        'bplut': BPLUT,
        'inputs_file_path': sys.argv[1],
        'site_count': sys.argv[2],
        'time_steps': sys.argv[3],
        'start': None
    }

    with h5py.File(config['inputs_file_path'], 'r') as hdf:
        report(hdf, config)
