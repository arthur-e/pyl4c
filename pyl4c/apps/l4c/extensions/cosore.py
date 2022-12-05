'''
'''

import pickle
import datetime
import numpy as np
import h5py
from pyl4c import suppress_warnings
from pyl4c.science import ordinals365
from pyl4c.stats import linear_constraint
from pyl4c.lib.cli import ProgressBar
from pyl4c.apps.l4c import L4CStratifiedState
from pyl4c.apps.l4c.io import L4CStreamingInputDataset
from pyl4c.apps.l4c.main import L4CForwardProcessPoint
from pyl4c.apps.l4c.extensions.hydrology import O2DiffusionLimitMixin, StratifiedRespirationMixin, StratifiedLitterfallMixin

class LitterfallMixin:
    '''
    Implements support for a litterfall phenology in the soil organic carbon
    (SOC) decomposition model as a mix-in. Should not be instantiated directly.
    '''
    def soc(self, rh, t):
        '''
        Calculate change in SOC for a single time step.

        Parameters
        ----------
        rh : numpy.ndarray
            (3 x N x M) array of RH at the current time step
        t : int
            Current time step

        Returns
        -------
        numpy.ndarray
        '''
        # Change in SOC according to diff. eq. in Jones et al. (2017)
        doy = self._doy[t] - 1 # Get DOY on [1,365] then on [0,364] for Python
        litter = self.constants.litterfall[doy,...]
        dc1 = (litter * self.constants.f_metabolic) - rh[0,...]
        dc2 = (litter * (1 - self.constants.f_metabolic)) - rh[1,...]
        dc3 = (self.constants.f_structural * rh[1,...]) - rh[2,...]
        return (dc1, dc2, dc3)


class L4CPrescribedGPPModel(L4CForwardProcessPoint):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data.

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
    DRIVERS_INDEX = ('tsoil', 'smsf')
    FLUX_INDEX = ('rh', 'nee')
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_smsf')
    REQUIRED_CONFIGURATION = ('bplut', 'inputs_file_path', 'site_count', 'soc_data_path')

    def __init__(
            self, config, stream = True, use_legacy_pft = False,
            verbose = True, debug = False):
        super().__init__(
            config = config, stream = stream, use_legacy_pft = use_legacy_pft,
            verbose = verbose, debug = debug)

        if config['extra_parameters'] is not None:
            with open(config['extra_parameters'], 'rb') as file:
                data_dict = pickle.load(file)
            # Identify the extra parameters
            params = set(data_dict.keys()).difference(self._bplut.keys())
            # Set invalid PFT codes to PFT 0
            self._pft = np.where(self._pft > max(self.PFT_CODES), 0, self._pft)
            # Vectorize the new parameters
            try:
                self._extra_params = dict([
                    (p, data_dict[p][0,self._pft]) for p in params
                ])
            except IndexError:
                # Some older parameter datasets have inconsistent dimensions
                self._extra_params = dict([
                    (p, data_dict[p][self._pft]) for p in params
                ])

    @suppress_warnings
    def _load_state(self, hdf, keys):
        '''
        Overrides original _load_state() so as to accept an alternate source
        of SOC data, without having to duplicate all the other data in the
        drivers HDF5.
        '''
        soc_path = self.config['soc_data_path']
        with open(soc_path, 'rb') as file:
            _, soil_organic_carbon = pickle.load(file)
        shp = (1, self.config['time_steps'], self.config['site_count'], 81)
        init_state = []
        new_state = []
        for p, key in enumerate(keys):
            # Create an empty state array, allocated T time steps
            new_state.append(np.full(shp, np.nan))
            if key.startswith('soc'):
                arr = soil_organic_carbon[p,...]
                # Filter out any NoData, which (should) only correspond to
                #   1-km subgrid pixels that are outside the PFT range [1, 8]
                init_state.append(
                    np.where(arr < 0, np.nan, arr).reshape((1, 1, *shp[2:])))
            else:
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

        # LOAD CONSTANTS
        self._print('Loading constants...')
        self._load_constants()

        # INITIALIZE CONSTRAINT FUNCTIONS
        self._print('Creating linear constraint functions...')
        self._load_constraints(
            self._pft, filter(lambda x: x not in ('apar',), self.DRIVERS_INDEX))

    def run(self, gpp, steps = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        gpp : numpy.ndarray
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: `None`)
        '''
        @suppress_warnings
        def step(t, gpp, fields_rh):
            'Calculate fluxes, new states for the next time step t'
            if t == 0:
                # Retrieve initial SOC pool sizes
                state = self.state_initial[0:3,0,...]
            else:
                # Retrieve SOC in each pool from prior step
                state = self.state.data[0:3,t-1,...]
            # Calculate fluxes, new states
            rh, k_mult = self.rh(
                state, zip(hdf.index(t + self._t0, *fields_rh), fields_rh))
            npp = gpp * self.constants.CUE
            d_soc = self.soc(rh, t + self._t0)
            # Record fluxes for this time step
            self.fluxes.update('rh', t, rh.sum(axis = 0))
            self.fluxes.update('nee', t, rh.sum(axis = 0) - npp)
            # Record diagnostics at this time step
            if self._debug:
                # In debug mode, k_mult is not a single quantity but multiple;
                #   the order of k_mult_fields MUST match the order of the
                #   return signature for rh()
                k_mult_fields = ('f_tsoil', 'f_smsf')
                if all(f in self.DIAGNOSTICS_INDEX for f in k_mult_fields):
                    for k, key in enumerate(self.DIAGNOSTICS_INDEX):
                        if key in k_mult_fields:
                            idx = k_mult_fields.index(key)
                            self.state.update(key, t, k_mult[idx])

            # Treat Kmult and Emult differently (i.e., break out the former
            #   but not the latter) because that is what L4C Ops does
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

        # Fill in out-of-bounds SMSF with 100% wetness
        fill = {'tsoil': (np.nan, np.nan), 'smsf': (0, 100)}
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS, fill) as hdf:
            fields_rh = ['tsoil', 'smsf']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(t, gpp[t,...], fields_rh)
                    self._time_idx += 1
                    progress.update(t)


class L4CPrescribedGPPModelWithKokEffect(L4CPrescribedGPPModel):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data. Also includes modeling of
    the Kok effect; see `pyl4c.apps.l4c.extensions.phenology.L4CWithKokEffect`.

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
    DRIVERS_INDEX = ('tsoil', 'smsf', 'par')

    def run(self, gpp, steps = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        gpp : numpy.ndarray
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: `None`)
        '''
        @suppress_warnings
        def step(t, gpp, fields_rh):
            'Calculate fluxes, new states for the next time step t'
            if t == 0:
                # Retrieve initial SOC pool sizes
                state = self.state_initial[0:3,0,...]
            else:
                # Retrieve SOC in each pool from prior step
                state = self.state.data[0:3,t-1,...]
            # Calculate fluxes, new states
            rh, k_mult = self.rh(
                state, zip(hdf.index(t + self._t0, *fields_rh), fields_rh))
            # NOTE: Different in this model:
            #   CUE is constrained by PAR
            par = hdf.index(t + self._t0, 'par')
            cue = self.constrain(par[0], 'par') * self.constants.CUE
            npp = gpp * cue
            d_soc = self.soc(rh, t + self._t0)
            # Record fluxes for this time step
            self.fluxes.update('rh', t, rh.sum(axis = 0))
            self.fluxes.update('nee', t, rh.sum(axis = 0) - npp)
            # Record diagnostics at this time step
            if self._debug:
                # In debug mode, k_mult is not a single quantity but multiple;
                #   the order of k_mult_fields MUST match the order of the
                #   return signature for rh()
                k_mult_fields = ('f_tsoil', 'f_smsf')
                if all(f in self.DIAGNOSTICS_INDEX for f in k_mult_fields):
                    for k, key in enumerate(self.DIAGNOSTICS_INDEX):
                        if key in k_mult_fields:
                            idx = k_mult_fields.index(key)
                            self.state.update(key, t, k_mult[idx])

            # Treat Kmult and Emult differently (i.e., break out the former
            #   but not the latter) because that is what L4C Ops does
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

        # Fill in out-of-bounds SMSF with 100% wetness
        fill = {'tsoil': (np.nan, np.nan), 'smsf': (0, 100)}
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS, fill) as hdf:
            fields_rh = ['tsoil', 'smsf']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(t, gpp[t,...], fields_rh)
                    self._time_idx += 1
                    progress.update(t)


class L4CPrescribedGPPModelWithO2Limit(
        L4CPrescribedGPPModel, O2DiffusionLimitMixin):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data. Also includes modeling of
    an O2 diffusion limit on heterotrophic respiration; see
    `pyl4c.apps.l4c.extensions.phenology.L4CWithO2Diffusion`.

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
    DRIVERS_INDEX = ('tsoil', 'smsf')
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_smsf', 'conc_O2', 'mm_O2')
    AIR_FRAC_O2 = 0.2095 # Liters of O2 per liter of air (20.95%)

    def __init__(
            self, config, stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        with h5py.File(self.file_path, 'r') as hdf:
            # Read in porosity, copy to 1-km subgrid
            self._porosity = hdf['state/porosity'][:][:,np.newaxis]\
                .repeat(81, axis = 1)

    def run(self, gpp, steps = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        gpp : numpy.ndarray
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: `None`)
        '''
        @suppress_warnings
        def step(t, gpp, fields_rh):
            'Calculate fluxes, new states for the next time step t'
            if t == 0:
                # Retrieve initial SOC pool sizes
                state = self.state_initial[0:3,0,...]
            else:
                # Retrieve SOC in each pool from prior step
                state = self.state.data[0:3,t-1,...]
            # Calculate fluxes, new states
            rh, k_mult = self.rh(
                state, zip(hdf.index(t + self._t0, *fields_rh), fields_rh))
            npp = gpp * self.constants.CUE
            d_soc = self.soc(rh, t + self._t0)
            # Record fluxes for this time step
            self.fluxes.update('rh', t, rh.sum(axis = 0))
            self.fluxes.update('nee', t, rh.sum(axis = 0) - npp)
            # Record diagnostics at this time step
            if self._debug:
                # In debug mode, k_mult is not a single quantity but multiple;
                #   the order of k_mult_fields MUST match the order of the
                #   return signature for rh()
                k_mult_fields = ('f_tsoil', 'f_smsf', 'conc_O2', 'mm_O2')
                if all(f in self.DIAGNOSTICS_INDEX for f in k_mult_fields):
                    for k, key in enumerate(self.DIAGNOSTICS_INDEX):
                        if key in k_mult_fields:
                            idx = k_mult_fields.index(key)
                            self.state.update(key, t, k_mult[idx])

            # Treat Kmult and Emult differently (i.e., break out the former
            #   but not the latter) because that is what L4C Ops does
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

        # Fill in out-of-bounds SMSF with 100% wetness
        fill = {'tsoil': (np.nan, np.nan), 'smsf': (0, 100)}
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS, fill) as hdf:
            fields_rh = ['tsoil', 'smsf']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(t, gpp[t,...], fields_rh)
                    self._time_idx += 1
                    progress.update(t)


class L4CPrescribedGPPModelWithLitterfallPhenology(
        LitterfallMixin, L4CPrescribedGPPModel):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data. Also includes modeling of
    the CASA litterfall phenology scheme; see
    `pyl4c.apps.l4c.extensions.phenology.L4CWithLitterfallPhenology`.

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
    def __init__(
            self, config, stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        # It is necessary to have a way to look up the day of year for
        #   indexing the litterfall array, which is (365 x N x 81)
        with h5py.File(self.file_path, 'r') as hdf:
            dt = [datetime.date(*t) for t in hdf['time'][:].tolist()]
            self._doy = ordinals365(dt)
        # Set the daily litterfall
        with open(self.config['soc_data_path'], 'rb') as file:
            litterfall, _ = pickle.load(file)
        # Overwrite the daily litterfall that was set in the parent model
        self.constants.add(litterfall, 'litterfall')


class L4CPrescribedGPPModelWithSoilProfile(
        StratifiedRespirationMixin, StratifiedLitterfallMixin,
        L4CPrescribedGPPModel):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data. Also includes modeling of
    the vertical soil profile.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    sm_profiles : str
        File path to HDF5 file containing soil moisture profiles
    litterfall_schedule : bool
        True if litterfall is a function of day-of-year (DOY); False if
        litterfall is a pre-computed fraction of daily NPP (Default: False)
    stream : bool
        True to use L4CStreamingInputDataset instead of reading in all driver
        data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    CONSTANTS_INDEX = (
        'CUE', 'LUE', 'f_metabolic', 'f_structural', 'decay_rates',
        'k_depth_decay')
    # All DEPTHS must be positive
    DEPTHS = np.array((0.05, 0.15, 0.35, 0.75, 1.5, 3.0))\
        .reshape((6,1)) # meters
    # Get change in depth (layer thickness)
    DELTA_Z = (DEPTHS - np.vstack((0, DEPTHS[:-1])))\
        .reshape((DEPTHS.size, 1, 1))
    DIFFUSIVITY = 2e-4 # m2 yr-1 (Yi et al. 2020)

    def __init__(
            self, config, sm_profiles, litterfall_schedule = False,
            stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        self._multilayer = True # Multiple soil layers
        self._scheduled = litterfall_schedule
        # After Yi et al. (2020), diffusivity declines linearly with depth to
        #   3 meters; also convert to units of m2 day-1
        self.diffusivity = (self.DIFFUSIVITY / 365) *\
            linear_constraint(0, 3, form = 'reversed')(np.abs(self.DEPTHS))
        self.profiles_file_path = sm_profiles
        # Initialize the vertically discretized state
        self.state_by_depth = np.full((
            1 + len(self.DIAGNOSTICS_INDEX), config['time_steps'],
            self.DEPTHS.size, config['site_count'], 81
        ), np.nan, dtype = np.float32)

    @property
    def state_by_depth(self):
        return self._state_by_depth

    @state_by_depth.setter
    def state_by_depth(self, new):
        self._state_by_depth.data = new

    def run(self, gpp, steps = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        gpp : numpy.ndarray
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: `None`)
        '''
        @suppress_warnings
        def step(t, gpp, fields_rh, soil_t, soil_m, litterfall):
            'Calculate fluxes, new states for the next time step t'
            if t == 0:
                # Retrieve initial SOC pool sizes
                state = self.state_initial[0:3,0,...]
            else:
                # Retrieve SOC in each pool from prior step
                state = self.state.data[0:3,t-1,...]
            # Calculate fluxes, new states
            rh, k_mult = self.rh(state, (soil_t, soil_m))
            npp = gpp * self.constants.CUE
            # NOTE: Litterfall inputs may not scheduled
            d_soc = self.soc(
                state, rh, litterfall, t + self._t0,
                scheduled = self._scheduled)
            # Update RH in each soil layer, convert from g C m-3 to g C m-2
            rh = np.nansum(rh, axis = 0) * self.DELTA_Z
            self.state_by_depth.update('rh', t, rh)
            # Update NEE, taking sum of RH across all soil layers
            self.fluxes.update('nee', t, np.nansum(rh, axis = 0) - npp)
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

        # Load supplemental datasets
        with open(self.config['soc_data_path'], 'rb') as file:
            litterfall, _ = pickle.load(file)
        with h5py.File(self.file_path, 'r') as hdf:
            porosity = hdf['state/porosity'][:]
            # Calculate extent of soil layers, given bedrock depth
            bedrock = hdf['LAND_MODEL_CONSTANTS/depth_to_bedrock_m'][:]
            layer_mask = self.DEPTHS < bedrock
            # "surface_temp" is used for the surface layer (0-5 cm)
            soil_t = []
            soil_t.append(hdf['L4SM_DAILY_MEAN/surface_temp'][self._t0:,:])
            for i in range(1, self.DEPTHS.size):
                soil_t.append(
                    hdf['L4SM_DAILY_MEAN/soil_temp_layer%d' % i][self._t0:,:])
            soil_t = np.stack(soil_t)
            # Mask out measurements below bedrock depth
            soil_t.swapaxes(1, 2)[~layer_mask,...] = np.nan
        with h5py.File(self.profiles_file_path, 'r') as hdf:
            soil_m = 100 * np.divide(
                hdf['soil_moisture_vwc'][:,self._t0:,:], porosity)
            # Clip f(SM) response, as wetness values might be unrealistic
            #   given problems in ice-filled soil layers
            soil_m[soil_m > 100] = 100
            # Mask out measurements below bedrock depth
            soil_m.swapaxes(1, 2)[~layer_mask,...] = np.nan
        porosity = None
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS) as hdf:
            fields_rh = ['tsoil', 'smsf']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(t, gpp[t,...], fields_rh, soil_t[:,t,:,None],
                        soil_m[:,t,:,None], litterfall)
                    self._time_idx += 1
                    progress.update(t)


class L4CPrescribedGPPModelWithSoilProfileAndO2Limit(
        O2DiffusionLimitMixin, L4CPrescribedGPPModelWithSoilProfile):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data. Also includes modeling of
    the vertical soil profile AND an O2 diffusion limit.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    sm_profiles : str
        File path to HDF5 file containing soil moisture profiles
    litterfall_schedule : bool
        True if litterfall is a function of day-of-year (DOY); False if
        litterfall is a pre-computed fraction of daily NPP (Default: False)
    stream : bool
        True to use L4CStreamingInputDataset instead of reading in all driver
        data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_smsf', 'conc_O2', 'mm_O2')
    AIR_FRAC_O2 = 0.2095 # Liters of O2 per liter of air (20.95%)

    def __init__(
            self, config, sm_profiles, litterfall_schedule = False,
            stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, sm_profiles = sm_profiles,
            litterfall_schedule = litterfall_schedule, stream = stream,
            verbose = verbose, debug = debug)
        # Read in porosity, copy to 1-km subgrid
        with h5py.File(self.file_path, 'r') as hdf:
            self._porosity = hdf['state/porosity'][:][:,np.newaxis]\
                .repeat(81, axis = 1)


class L4CPrescribedGPPModelWithSoilProfileAndLitterfall(
        L4CPrescribedGPPModelWithSoilProfile):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data. Also includes modeling of
    the vertical soil profile AND a litterfall phenology

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    sm_profiles : str
        File path to HDF5 file containing soil moisture profiles
    litterfall_schedule : bool
        NOTE: Cannot be changed, defaults to True; see
        `pyl4c.apps.l4c.extensions.cosore.L4CPrescribedGPPModelWithSoilProfile`
    stream : bool
        True to use L4CStreamingInputDataset instead of reading in all driver
        data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    def __init__(
            self, config, sm_profiles, litterfall_schedule = True,
            stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, sm_profiles = sm_profiles,
            litterfall_schedule = True, stream = stream,
            verbose = verbose, debug = debug)
        # It is necessary to have a way to look up the day of year for
        #   indexing the litterfall array, which is (365 x N x 81)
        with h5py.File(self.file_path, 'r') as hdf:
            dt = [datetime.date(*t[:]) for t in hdf['time'][:].tolist()]
            self._doy = ordinals365(dt)
        # Set the daily litterfall
        with open(self.config['soc_data_path'], 'rb') as file:
            litterfall, _ = pickle.load(file)
        # Overwrite the daily litterfall that was set in the parent model
        self.constants.add(litterfall, 'litterfall')


class L4CPrescribedGPPModelWithO2LimitAndLitterfallPhenology(
        LitterfallMixin, L4CPrescribedGPPModelWithO2Limit):
    '''
    Variation on L4CForwardProcessPoint where GPP is not calculated but is
    prescribed by the user. This can be used for developing or testing SOC
    and respiration sub-models with fixed GPP data. Also includes modeling of
    the CASA litterfall phenology scheme; see
    `pyl4c.apps.l4c.extensions.phenology.L4CWithLitterfallPhenology`.

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
    def __init__(
            self, config, stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        # It is necessary to have a way to look up the day of year for
        #   indexing the litterfall array, which is (365 x N x 81)
        with h5py.File(self.file_path, 'r') as hdf:
            dt = [datetime.date(*t) for t in hdf['time'][:].tolist()]
            self._doy = ordinals365(dt)
        # Set the daily litterfall
        with open(self.config['soc_data_path'], 'rb') as file:
            litterfall, _ = pickle.load(file)
        # Overwrite the daily litterfall that was set in the parent model
        self.constants.add(litterfall, 'litterfall')
