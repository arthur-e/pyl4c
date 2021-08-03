'''
Variation on L4C model but with vertically stratified soil decomposition and
heterotrophic respiration sub-models based on soil hydrology profile.
'''

import datetime
import pickle
import numpy as np
import h5py
from functools import partial
from pyl4c import suppress_warnings
from pyl4c.science import ordinals365
from pyl4c.stats import linear_constraint
from pyl4c.lib.cli import ProgressBar
from pyl4c.apps.l4c import L4CConstants, L4CState, L4CStratifiedState
from pyl4c.apps.l4c.io import L4CStreamingInputDataset
from pyl4c.apps.l4c.main import L4CForwardProcessPoint

# File paths to the L4SM soil temperature data by layer; the SM profile data from simulation
TSOIL_HDF = '/home/arthur.endsley/DATA/L4_SM_gph_NRv8-3_profile_at_356_tower_sites.h5'
SM_HDF = '/home/arthur.endsley/DATA/L4_C_NRv8-3_soil_moisture_profiles_simulated_at_356_tower_sites.h5'

class O2DiffusionLimitMixin:
    '''
    Provides an O2 diffusion limit as a mix-in. Not to be instantiated
    directly.
    '''
    def concentration_O2(self, soil_vwc):
        'Concentration of O2 given soil vegetation water content (VWC)'
        d_gas = self._extra_params['d_gas']
        return d_gas * self.AIR_FRAC_O2 * np.power(
            self._porosity - soil_vwc, 4/3)

    def rh(self, state, drivers):
        '''
        Calculate RH for a single time step.

        Parameters
        ----------
        state : numpy.ndarray
            `(3 x N x M)` array of current SOC state in each pool
        drivers : list or tuple
            Sequence of `numpy.ndarray`: `(tsoil, smsf)` where `tsoil` is the
            the soil temperature and `smsf` is the surface soil moisture, each
            is an (N x M) array

        Returns
        -------
        numpy.ndarray
        '''
        # Translate Tsoil and SMSF into environmental constraints on RH
        tsoil, smsf = drivers
        # Some classes pass zipped (array, label) pairs for the drivers,
        #   others just pass arrays
        if hasattr(tsoil, 'ndim'):
            f_tsoil = self.constrain(tsoil, 'tsoil')
            f_smsf = self.constrain(smsf, 'smsf')
        else:
            f_tsoil = self.constrain(tsoil)
            f_smsf = self.constrain(*smsf)
            smsf, _ = smsf # Pop off the label "smsf" and just get the data
        # NOTE: Converting from "wetness" to volumetric water content (VWC)
        #   (in % units); this requires multiplying (wetness * porosity) as
        #   (wetness = VWC / porosity)
        soil_vwc = np.multiply(smsf / 100, self._porosity)
        conc_O2 = self.concentration_O2(soil_vwc)
        mm_O2 = conc_O2 / (self._extra_params['km_oxy'] + conc_O2)
        # Take the minimum of the SMSF and Soil VWC constraint
        k_mult = f_tsoil * np.min(np.stack((f_smsf, mm_O2)), axis = 0)
        k_mult = k_mult[np.newaxis,...].repeat(3, axis = 0)
        # NOTE: These are true decay rates for 2nd and 3rd pools, so it
        #   is straightfoward to multiply them against SOC
        if self._multilayer:
            rh = k_mult * self.constants.decay_rates[:,None,...] * state
        else:
            rh = k_mult * self.constants.decay_rates * state
        # "the adjustment...to account for material transferred into the
        #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
        rh[1,...] = rh[1,...] * (1 - self.constants.f_structural)
        # T_mult, W_mult same for each pool
        return (rh, (f_tsoil, f_smsf, conc_O2, mm_O2))


class StratifiedLitterfallMixin:
    '''
    Provides the CASA litterfall phenology for a vertical soil profile as a
    mix-in; not meant to be instantiated directly.
    '''
    def soc(self, state, rh, litterfall, t, scheduled = True):
        '''
        Calculate change in soil organic carbon (SOC) for a single time step.

        Parameters
        ----------
        state : numpy.ndarray
            `(3 x Z x N x M)` array of SOC at the previous time step
        rh : numpy.ndarray
            `(3 x Z x N x M)` array of RH at the current time step
        litterfall : numpy.ndarray
            `(365 x Z x N x M)` array of average daily litterfall throughout
            the 365-day climatological year
        t : int
            Current time step
        scheduled : bool
            True if litterfall is a function of day-of-year (DOY); False if
            litterfall is a pre-computed fraction of daily NPP (Default: True)

        Returns
        -------
        numpy.ndarray
        '''
        if scheduled:
            doy = self._doy[t] - 1 # Get DOY on [1,365] then on [0,364] for Python
            litter = litterfall[:,doy,...]
        else:
            litter = litterfall
        # Change in Cmet, Cstr, Crec with depth (z)
        shp = state.shape[-2:]
        dc0_dz = (state[0] - np.vstack((np.zeros((1, *shp)), state[0,:-1]))) / self.DELTA_Z
        dc1_dz = (state[1] - np.vstack((np.zeros((1, *shp)), state[1,:-1]))) / self.DELTA_Z
        dc2_dz = (state[1] - np.vstack((np.zeros((1, *shp)), state[2,:-1]))) / self.DELTA_Z
        # Change in diffusivity with depth (z)
        diff0 = self.diffusivity[:,None] * dc0_dz
        diff1 = self.diffusivity[:,None] * dc1_dz
        diff2 = self.diffusivity[:,None] * dc2_dz
        dd0_dz = (diff0 - np.vstack((np.zeros((1, *shp)), diff0[:-1]))) / self.DELTA_Z
        dd1_dz = (diff1 - np.vstack((np.zeros((1, *shp)), diff1[:-1]))) / self.DELTA_Z
        dd2_dz = (diff2 - np.vstack((np.zeros((1, *shp)), diff2[:-1]))) / self.DELTA_Z
        # Change in SOC according to diff. eq. in Jones et al. (2017)
        dc1 = (litter * self.constants.f_metabolic) - rh[0,...] + dd0_dz
        dc2 = (litter * (1 - self.constants.f_metabolic)) - rh[1,...] + dd1_dz
        dc3 = (self.constants.f_structural * rh[1,...]) - rh[2,...] + dd2_dz
        return (dc1, dc2, dc3)


class StratifiedRespirationMixin:
    '''
    Provides the vertically stratified heterotrophic respiration as a mix-in;
    not meant to be instantiated directly.
    '''
    @suppress_warnings
    def _load_state(self, hdf, keys):
        '''
        Overrides original _load_state() so as to allow for vertically
        stratified SOC data.
        '''
        t = self.config['time_steps']
        z = self.DEPTHS.size
        n = self.config['site_count']
        soc_path = self.config['soc_data_path']
        init_state = []
        new_state = []
        with open(soc_path, 'rb') as file:
            _, soil_organic_carbon = pickle.load(file)
        for p, key in enumerate(keys):
            # Create an empty state array, allocated T time steps
            new_state.append(np.full((1, t, z, n, 81), np.nan))
            if key.startswith('soc'):
                arr = soil_organic_carbon[p,...].astype(np.float32)
                # Filter out any NoData, which (should) only correspond to
                #   1-km subgrid pixels that are outside the PFT range [1, 8]
                init_state.append(
                    np.where(arr < 0, np.nan, arr).reshape((1, 1, z, n, 81)))
            elif key not in self.DIAGNOSTICS_INDEX:
                init_state.append(
                    np.full((1, 1, z, n, 81), np.nan, dtype = np.float32))
        return (init_state, new_state)

    def _setup_data_storage(self, config, hdf):
        '''
        Initialize containers for various datasets; including containers with
        vertical discretization.
        '''
        site_names = hdf['site_id'][:].tolist()
        shp = (config['time_steps'], config['site_count'], 81)
        self._constants = L4CConstants(
            config['site_count'], self._pft, self.PFT_CODES)
        self._drivers = None
        self._fluxes = L4CState(
            config['site_count'], np.full((len(self.FLUX_INDEX), *shp),
            np.nan), self.FLUX_INDEX,
            axis_labels = [None, None, site_names, None])
        self._state_by_depth = L4CStratifiedState(
            self.DEPTHS.size, config['site_count'],
            labels = ('rh', *self.DIAGNOSTICS_INDEX),
            axis_labels = [
                None, None, self.DEPTHS.ravel().tolist(), site_names, None
            ])
        self._state = L4CStratifiedState(
            self.DEPTHS.size, config['site_count'], axis_labels = [
                None, None, self.DEPTHS.ravel().tolist(), site_names, None
            ])

    def rh(self, state, drivers):
        '''
        Calculate heterotrophic respiration (RH) for a single time step.

        Parameters
        ----------
        state : numpy.ndarray
            `(3 x N x M)` array of current SOC state in each pool
        drivers : list or tuple
            Sequence of `numpy.ndarray`: `(tsoil, sm)` where `tsoil` is the
            the soil temperature and `sm` is the soil moisture, each is a
            (Z x N x M) array

        Returns
        -------
        numpy.ndarray
            Heterotrophic respiration (g C m-2 day-1)
        '''
        # Translate Tsoil and SMSF into environmental constraints on RH
        tsoil, sm = drivers
        f_tsoil = self.constrain(tsoil, 'tsoil')
        f_sm = self.constrain(sm, 'smsf')
        # Extinction rate of heterotrophic respiration with depth, due to
        #   factors OTHER THAN temperature, moisture (Koven et al. 2013)
        f_z = np.exp(
            -np.abs(self.DEPTHS[...,None]) / self.constants.k_depth_decay)
        k_mult = f_tsoil * f_sm
        k_mult = k_mult[np.newaxis,...].repeat(3, axis = 0)
        # NOTE: These are true decay rates for 2nd and 3rd pools, so it
        #   is straightfoward to multiply them against SOC
        rh = (k_mult * f_z) * (self.constants.decay_rates[:,None,...] * state)
        # "the adjustment...to account for material transferred into the
        #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
        rh[1,...] = rh[1,...] * (1 - self.constants.f_structural)
        # T_mult, W_mult same for each pool
        return (rh, (f_tsoil, f_sm))


class L4CStratifiedModel(
        StratifiedRespirationMixin, StratifiedLitterfallMixin,
        L4CForwardProcessPoint):
    '''
    Variation on SMAP L4C model, but vertically stratified SOC.

    Notes for developers:

    1. The litterfall constant is not set, as the `L4CConstant` class does
    not have support for vertical discretization. Instead, litterfall is
    read in at `L4CStratifiedModel.run()` and consumed by the private
    `step()` method of that function.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    litterfall_schedule : bool
        True if litterfall is a function of day-of-year (DOY); False if
        litterfall is a pre-computed fraction of daily NPP (Default: False)
    stream : bool
        True to use `L4CStreamingInputDataset` instead of reading in all
        driver data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    # All DEPTHS must be positive
    DEPTHS = np.array((0.05, 0.15, 0.35, 0.75, 1.5, 3.0))\
        .reshape((6,1)) # meters
    # Get change in depth (layer thickness)
    DELTA_Z = (DEPTHS - np.vstack((0, DEPTHS[:-1])))\
        .reshape((DEPTHS.size, 1, 1))
    DIFFUSIVITY = 2e-4 # m2 yr-1 (Yi et al. 2020)
    PFT_CODES = range(1, 9)
    CONSTANTS_INDEX = (
        'CUE', 'LUE', 'f_metabolic', 'f_structural', 'decay_rates',
        'k_depth_decay')
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_sm')
    FLUX_INDEX = ('nee',) # NOTE: Ignoring GPP; RH tracked separately
    STATE_INDEX = ('soc1', 'soc2', 'soc3') # NOTE: RH is vertically stratified

    def __init__(
            self, config, litterfall_schedule = False, stream = True,
            verbose = True, debug = False):
        for key in ('extra_parameters', 'soc_data_path'):
            assert key in config.keys(),\
                'L4CPhenologyProcess model requires "%s" configuration key' % key
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        self._multilayer = True # Multi-layer model
        self._scheduled = litterfall_schedule
        with h5py.File(self.file_path, 'r') as hdf:
            site_names = hdf['site_id'][:].tolist()
        if config['extra_parameters'] is not None:
            with open(config['extra_parameters'], 'rb') as file:
                data_dict = pickle.load(file)
            # Identify the extra parameters
            params = set(data_dict.keys()).difference(self._bplut.keys())
            # In case a parameter doesn't belong (i.e., they are all NaN),
            #   remove it
            for p in list(params):
                if np.isnan(data_dict[p]).all():
                    print('WARNING: Ignoring all-NaN parameter "%s"' % p)
                    params.remove(p)
            with h5py.File(self.file_path, 'r') as hdf:
                # Set invalid PFT codes to PFT 0
                self._pft = np.where(self._pft > max(self.PFT_CODES), 0, self._pft)
                # Vectorize the new parameters
                self._extra_params = dict([
                    (p, data_dict[p][self._pft]) for p in params
                ])
        # After Yi et al. (2020), diffusivity declines linearly with depth to
        #   3 meters; also convert to units of m2 day-1
        self.diffusivity = (self.DIFFUSIVITY / 365) *\
            linear_constraint(0, 3, form = 'reversed')(np.abs(self.DEPTHS))
        # Initialize the vertically discretized state
        self.state_by_depth = np.full((
            1 + len(self.DIAGNOSTICS_INDEX), config['time_steps'],
            self.DEPTHS.size, len(site_names), 81
        ), np.nan, dtype = np.float32)

    @property
    def state_by_depth(self):
        return self._state_by_depth

    @state_by_depth.setter
    def state_by_depth(self, new):
        self._state_by_depth.data = new

    def run(self, steps = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: `None`)
        '''
        @suppress_warnings
        def step(t, fields_gpp, soil_t, soil_m, litterfall):
            'Calculate fluxes, new states for the next time step t'
            if t == 0:
                # Retrieve intial SOC pool sizes
                state = self.state_initial[0:3,0,...]
            else:
                # Retrieve SOC in each pool from prior step
                state = self.state.data[0:3,t-1,...]
            # Calculate fluxes, new states
            gpp, _ = self.gpp(
                zip(hdf.index(t + self._t0, *fields_gpp), fields_gpp))
            rh, k_mult = self.rh(state, (soil_t, soil_m))
            f_tsoil, f_sm = k_mult
            npp = gpp * self.constants.CUE
            # NOTE: Litterfall inputs may not scheduled
            d_soc = self.soc(
                state, rh, litterfall, t + self._t0,
                scheduled = self._scheduled)
            # Update RH in each soil layer, convert from g C m-3 to g C m-2
            rh = np.nansum(rh, axis = 0) * self.DELTA_Z
            self.state_by_depth.update('rh', t, rh)
            self.state_by_depth.update('f_tsoil', t, f_tsoil)
            self.state_by_depth.update('f_sm', t, f_sm)
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
        with h5py.File(TSOIL_HDF, 'r') as hdf:
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
        with h5py.File(SM_HDF, 'r') as hdf:
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
            fields_gpp = ['apar', 'tmin', 'vpd', 'ft', 'smrz']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(
                        t, fields_gpp, soil_t[:,t,:,None], soil_m[:,t,:,None],
                        litterfall)
                    self._time_idx += 1
                    progress.update(t)


class L4CStratifiedDiffusionModel(
        O2DiffusionLimitMixin, L4CStratifiedModel):
    '''
    Variation on SMAP L4C model, but with:

    1. Vertically stratified soil organic carbon (SOC); and
    2. Oxygen diffusion limitation for heterotrophic respiration (RH).

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    litterfall_schedule : bool
        True if litterfall is a function of day-of-year (DOY); False if
        litterfall is a pre-computed fraction of daily NPP (Default: False)
    stream : bool
        True to use `L4CStreamingInputDataset` instead of reading in all
        driver data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    CONSTANTS_INDEX = (
        'CUE', 'LUE', 'f_metabolic', 'f_structural', 'decay_rates',
        'k_depth_decay', 'd_gas', 'km_oxy')
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_sm', 'mm_O2')
    FLUX_INDEX = ('nee',) # NOTE: Ignoring GPP; RH tracked separately
    STATE_INDEX = ('soc1', 'soc2', 'soc3') # NOTE: RH is vertically stratified

    def run(self, steps = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: `None`)
        '''
        @suppress_warnings
        def step(t, fields_gpp, soil_t, soil_m, soil_vwc, litterfall):
            'Calculate fluxes, new states for the next time step t'
            if t == 0:
                # Retrieve intial SOC pool sizes
                state = self.state_initial[0:3,0,...]
            else:
                # Retrieve SOC in each pool from prior step
                state = self.state.data[0:3,t-1,...]
            # Calculate fluxes, new states
            gpp, _ = self.gpp(
                zip(hdf.index(t + self._t0, *fields_gpp), fields_gpp))
            rh, k_mult = self.rh(state, (soil_t, soil_m, soil_vwc))
            f_tsoil, f_sm, mm_O2 = k_mult
            npp = gpp * self.constants.CUE
            # NOTE: Litterfall inputs may not scheduled
            d_soc = self.soc(
                state, rh, litterfall, t + self._t0,
                scheduled = self._scheduled)
            # Update RH in each soil layer, convert from g C m-3 to g C m-2
            rh = np.nansum(rh, axis = 0) * self.DELTA_Z
            self.state_by_depth.update('rh', t, rh)
            self.state_by_depth.update('f_tsoil', t, f_tsoil)
            self.state_by_depth.update('f_sm', t, f_sm)
            self.state_by_depth.update('mm_O2', t, mm_O2)
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
            self.porosity = hdf['state/porosity'][:]
        with h5py.File(TSOIL_HDF, 'r') as hdf:
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
        with h5py.File(SM_HDF, 'r') as hdf:
            soil_vwc = hdf['soil_moisture_vwc'][:,self._t0:,:]
            soil_m = 100 * np.divide(soil_vwc, self.porosity)
            # Clip f(SM) response, as wetness values might be unrealistic
            #   given problems in ice-filled soil layers
            soil_m[soil_m > 100] = 100
            # Mask out measurements below bedrock depth
            soil_m.swapaxes(1, 2)[~layer_mask,...] = np.nan
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS) as hdf:
            fields_gpp = ['apar', 'tmin', 'vpd', 'ft', 'smrz']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(
                        t, fields_gpp, soil_t[:,t,:,None], soil_m[:,t,:,None],
                        soil_vwc[:,t,:,None], litterfall)
                    self._time_idx += 1
                    progress.update(t)


class L4CStratifiedLitterfallModel(L4CStratifiedModel):
    '''
    Variation on SMAP L4C model, but with:

    1. Vertically stratified soil organic carbon (SOC); and
    2. Litterfall phenology, wherein average daily litterfall varies overy a
        365-day climatological year, according to the CASA scheme.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    litterfall_schedule : bool
        NOTE: Cannot be changed, defaults to True; see
        `pyl4c.apps.l4c.extensions.hydrology.L4CStratifiedModel`
    stream : bool
        True to use `L4CStreamingInputDataset` instead of reading in all
        driver data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    CONSTANTS_INDEX = (
        'CUE', 'LUE', 'f_metabolic', 'f_structural', 'decay_rates',
        'k_depth_decay')
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_sm')
    FLUX_INDEX = ('nee',) # NOTE: Ignoring GPP; RH tracked separately
    STATE_INDEX = ('soc1', 'soc2', 'soc3') # NOTE: RH is vertically stratified

    def __init__(
            self, config, litterfall_schedule = True, stream = True,
            verbose = True, debug = False):
        for key in ('soc_data_path',):
            assert key in config.keys(),\
                'L4CPhenologyProcess model requires "%s" configuration key' % key
        if 'extra_parameters' in config.keys():
            assert config['extra_parameters'] is None,\
                'L4CStratifiedLitterfallModel does not expect "extra_parameters" in configuration'
        super().__init__(
            config = config, litterfall_schedule = True,
            stream = stream, verbose = verbose, debug = debug)
        # It is necessary to have a way to look up the day of year for
        #   indexing the litterfall array, which is (365 x N x 81)
        with h5py.File(self.file_path, 'r') as hdf:
            dt = [datetime.date(*t[:-1]) for t in hdf['time'][:].tolist()]
            self._doy = ordinals365(dt)
        # Set the daily litterfall
        with open(self.config['soc_data_path'], 'rb') as file:
            litterfall, _ = pickle.load(file)
        # Overwrite the daily litterfall that was set in the parent model
        self.constants.add(litterfall, 'litterfall')


class L4CStratifiedDiffusionLitterfallModel(L4CStratifiedDiffusionModel):
    '''
    Variation on SMAP L4C model, but with:

    1. Vertically stratified soil organic carbon (SOC);
    2. Oxygen diffusion limitation for heterotrophic respiration (RH); and
    3. Litterfall phenology, wherein average daily litterfall varies overy a
        365-day climatological year, according to the CASA scheme.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    litterfall_schedule : bool
        NOTE: Cannot be changed, defaults to True; see
        `pyl4c.apps.l4c.extensions.hydrology.L4CStratifiedModel`
    stream : bool
        True to use `L4CStreamingInputDataset` instead of reading in all
        driver data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    def __init__(
            self, config, litterfall_schedule = True, stream = True,
            verbose = True, debug = False):
        for key in ('extra_parameters', 'soc_data_path'):
            assert key in config.keys(),\
                'L4CPhenologyProcess model requires "%s" configuration key' % key
        super().__init__(
            config = config, litterfall_schedule = True, stream = stream,
            verbose = verbose, debug = debug)
        # It is necessary to have a way to look up the day of year for
        #   indexing the litterfall array, which is (365 x N x 81)
        with h5py.File(self.file_path, 'r') as hdf:
            dt = [datetime.date(*t[:-1]) for t in hdf['time'][:].tolist()]
            self._doy = ordinals365(dt)
        # Set the daily litterfall
        with open(self.config['soc_data_path'], 'rb') as file:
            litterfall, _ = pickle.load(file)
        # Overwrite the daily litterfall that was set in the parent model
        self.constants.add(litterfall, 'litterfall')
