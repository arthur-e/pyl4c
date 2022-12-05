'''
Couple of the operational L4C model and various phenology-related
extensions; see corresponding calibration code in the module:
`pyl4c.apps.calibration.extensions.phenology`.

In sub-classes of `L4CPhenologyProcess` (and of `L4CForwardProcessPoint` more
generally), typically all that is required to change the start-up and
initialization of a forward run is to modify one of the class-level variables
in all caps (e.g., `DRIVERS_INDEX`, which registers constraint functions
during setup).
'''

import datetime
import pickle
import h5py
import numpy as np
from pyl4c import suppress_warnings
from pyl4c.science import ordinals365
from pyl4c.lib.cli import ProgressBar
from pyl4c.apps.l4c.io import L4CStreamingInputDataset
from pyl4c.apps.l4c.main import L4CForwardProcessPoint
from pyl4c.apps.calibration import BPLUT

class L4CPhenologyProcess(L4CForwardProcessPoint):
    '''
    Variation on SMAP L4C model, where the RH sub-model is modified to allow
    for respiration phenology. Driver data are the same as for any L4C Science
    run, so additional configuration data must be provided:

    1. To update the SOC and litterfall parameters, an `soc_data_path`
        configuration option must be supplied.
    2. New model parameters must be provided in the `extra_parameters`
        configuration option.

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
    PFT_CODES = range(1, 9)
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_smsf')
    STATE_INDEX = ('soc1', 'soc2', 'soc3')

    def __init__(
            self, config, stream = True, verbose = True, debug = False):
        for key in ('extra_parameters', 'soc_data_path'):
            assert key in config.keys(),\
                'L4CPhenologyProcess model requires "%s" configuration key' % key
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        if config['extra_parameters'] is not None:
            with open(config['extra_parameters'], 'rb') as file:
                data_dict = pickle.load(file)
            # Identify the extra parameters
            params = set(data_dict.keys()).difference(self._bplut.keys())
            # Set invalid PFT codes to PFT 0
            self._pft = np.where(self._pft > max(self.PFT_CODES), 0, self._pft)
            # Vectorize the new parameters
            self._extra_params = dict([
                (p, data_dict[p][self._pft]) for p in params
            ])
        # Set the daily litterfall
        with open(self.config['soc_data_path'], 'rb') as file:
            litterfall, _ = pickle.load(file)
        # Overwrite the daily litterfall that was set in the parent model
        self.constants.add(litterfall, 'litterfall')

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


class L4CWithO2Diffusion(L4CPhenologyProcess):
    '''
    Variation on SMAP L4C model, where the RH sub-model is modified to
    include an O2 diffusion limitation. Driver data are the same as for any
    L4C Science run, so additional configuration data must be provided:

    1. To update the SOC and litterfall parameters, an `soc_data_path`
        configuration option must be supplied.
    2. New model parameters must be provided in the `extra_parameters`
        configuration option.

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
    PFT_CODES = range(1, 9)
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_smsf', 'conc_O2', 'mm_O2')
    STATE_INDEX = ('soc1', 'soc2', 'soc3')
    AIR_FRAC_O2 = 0.2095 # Liters of O2 per liter of air (20.95%)

    def __init__(
            self, config, stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        with h5py.File(self.file_path, 'r') as hdf:
            # Read in porosity, copy to 1-km subgrid
            self._porosity = hdf['state/porosity'][:][:,np.newaxis]\
                .repeat(81, axis = 1)

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
        f_tsoil = self.constrain(*tsoil)
        f_smsf = self.constrain(*smsf)
        smsf, _ = smsf # Pop off the label "smsf" and just get the data
        # NOTE: Converting from "wetness" to volumetric water content (VWC)
        soil_vwc = np.multiply(smsf / 100, self._porosity)
        conc_O2 = self.concentration_O2(soil_vwc)
        mm_O2 = conc_O2 / (self._extra_params['km_oxy'] + conc_O2)
        # Take the minimum of the SMSF and Soil VWC constraint
        k_mult = f_tsoil * np.min(np.stack((f_smsf, mm_O2)), axis = 0)
        k_mult = k_mult[np.newaxis,...].repeat(3, axis = 0)
        # NOTE: These are true decay rates for 2nd and 3rd pools, so it
        #   is straightforward to multiply them against SOC
        rh = k_mult * self.constants.decay_rates * state
        # "the adjustment...to account for material transferred into the
        #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
        rh[1,...] = rh[1,...] * (1 - self.constants.f_structural)
        # T_mult, W_mult same for each pool
        return (rh, (f_tsoil, f_smsf, conc_O2, mm_O2))

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
            rh, rh_diag = self.rh(
                state, zip(hdf.index(t + self._t0, *fields_rh), fields_rh))
            npp = gpp * self.constants.CUE
            d_soc = self.soc(rh)
            # Record fluxes for this time step
            self.fluxes.update('gpp', t, gpp)
            self.fluxes.update('rh', t, rh.sum(axis = 0))
            self.fluxes.update('nee', t, rh.sum(axis = 0) - npp)
            if self._debug:
                f_tsoil, f_smsf, conc_O2, mm_O2 = rh_diag
                self.state.update('f_tsoil', t, f_tsoil)
                self.state.update('f_smsf', t, f_smsf)
                self.state.update('conc_O2', t, conc_O2)
                self.state.update('mm_O2', t, mm_O2)
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
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS) as hdf:
            fields_gpp = ['apar', 'tmin', 'vpd', 'ft', 'smrz']
            fields_rh = ['tsoil', 'smsf']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(t, fields_gpp, fields_rh)
                    self._time_idx += 1
                    progress.update(t)


class L4CWithKokEffect(L4CPhenologyProcess):
    '''
    Variation on SMAP L4C model, where the NEE model is modified to include
    the Kok effect (CUE is dependent on PAR). Driver data are the same as for
    any L4C Science run, so additional configuration data must be provided:

    1. To update the SOC and litterfall parameters, an `soc_data_path`
        configuration option must be supplied.
    2. New model parameters must be provided in the `extra_parameters`
        configuration option.

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
    PFT_CODES = range(1, 9)
    DIAGNOSTICS_INDEX = ('f_tsoil', 'f_smsf', 'CUE')
    DRIVERS_INDEX = ('apar', 'tmin', 'vpd', 'ft', 'smrz', 'tsoil', 'smsf', 'par')
    STATE_INDEX = ('soc1', 'soc2', 'soc3')

    def run(self, steps = None):
        '''
        A forward run in serial over multiple time steps; currently works in
        streaming mode ONLY.

        Parameters
        ----------
        steps : int
            Number of time steps to run or None to run through the end of the
            available time steps (exhaust driver data) (Default: None)
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
            rh, rh_diag = self.rh(
                state, zip(hdf.index(t + self._t0, *fields_rh), fields_rh))
            # NOTE: Different in this model:
            #   CUE is constrained by PAR
            par = hdf.index(t + self._t0, 'par')
            cue = self.constrain(par[0], 'par') * self.constants.CUE
            npp = gpp * cue
            d_soc = self.soc(rh)
            # Record fluxes for this time step
            self.fluxes.update('gpp', t, gpp)
            self.fluxes.update('rh', t, rh.sum(axis = 0))
            self.fluxes.update('nee', t, rh.sum(axis = 0) - npp)
            if self._debug:
                f_tsoil, f_smsf = rh_diag
                self.state.update('f_tsoil', t, f_tsoil)
                self.state.update('f_smsf', t, f_smsf)
                self.state.update('CUE', t, cue)
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
        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS) as hdf:
            fields_gpp = ['apar', 'tmin', 'vpd', 'ft', 'smrz']
            fields_rh = ['tsoil', 'smsf']
            num_steps = self.config['time_steps'] if steps is None else steps
            with ProgressBar(num_steps, 'Running...') as progress:
                for t in range(self._time_idx + 1, num_steps):
                    step(t, fields_gpp, fields_rh)
                    self._time_idx += 1
                    progress.update(t)


class L4CWithLitterfallPhenology(L4CPhenologyProcess):
    '''
    Variation on SMAP L4C model, where the SOC model is modified to include
    a litterfall phenology. Driver data are the same as for any L4C Science
    run, except that litterfall is different.
    '''
    def __init__(
            self, config, stream = True, verbose = True, debug = False):
        super().__init__(
            config = config, stream = stream, verbose = verbose, debug = debug)
        # It is necessary to have a way to look up the day of year for
        #   indexing the litterfall array, which is (365 x N x 81)
        with h5py.File(self.file_path, 'r') as hdf:
            dt = [datetime.date(*t[:-1]) for t in hdf['time'][:].tolist()]
            self._doy = ordinals365(dt)

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
