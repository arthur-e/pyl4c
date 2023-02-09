'''
Couplers for L4C and other models.
'''

import datetime
import pickle
import numpy as np
import h5py
from pyl4c import suppress_warnings
from pyl4c.science import ordinals365
from pyl4c.lib.cli import ProgressBar
from pyl4c.apps.l4c.io import L4CStreamingInputDataset
from pyl4c.apps.l4c.main import L4CForwardProcessPoint
from pyl4c.apps.l4c.extensions.damm import DAMMDecompositionModel2, g_cm3_to_g_m2
from pyl4c.apps.l4c.extensions.phenology import L4CPhenologyProcess

class L4CEnhancedDecomposition(L4CPhenologyProcess):
    '''
    Variation on SMAP L4C model, with two improvements to the SOC deomposition
    model:

    1. The RH sub-model is modified to include an O2 diffusion limitation.
        Driver data are the same as for any L4C Science run, so additional
        configuration data must be provided: "soc_data_path" and
        "extra_parameters".
    2. The SOC sub-model is modified to include a litterfall phenology.
        Driver data are the same as for any L4C Science run, except that
        litterfall is different.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    stream : bool
        True to use `L4CStreamingInputDataset` instead of reading in all
        driver data; this reduces memory use but increases I/O
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
        # For the litterfall phenology: It is necessary to have a way to look
        #   up the day of year for indexing the litterfall array, which is
        #   (365 x N x 81)
        with h5py.File(self.file_path, 'r') as hdf:
            dt = [datetime.date(*t[:-1]) for t in hdf['time'][:].tolist()]
            self._doy = ordinals365(dt)
        # For the O2 diffusion limit: Read in porosity, copy to 1-km subgrid
        with h5py.File(self.file_path, 'r') as hdf:
            self._porosity = hdf['state/porosity'][:][:,np.newaxis]\
                .repeat(81, axis = 1)

    def concentration_O2(self, soil_vwc):
        '''
        Concentration of O2 given soil vegetation water content (VWC).

        Parameters
        ----------
        soil_vwc : float or numpy.ndarray
            The soil moisture volumetric water content (m3 m-3)

        Returns
        -------
        float or numpy.ndarray
            The O2 concentration in soil pore spaces (dimensionless, L L-1)
        '''
        d_gas = self._extra_params['d_gas']
        return d_gas * self.AIR_FRAC_O2 * np.power(
            self._porosity - soil_vwc, 4/3)

    def rh(self, state, drivers):
        '''
        Calculate heterotrophic respiration (RH) for a single time step.

        Parameters
        ----------
        state : numpy.ndarray
            `(3 x N x M)` array of current SOC state in each pool
        drivers : list or tuple
            Nested sequence of (driver: `numpy.ndarray`, label: `str`) pairs

        Returns
        -------
        numpy.ndarray
            Heterotrophic respiration (g C m-2 day-1)
        '''
        # Translate Tsoil and SMSF into environmental constraints on RH
        tsoil, smsf = drivers
        f_tsoil = self.constrain(*tsoil)
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
            d_soc = self.soc(rh, t + self._t0)
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

    def soc(self, rh, t):
        '''
        Calculate change in soil organic carbon (SOC) for a single time step.

        Parameters
        ----------
        rh : numpy.ndarray
            `(3 x N x M)` array of RH at the current time step
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


class L4CDAMM(L4CForwardProcessPoint):
    '''
    L4C-DAMM model, coupling the L4C GPP and SOC models to the DAMM RH model.
    L4C computes daily GPP and litterfall, then DAMM computes RH, which L4C
    then uses to compute change in SOC.

    In debug mode, diagnostics includes the substrate and O2 concentrations as
    well as the half-saturation (MM) constants for substrate and O2. The MM
    constant for substrate reported is the minimum across the three substrate
    pools.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    rh_model : AbstractDAMM
        The class of DAMM model to use for RH calculations
    use_l4c_cue : bool
        True to use the CUE parameters from L4C instead of those fit by
        L4C-DAMM (Default: True)
    use_legacy_pft : bool
        True to use the L4C Nature Run v7.2 "legacy" PFT map ("lc_dom")
        for the PFT class assignments of each pixel (Default: True)
    stream : bool
        True to use `L4CStreamingInputDataset` instead of reading in all driver
        data; this reduces memory use but increases I/O
    verbose : bool
        True to print all output to stdout
    debug : bool
        True to store additional diagnostic information from each time step
    '''
    PFT_CODES = range(1, 9)
    SOIL_DEPTH_CM = 5
    DIAGNOSTICS_INDEX = ('conc_Sx', 'conc_O2', 'v_max', 'mm_O2', 'mm_Sx')
    STATE_INDEX = ('soc1', 'soc2', 'soc3')
    V_MAX_SCALE = 1e12 # Factor by which to scale Vmax before exporting

    def __init__(
            self, config, rh_model = DAMMDecompositionModel2,
            use_l4c_cue = True, use_legacy_pft = True, stream = True,
            verbose = True, debug = False):
        super().__init__(config, stream, verbose, debug)
        assert 'rh_parameters' in config.keys(),\
            'Coupled L4C-DAMM model requires "rh_parameters" configuration key'
        if debug and self.V_MAX_SCALE != 1:
            print('WARNING: V_MAX_SCALE is non-unity: V_max should not be interpreted physically')
        self.DAMM = rh_model
        with open(config['rh_parameters'], 'rb') as file:
            data_dict = pickle.load(file)
            # Add NaNs to position 0 so that PFT numeric id == index value
            params = np.vstack((
                [np.nan] * data_dict['parameters'].shape[1],
                data_dict['parameters']))

        with L4CStreamingInputDataset(
                self.file_path, self.CONSTANTS, self.BOUNDS) as hdf:
            if use_legacy_pft:
                self._pft = hdf['legacy/lc_dom'][:].swapaxes(0, 1)
            else:
                self._pft = hdf['state/PFT'][:]
            # Set invalid PFT codes to PFT 0
            self._pft = np.where(self._pft > max(self.PFT_CODES), 0, self._pft)
            # Vectorize the DAMM parameters
            self._rh_params = [
                params[self._pft,i] for i in range(0, params.shape[1])
            ]
            # Read in porosity, copy to 1-km subgrid
            self._porosity = hdf['state/porosity'][:][:,np.newaxis]\
                .repeat(81, axis = 1)
            # Option to use CUE parameters fit in L4C-DAMM...
            if not use_l4c_cue:
                # Convert annual NPP sum to annual GPP sum (using L4C CUE)
                gpp_sum = hdf['state/npp_sum'][:] / self.constants.CUE
                # Then, change CUE constant
                p = data_dict['metadata'].index('CUE')
                cue = np.hstack((np.nan, data_dict['parameters'][:,p]))
                self.constants.CUE = cue[self._pft]
                npp_sum = gpp_sum * self.constants.CUE
                # Calculate daily litterfall based on the annual NPP sum
                self.constants.add(
                    npp_sum / self.DAYS_PER_YEAR, 'litterfall')

    def rh(self, state, drivers):
        '''
        Calculate heterotrophic respiration (RH) for a single time step.

        Parameters
        ----------
        state : numpy.ndarray
            `(3 x N x M)` array of current SOC state in each pool
        drivers : list or tuple
            Nested sequence of (driver: `numpy.ndarray`, label: `str`) pairs

        Returns
        -------
        numpy.ndarray
            Heterotrophic respiration (g C m-2 day-1)
        '''
        tsoil, smsf = drivers
        # NOTE: Converting from "wetness" to volumetric water content (VWC)
        #   (in % units); this requires multiplying (wetness * porosity) as
        #   (wetness = VWC / porosity)
        vwc = 100 * np.multiply(smsf / 100, self._porosity)
        damm = self.DAMM(self._rh_params, self.SOIL_DEPTH_CM)
        # Convert SOC from g m-2 to g cm-3
        substrate = [
            np.array(s)
            for s in g_cm3_to_g_m2(state, self.SOIL_DEPTH_CM).tolist()
        ]
        rh = np.stack(
            damm.respiration(substrate, vwc, tsoil, self._porosity), axis = 0)
        if self._debug:
            conc_Sx, conc_O2 = damm.concentrations(
                substrate, vwc, self._porosity)
            v_max = np.stack([ # alpha0, alpha1, and alpha2 are first 3 params
                damm.v_max(tsoil, a) for a in self._rh_params[0:3]
            ], axis = 0)
        else:
            conc_Sx, conc_O2, v_max = (None, None, None)
        # "the adjustment...to account for material transferred into the
        #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
        rh[1,...] = rh[1,...] * (1 - self.constants.f_structural)
        return (rh, conc_Sx, conc_O2, v_max)

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
            rh, conc_Sx, conc_O2, v_max = self.rh(
                state, hdf.index(t + self._t0, *fields_rh))
            npp = gpp * self.constants.CUE
            d_soc = self.soc(rh)
            # Record fluxes for this time step
            self.fluxes.update('gpp', t, gpp)
            self.fluxes.update('rh', t, rh.sum(axis = 0))
            self.fluxes.update('nee', t, rh.sum(axis = 0) - npp)
            if self._debug:
                self.state.update('conc_Sx', t, conc_Sx.sum(axis = 0))
                self.state.update('conc_O2', t, conc_O2)
                km_s = km_s_int + (km_s_slope * hdf.index(t + self._t0, 'tsoil')[0])
                self.state.update('mm_Sx', t, (conc_Sx / (km_s + conc_Sx)).min(axis = 0))
                self.state.update('mm_O2', t, conc_O2 / (km_O2 + conc_O2))
                self.state.update('v_max', t,
                    np.multiply(self.V_MAX_SCALE, v_max).mean(axis = 0))
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

        if self._debug:
            # Read in the following parameter values in debug mode
            km_O2 = self._rh_params[self.DAMM.parameter_names.index('km_O2')]
            km_s_int = self._rh_params[self.DAMM.parameter_names.index('km_s_int')]
            km_s_slope = self._rh_params[self.DAMM.parameter_names.index('km_s_slope')]
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
