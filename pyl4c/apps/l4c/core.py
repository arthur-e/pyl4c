'''
'''

import numpy as np
import warnings
from numbers import Number
from typing import Sequence
from tqdm import tqdm
from pyl4c import Namespace
from pyl4c.science import arrhenius, linear_constraint, climatology365, ordinals365


class TCF(object):
    '''
    The Terrestrial Carbon Flux (TCF) model, which is the basis for the NASA
    Soil Moisture Active Passive (SMAP) Level 4 Carbon (L4C) model. TCF
    incorporates these basic assumptions:

    1. Carbon assimilation (gross primary production) is linearly related to
        the amount of solar radiation and green photosynthetic land cover.
    2. There is a maximum rate of carbon assimilation. Low water availability,
        cold or freezing temperatures, and high atmospheric demand for water
        vapor each reduce the efficiency of carbon assimilation.
    3. Soil organic carbon (SOC) decomposes at different rates based on the
        type of material. The optimal rate of decomposition is fixed and
        depends only on the land-cover type.
    4. The efficiency of soil organic carbon decomposition is reduced under
        low soil temperatures or low soil moisture.

    A complete description is available in Kimball et al. (2009) and in Jones
    et al. (2017). Soil organic carbon (SOC) and soil decomposition are
    discussed in Endsley et al. (2020) and Endsley et al. (2022).

    If initial model "state" is provided it should be a 2D array with the
    first axis containing, in order:

    - Initial state of "metabolic" SOC pool
    - Initial state of "structural" SOC pool
    - Initial state of "recalcitrant" SOC pool

    NOTE: For developers, `self.params` refers to model parameters that have
    been vectorized to improve performance. If a function performs
    vectorized calculations on longitudinal (N x T) arrays, the vectorized
    parameter(s) can be used with their current shape. For cross-sectional
    calculations, e.g., on (N,) or (1 x N) arrays, make sure to use the
    transpose, `self.params.name.T`, where `name` is the parameter name.

    Example use for a single pixel:

        tcf = TCF(params, land_cover_map = [7])
        tcf.nee(state = [500, 500, 1000], drivers = [...])


    Parameters
    ----------
    params : dict
        Dictionary of model parameters; keys should be among
        `TCF.required_parameters` and values should be sequences ordered by
        PFT code, either 9 values (value at index 0 being `np.nan`) or as many
        values as there are keys in `TCF.valid_pft`
    land_cover_map : Sequence or numpy.ndarray
        1-dimensional sequence of one or more land-cover types, one for each
        model resolution cell (pixel). These types should be represented as
        integers (unsigned 16-bit type or lower).
    state : Sequence or numpy.ndarray or None
        A sequence of 3 values or a 2D (S x N) array representing the state
        variables for each of N pixels. The first three (3) state variables
        should be the initial states of each SOC pool.
    litterfall : Sequence or numpy.ndarray or None
        A sequence of values or 1D array representing average annual
        litterfall (units: g C m-2 year-1) for each model resolution cell
        (pixel). If not provided, it will be based on the climatological NPP
        (when it is calculated).
    '''
    required_drivers = [
        'fPAR', 'PAR', 'Tmin', 'VPD', 'SMRZ', 'FT', 'Tsoil', 'SMSF'
    ]
    required_parameters = [
        'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0',
        'CUE', 'tsoil', 'smsf0', 'smsf1', 'decay_rates', 'f_structural',
        'f_metabolic'
    ]
    valid_pft = {
        1: 'Evergreen Needleleaf',
        2: 'Evergreen Broadleaf',
        3: 'Deciduous Needleleaf',
        4: 'Deciduous Broadleaf',
        5: 'Shrubland',
        6: 'Grassland',
        7: 'Cereal Croplands',
        8: 'Broadleaf Croplands'
    }

    num_soc_pools = 3 # Number of soil organic carbon (SOC) pools
    version = '7.4.1' # Pegged to the SMAP L4C version

    def __init__(
            self, params: dict, land_cover_map: Sequence,
            state: Sequence = None, litterfall: Sequence = None
        ):
        self.constants = Namespace()
        self.history = Namespace()
        self.state = Namespace()
        self.params = Namespace() # Parameters accessed, e.g., tcf.params.LUE
        self.lc_map = np.array(land_cover_map, dtype = np.uint16)

        # Load available annual litterfall (per resolution cell/ pixel)
        if litterfall is not None:
            litterfall = np.array(litterfall, dtype = np.float32)
        self.constants.add('litterfall', litterfall)

        # Load soil organic carbon (SOC) state
        if state is not None:
            if hasattr(state, 'ndim'):
                assert state.ndim <= 2, '"state" should have at most 2 dimensions'
            # "state" either begins as or is converted to a numpy.ndarray
            state = np.array(state, dtype = np.float32)
            if state.ndim == 1:
                state = state[:,np.newaxis]
            assert len(state) == 3, 'Expected one "state" value for each SOC pool'
            self.state.add('soc', np.array(state))
        # Create a parameters vector; e.g., for a land-cover map:
        #   array([  1,   2,   1,   3])
        # We create an array of parameters:
        #   array([1.5, 3.0, 1.5, 1.0])
        # Where the unique parameter for each land-cover class is copied as
        #   many times as that class appears; result is an array that is the
        #   same shape and size as the land-cover array
        # Possible items of the parameters dictionary include, e.g.:
        # {"key": 3.14}
        # {"key": [3.14, 1.10]}
        # {"key": np.array([3.14, 1.10])}
        # {"key": np.array([[3.14, 1.10], [...], ...])}
        for key, value in params.items():
            if key not in self.required_parameters:
                continue
            p_vector = np.array(value)
            # Copy parameter values based on PFT map
            if key == 'decay_rates':
                if p_vector.shape == (self.num_soc_pools,):
                    p_vector = p_vector[:,np.newaxis]\
                        .repeat(self.lc_map.size, axis = -1)
                elif hasattr(value, 'count') and p_vector.ndim == 2:
                    # i.e., "value" was nested lists and result of converting
                    #   to a NumPy array was a (N x self.num_soc_pools) array
                    p_vector = p_vector.swapaxes(0, 1)[:,self.lc_map]
                elif p_vector.ndim == 2:
                    p_vector = p_vector[:,self.lc_map]
                assert p_vector.shape == (self.num_soc_pools, self.lc_map.size)
            else:
                if p_vector.ndim == 0:
                    p_vector = p_vector[np.newaxis][np.newaxis]\
                        .repeat(self.lc_map.size, axis = 0)
                elif p_vector.ndim == 1:
                    p_vector = p_vector.ravel()[self.lc_map]\
                        .reshape((self.lc_map.size, 1))
                elif p_vector.ndim == 2:
                    p_vector = p_vector[:,self.lc_map].swapaxes(0, 1)
            # Result should be a 2D vectorized parameter array, either:
            #   (N,) or (S x N), where S is, e.g., each SOC pool
            assert p_vector.ndim in (1, 2)
            assert p_vector.shape[0] == self.lc_map.size or p_vector.shape[0] == 3
            self.params.add(key, p_vector)

    def _rescale_smrz(self, smrz0, smrz_min, smrz_max = 100):
        r'''
        Rescales root-zone soil-moisture (SMRZ) to increase plant sensitivity to
        very low water availability.

        $$
        \hat{\theta} &= 100 \times\left(
        \frac{\theta - \theta_{WP}}{\text{max}(\theta) - \theta_{WP}}
        \right) + 1\\
        \theta_{RZ} &= 95 \times
        \frac{\text{ln}(\hat{\theta})}{\text{ln(101)}} + 5
        $$

        Parameters
        ----------
        smrz0 : numpy.ndarray
            (N x T) array of original SMRZ data, in percent wetness [%]
            units for N sites and T time steps
        smrz_min : numpy.ndarray or float
            Site-level long-term minimum SMRZ (proportion saturation)
        smrz_max : numpy.ndarray or float
            Site-level long-term maximum SMRZ (proportion saturation); can
            optionally provide a fixed upper-limit on SMRZ

        Returns
        -------
        tuple
            A tuple of `(gpp, npp, tmult, wmult)`, each an numpy.ndarray
        '''
        smrz_min = np.array(smrz_min)
        if smrz_min.ndim == 1:
            smrz_min = smrz_min[:,np.newaxis]
        # Clip input SMRZ to the lower, upper bounds
        smrz0 = np.where(smrz0 < smrz_min, smrz_min, smrz0)
        smrz0 = np.where(smrz0 > smrz_max, smrz_max, smrz0)
        smrz_norm = np.add(np.multiply(100, np.divide(
            np.subtract(smrz0, smrz_min),
            np.subtract(smrz_max, smrz_min))), 1)
        # Log-transform normalized data and rescale to range between
        #   5.0 and 100% saturation)
        return np.add(
            np.multiply(95, np.divide(np.log(smrz_norm), np.log(101))), 5)

    def _setup_forward(
            self, drivers: Sequence, state: Sequence = None,
            dates: Sequence = None
        ) -> np.ndarray:
        'Pre-computes some vectorized quantities prior to forward run'
        self.check_drivers(drivers)
        litter = self.constants.litterfall
        if litter is None:
            assert dates is not None,\
                'Either: "litterfall" must be provided to TCF() or "dates" must be provided at runtime'
            assert len(dates) >= 365 and drivers[0].shape[-1] >= 365,\
                'At least 365 daily time steps must be provided to allow computation of annual NPP sum'
            assert hasattr(dates[0], 'year') and hasattr(dates[0], 'strftime'),\
                'The values of "dates" must be datetime.date or datetime.datetime instances'
        # GPP can be computed matrix-wise, in a single time step
        gpp = self.gpp(drivers[0:6])
        npp = self.params.CUE * gpp
        if litter is None:
            # Compute litterfall from the mean annual NPP sum
            npp_sum = climatology365(npp, dates).sum(axis = 0)
            self.constants.add('litterfall', npp_sum)
        # Pre-compute environmental constraints for soil RH
        tsoil, smsf = drivers[-2:]
        f_smsf = linear_constraint(self.params.smsf0, self.params.smsf1)
        # Swap axes here only to make time the major (first) axis
        tmult = arrhenius(tsoil, self.params.tsoil).swapaxes(0, 1)
        wmult = f_smsf(smsf).swapaxes(0, 1)
        return (gpp, npp, tmult, wmult)

    def check_drivers(self, drivers: Sequence):
        '''
        Checks that driver datasets have correct shape and units. Issues
        warnings otherwise.

        Parameters
        ----------
        drivers : Sequence
            Flat sequence of drivers or (P x ...) array for P drivers
        '''
        n = len(drivers)
        _drivers = dict(zip(self.required_drivers[0:n], drivers))
        if 'fPAR' in _drivers.keys():
            if np.nanmax(_drivers['fPAR']) > 1:
                warnings.warn('WARNING: fPAR might not have correct units; maximum value exceeds 1.0')
        if 'SMRZ' in _drivers.keys():
            if np.nanmax(_drivers['SMRZ']) <= 1:
                warnings.warn('WARNING: Root-zone soil moisture might not have correct units; maximum value is less than 1.0')
        if 'SMSF' in _drivers.keys():
            if np.nanmax(_drivers['SMSF']) <= 1:
                warnings.warn('WARNING: Surface soil moisture might not have correct units; maximum value is less than 1.0')

    def diagnose_emult(self, drivers):
        '''
        Returns the environmental constraint multiplier on GPP (Emult). This
        dimensionless quantity indicates the aggregate impact of
        meteorological conditions on GPP. Order of driver variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            UNSCALED Root-zone soil moisture wetness [percent, %]
            Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a flat sequence of P driver variables, each an array with
            (N x T) or (... x N x T) shape for N sites and T time steps; or a
            single array, with shape (P x N) or (P x N x T)

        Returns
        -------
        tuple
            4-tuple of (FT, Tmin, VPD, SMRZ) environmental constraints
        '''
        if len(drivers) == 5:
            fpar, par, tmin, vpd, smrz0 = drivers
            ft0 = np.where(tmin < 273.15, 0, 1)
        else:
            fpar, par, tmin, vpd, smrz0, ft0 = drivers
        # Rescale root-zone soil moisture
        smrz = self._rescale_smrz(smrz0, np.nanmin(smrz0, axis = -1))
        # Convert freeze-thaw flag to a multiplier (always 1 when thawed but
        #   potentially non-zero and less than 1 when thawed)
        ft = np.where(ft0 == 0, self.params.ft0, 1)
        # Get a function that constrains each met. driver to [0, 1]
        f_tmin = linear_constraint(self.params.tmin0, self.params.tmin1)(tmin)
        f_vpd = linear_constraint(
            self.params.vpd0, self.params.vpd1, 'reversed')(vpd)
        f_smrz = linear_constraint(self.params.smrz0, self.params.smrz1)(smrz)
        # Compute the environmental constraint
        return (ft, f_tmin, f_vpd, f_smrz)

    def diagnose_kmult(self, drivers):
        '''
        Returns the environmental constraint multiplier on RH (Kmult). This
        dimensionless quantity indicates the aggregate impact of
        meteorological conditions on heterotrophic respiration (RH). Order of
        driver variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            UNSCALED Root-zone soil moisture wetness [percent, %]
            Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]
            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness [percent, %]

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a flat sequence of P driver variables, each an array with
            (N x T) or (... x N x T) shape for N sites and T time steps; or a
            single array, with shape (P x N) or (P x N x T)

        Returns
        -------
        tuple
            2-tuple of (Tmult, Wmult), or (Tsoil, SMSF), environmental
            constraints
        '''
        tsoil, smsf = drivers[-2:]
        f_smsf = linear_constraint(self.params.smsf0, self.params.smsf1)
        # Swap axes here only to make time the major (first) axis
        tmult = arrhenius(tsoil, self.params.tsoil)
        wmult = f_smsf(smsf)
        return (tmult, wmult)

    def forward_run(
            self, drivers: Sequence, state: Sequence = None,
            dates: Sequence = None, track_state: bool = False,
            verbose: bool = True
        ) -> np.ndarray:
        '''
        Runs the TCF model forward in time for daily time steps. This is the
        recommended interface for most users. If `litterfall` was not provided
        to `TCF` at initialization, it will be necessary to provide at least
        365 daily steps and the `years` of each time step. Order of driver
        variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            UNSCALED Root-zone soil moisture wetness [percent, %]
            Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]
            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness [percent, %]

        GPP calculation is vectorized but RH and NEE calculation proceed
        step-wise because they depend on the model state (SOC).

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a flat sequence of P driver variables, each an array with
            (N x T) or (... x N x T) shape for N sites and T time steps; or a
            single array, with shape (P x N) or (P x N x T)
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            SOC state in each SOC pool
        dates : Sequence or numpy.ndarray or None
            If `litterfall` was not provided to `TCF` during initialization,
            you must provide a sequence of `datetime.date` instances, of
            length T for T time steps, indicating the current year of each
            time step.
        track_state : bool
            True to track (soil organic carbon) state at each time step,
            rather than only tracking the current state (Default: False)
        verbose : bool
            True to show a progress bar and other messages (Default: True)

        Returns
        -------
        tuple
            A 3-element tuple of (NEE, GPP, RH)
        '''
        if drivers[0].shape[-2] > drivers[0].shape[-1]:
            warnings.warn('WARNING: Axes of input arrays might not be in the required (..., N, T) order')
        # NOTE: Allowing for state variables other than SOC to be included in
        #   a later version
        soc = state
        if soc is None:
            soc = self.state.soc
        # NOTE: self.constants.litterfall is also initialized in this function
        gpp, npp, tmult, wmult = self._setup_forward(drivers, state, dates)
        # Pre-allocate output arrays
        rh = np.ones((3, *gpp.shape), dtype = np.float32) # (3 x N x T)
        nee = np.ones((*gpp.shape,), dtype = np.float32) # (N x T)

        # Determine the number of forward time steps
        if dates is not None:
            steps = np.arange(0, len(dates))
        else:
            try:
                steps = np.arange(0, drivers[0].shape[-1])
            except:
                raise ValueError('Could not determine number of time steps; provide a "dates" argument')

        # Create a way to track SOC state over time
        if track_state:
            if not hasattr(self.history, 'soc'):
                self.history.add('soc', soc[...,np.newaxis].repeat(
                    steps.size, axis = -1).astype(np.float32))

        litter0 = self.constants.litterfall # Available annual litterfall
        litter = litter0 / 365 # Default: Allocate equal fraction every day

        for t in tqdm(steps, disable = not verbose):
            # If litterfall is scheduled: is a varying fraction of the annual
            if self.constants.litterfall_rate is not None:
                litter = litter0 * self.constants.litterfall_rate[:,t]
            if track_state:
                self.history.soc[...,t] = soc
            rh_t = np.empty((3, litter.shape[0])) # Allocate RH(t) array
            for pool in range(0, soc.shape[0]):
                rh_t[pool] = self.params.decay_rates[pool] *\
                    wmult[t] * tmult[t] * soc[pool]
            # Compute SOC change
            dc1 = (litter * self.params.f_metabolic.T) - rh_t[0,...]
            dc2 = (litter * (1 - self.params.f_metabolic.T)) - rh_t[1,...]
            dc3 = (self.params.f_structural.T * rh_t[1,...]) - rh_t[2,...]
            for i, delta in enumerate([dc1, dc2, dc3]):
                delta[np.isnan(delta)] = 0 # Protect against NaN contamination
                soc[i] += delta[0]
            # "the adjustment...to account for material transferred into the slow
            #   pool during humification" (Jones et al. 2017, TGARS, p.5); note
            #   that this is a loss FROM the "medium" (structural) pool
            rh_t[1,...] = rh_t[1,...] * (1 - self.params.f_structural.T)
            # Record RH and NEE at this time step
            rh[...,t] = rh_t
            nee[...,t] = rh_t.sum(axis = 0) - npp[...,t]
        return (nee, gpp, rh)

    def gpp(self, drivers: Sequence) -> np.ndarray:
        '''
        Calculates gross primary production (GPP) under prevailing climatic
        climatic conditions. Order of driver variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            UNSCALED Root-zone soil moisture wetness [percent, %]
            Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]

        The FT state is optional; if that axis of the data cube is not
        provided, FT state will be calculated from Tmin using a threshold
        of 32 degrees F (273.15 deg K).

        Unit of time should be consistent with the units of PAR. For example,
        if PAR is given as [MJ m-2 day-1], then GPP will be in units of
        [g C m-2 day-1]. It's assumed the other driver data are representative
        of that time step (e.g., daily averages). GPP doesn't depend on model
        state, so it can be estimated for an arbitrary number of time steps.

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a flat sequence of P driver variables, each an array with
            (N x T) or (... x N x T) shape for N sites and T time steps; or a
            single array, with shape (P x N) or (P x N x T)

        Returns
        -------
        numpy.ndarray
            Gross primary production (GPP) in [g C m-2 time-1] where time is
            the time step of the PAR data, e.g., [g C m-2 day-1]
        '''
        self.check_drivers(drivers)
        if len(drivers) == 5:
            fpar, par, _, _, _ = drivers
        else:
            fpar, par, _, _, _, _ = drivers
        ft, f_tmin, f_vpd, f_smrz = self.diagnose_emult(drivers)
        emult = ft * f_tmin * f_vpd * f_smrz
        return par * fpar * emult * self.params.LUE

    def rh(
            self, drivers: Sequence, state: Sequence = None
        ) -> np.ndarray:
        '''
        Calculates heterotrophic respiration (RH) based on the available
        soil organic carbon (SOC) state and prevailing climatic conditions.
        This calculation is NOT vectorized, so can only be applied to a single
        time point. Order of driver variables should be:

            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness [percent, %]

        Unit of time should be consistent with the units of PAR and the
        turnover time (`decay_rate`). It's assumed that PAR is denominated
        by daily time steps, so RH would be given in [g C m-2 day-1].

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a flat sequence of P driver variables, each an array with
            (N x T) or (... x N x T) shape for N sites and T time steps; or a
            single array, with shape (P x N) or (P x N x T)
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            SOC state in each SOC pool

        Returns
        -------
        numpy.ndarray
            A (3 x N) array representing the RH flux from each SOC pool, in
            units of [g C m-2] per unit time, most likely [g C m-2 day-1]
        '''
        if hasattr(drivers, 'ndim'):
            assert drivers.ndim <= 2,\
                'TCF.rh() computes a single time step; only (P x N) driver arrays should be provided'
        # NOTE: Allowing for state variables other than SOC to be included in
        #   a later version
        soc = state
        if soc is None:
            soc = self.state.soc
        tsoil, smsf = drivers # Unpack met. drivers
        # Take transpose of parameter vectors here because the driver datasets
        #   are cross-sectional; i.e., smsf and tsoil are 1D vectors
        f_smsf = linear_constraint(self.params.smsf0.T, self.params.smsf1.T)
        tmult = arrhenius(tsoil, self.params.tsoil.T)
        wmult = f_smsf(smsf)
        rh = wmult * tmult * self.params.decay_rates * soc
        # "the adjustment...to account for material transferred into the slow
        #   pool during humification" (Jones et al. 2017, TGARS, p.5); note
        #   that this is a loss FROM the "medium" (structural) pool
        rh[1,...] = rh[1,...] * (1 - self.params.f_structural.T)
        return rh

    def setup_litterfall(
            self, dates: Sequence, litterfall_rate: Sequence,
            period_days: int = 8):
        '''
        Create a schedule for daily litterfall allocation. This step is
        optional and, without it, the default behavior is to allocate an
        equal daily fraction (i.e., 1/365) of total annual litterfall. The
        resulting litterfall schedule is stored in the `constants` namespace
        as `constants.litterfall_rate`.

        Parameters
        ----------
        dates : Sequence or numpy.ndarray
            A sequence of `datetime.date` instances, of length T for T time
            steps, indicating the current year of each time step.
        litterfall_rate : numpy.ndarray
            An (P x N) array specifying the litterfall rate in each period
            (P periods per year) for N pixels. There are 46 8-day periods in
            a year.
        period_days : int
            The number of days; defaults to 8 (8-day periods) to match the
            8-day composite cycle of MODIS/VIIRS datasets
        '''
        n_periods = 1 + (365 // period_days) # e.g., 46 8-day periods in 365 days
        _preamble = f'Expected "litterfall_rate" to be shape ({n_periods} x N)'
        assert litterfall_rate.shape[0] == n_periods,\
            f'{_preamble} but first axis does not have {n_periods} elements'
        # Get a sequence of (e.g., 8-day) periods
        periods = np.arange(0, n_periods)\
            .reshape((n_periods, 1)).repeat(period_days, axis = 1)
        periods = periods.ravel()[:365]
        jdates = np.array(ordinals365(dates)) - 1
        assert jdates.max() == 364 # Python starts counting at zero
        # Index into periods [0,45], which can then index into litter_rate,
        #   a (46 x N) array, to produce an (T x N) array, then swap axes
        litterfall_rate = litterfall_rate[periods[jdates],:].swapaxes(0, 1)
        # Finally, convert litterfall rate from (1/period_days) to (1/day);
        #   i.e., for an 8-day period, instead of "this amount per 8 days"
        #   we want a daily amount
        litterfall_rate /= period_days
        # Check that, for a given year, 100% of litterfall is allocated; this
        #   is like a checksum for the schedule: The fractions in a year
        #   should add up to (close to) 1.0
        n_years = len(dates) // 365 # Expected number of years
        assert (np.abs(litterfall_rate.sum(axis = -1) - n_years) < 1).all(),\
            f'Summing daily "litterfall_rate" over time did not result in 100% of litterfall allocated each year; check data values or the value of "period_days"'
        self.constants.add('litterfall_rate', litterfall_rate)

    def spin_up(
            self, dates: Sequence, drivers: Sequence, state: Sequence = None,
            max_cycles: int = 1000, threshold: float = 1, verbose: bool = True,
            verbose_type = 'tqdm'
        ) -> np.ndarray:
        '''
        Repeatedly cycle climatology until SOC state reaches equilibrium. See
        `TCF.forward_run()` for details on `drivers` and `state` arguments.

        Parameters
        ----------
        dates : Sequence or numpy.ndarray
            A sequence of `datetime.date` instances, of length T for T time
            steps
        drivers : Sequence or numpy.ndarray
            Either a flat sequence of P driver variables, each an array with
            (N x T) or (... x N x T) shape for N sites and T time steps; or a
            single array, with shape (P x N) or (P x N x T)
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            SOC state in each SOC pool
        max_cycles : int
            Maximum number of climatology cycles (365-day years) to apply
            (Default: 100)
        threshold : float
            Threshold for inter-annual change in NEE [g C m-2 year-1]; when the
            difference in the annual NEE sum is less than this number, spin-up
            is complete (Default: 1 g C m-2 year-1)
        verbose : bool
            True to show a progress bar and other messages (Default: True)

        Returns
        -------
        numpy.ndarray
            The history of change in annual NEE
        '''
        # NOTE: Allowing for state variables other than SOC to be included in
        #   a later version
        if state is None:
            assert hasattr(self.state, 'soc'), 'No prior soil organic carbon ("soc") state defined'
        else:
            self.state.soc = state
        soc = self.state.soc

        clim = []
        for each in drivers:
            # Reshape the (365 x ...) arrays to (... x 365) arrays after
            #   calculating a climatology
            clim.append(climatology365(each, dates).swapaxes(0, -1))
        tolerance = np.nan * np.ones((soc.shape[-1], max_cycles), np.float32)
        disable = (not verbose or not verbose_type == 'tqdm')
        for cycle in tqdm(range(0, max_cycles), disable = disable):
            nee, gpp, rh = self.forward_run(drivers, soc, dates, verbose = False)
            # Diagnostics
            # rh_sum = rh.sum(axis = 0).sum(axis = -1)
            # npp_sum = (gpp * self.params.CUE).sum(axis = -1)
            nee_sum = np.nansum(nee, axis = -1)
            if cycle == 0:
                nee_last = nee_sum
            else:
                tolerance[:,cycle] = (nee_last - nee_sum)
                nee_last = nee_sum
                if (np.abs(tolerance[:,cycle]) < threshold).all():
                    break
            # Diagnostics
            # rh_track[:,step] = rh_sum
            # npp_track[:,step] = npp_sum
            # soc_track[:,step] = self.state.soc.sum(axis = 0)
            if cycle > 0 and verbose and verbose_type != 'tqdm':
                print(
                    'Change in annual NEE sum [SOC state]: %.2f, [%.0f]' %
                    (np.nanmean(tolerance[:,cycle]), self.state.soc.sum()))
        return tolerance
