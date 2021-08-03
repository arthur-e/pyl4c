r'''
The Dual Arrhenius Michaelis-Menten (DAMM) soil decomposition model (from
Davidson et al. 2012).

$$
R_H = V_{\mathrm{max}}\, \frac{S_X}{K_{M_S} + S_X} \frac{O_2}{K_{M_{O_2}} + O_2}
$$

NOTE: The only term in the equation that is not unitless (either intrinsically
or because the units cancel out) is `V_max`; therefore, the units of alpha,
which determine the units of `V_max`, ultimately determine the units of
respiration.
'''

import numpy as np
from pyl4c import Namespace

class AbstractDAMM(object):
    '''
    Abstract Dual Arrhenius Michaelis-Menten (DAMM) model framework. Not
    intended for end-user instantiation.
    '''
    def cbar0(self, litterfall, soil_m, soil_t, porosity, fmet, fstr):
        '''
        An alternative method for estimating the initital C pool sizes, this
        analytical approaches solves for C storage after setting the
        differential equation governing change in C to zero. See README for
        the equations. The C storage magnitudes tend to be higher because
        `V_max` is very small compared to litterfall, leading to very large
        empirical C storage estimates.

        Parameters
        ----------
        litterfall : numpy.ndarray
            Average daily litterfall [g C cm-3 day-1]
        soil_m : numpy.ndarray
            365-day soil moisture (%) climatology (365 x N ...)
        soil_t : numpy.ndarray
            365-day soil temperature (°K) climatology (365 x N ...)
        porosity : numpy.ndarray
            Total porosity, between [0, 1]
        fmet : float
            Fraction of daily litterfall entering metabolic pool
        fstr : float
            Fraction of structural pool transferred to the recalcitrant pool
            during "humification"

        Returns
        -------
        tuple
            (c0, c1, c2) numpy.ndarray instances, one for each of the
            steady-state C pools (Units: g cm-3)
        '''
        # Calculate the DAMM parameters needed in the steady-state C eq.
        alpha0, alpha1, alpha2, p, d_liq = (
            self.get(p)
            for p in ('alpha0', 'alpha1', 'alpha2', 'p', 'd_liq')
        )
        try:
            km_s = self.params[self.labels.index('km_s')]
        except ValueError:
            km_s_int, km_s_slope = (
                self.get(p)
                for p in ('km_s_int', 'km_s_slope')
            )
            km_s = km_s_int + (km_s_slope * soil_t)

        # Convert from [Mg C cm-3 hr-1] to [g C cm-3 day-1]
        vmax0 = self.v_max(soil_t, alpha0) * 1e6 * 24
        vmax1 = self.v_max(soil_t, alpha1) * 1e6 * 24
        vmax2 = self.v_max(soil_t, alpha2) * 1e6 * 24
        _, conc_O2 = self.concentrations(np.nan, soil_m, porosity)
        mm_O2 = conc_O2 / (self.get('km_O2') + conc_O2)
        sx_coef = np.multiply(p * d_liq, np.power(soil_m / 100, 3))
        c0 = (litterfall * fmet * km_s) / (sx_coef * (
            (vmax0 * mm_O2) - (litterfall * fmet)))
        c1 = (litterfall * (1 - fmet) * km_s) / (sx_coef * (
            (vmax1 * mm_O2) - (litterfall * (1 - fmet))))
        c2 = -(vmax1 * c1 * fstr * km_s) / (
            (sx_coef * (((vmax1 * c1 * fstr) + (vmax2 * c1)))) + (vmax2 * km_s))
        substrate_by_pool = []
        # Empirical C is now the steady-state value for each day of the
        #   climatological year; calculate the daily average
        for empirical_C in (c0, c1, c2):
            # Calculate mean across 365 days
            substrate_by_pool.append(np.nanmean(
                np.where(empirical_C < 0, 0, empirical_C), axis = 0))
        return substrate_by_pool

    def cbar(self, rh, soil_m, soil_t, porosity, perc = 80):
        r'''
        As in the L4C calibration, calculate the empirical C storage, based on
        observed RH, since we don't yet know how much C we have to burn. Note
        that this SINGLE value per (flux tower) site is used as a "constant
        effective SOC factor" (Jones et al. 2017) throughout calibration, for
        every time step. This is because C substrate pools are assumed to be
        in the steady state for calibration. The empirical C storage
        magnitudes estimated by this function compare well with the
        SoilGrids 250m global 0-5 cm SOC density estimates of 10-60 kg m-2.

        $$
        \bar{C} = \frac{R_H\, k_{M_{S_x}}}{(p\, D_{\mathrm{liq}}\, \theta^3)
            [V_{\mathrm{max}}\, [O_2](k_{M_{O_2}} + [O_2])^{-1} - R_H]}
        $$

        Parameters
        ----------
        rh : numpy.ndarray
            RH in g C m-2 day-1
        soil_m : numpy.ndarray
            Soil moisture (%)
        soil_t : numpy.ndarray
            Soil temperature (°K)
        porosity : numpy.ndarray
            Total porosity, between [0, 1]
        perc : int or float
            Empirical Cbar percentile to return

        Returns
        -------
        tuple
            (c0, c1, c2) numpy.ndarray instances, one for each of the
            steady-state C pools (Units: g cm-3)
        '''
        # Calculate the DAMM parameters needed in the steady-state C eq.
        alpha0, alpha1, alpha2, ea, p, d_liq = (
            self.get(p)
            for p in ('alpha0', 'alpha1', 'alpha2', 'ea', 'p', 'd_liq')
        )
        try:
            km_s = self.params[self.labels.index('km_s')]
        except ValueError:
            km_s_int, km_s_slope = (
                self.get(p)
                for p in ('km_s_int', 'km_s_slope')
            )
            km_s = km_s_int + (km_s_slope * soil_t)

        # Let's do this in the units of km_s, because substrate is needed
        #   in these units (g C cm-3)
        #   -- rh is in g m-2 day-1
        #   -- km_s is in g cm-3
        #   -- v_max is in Mg cm-3 hr-1
        # Calculate coefficients for substrate concentration
        sx_coef = np.multiply(p * d_liq, np.power(soil_m / 100, 3))
        # Calculate oxygen concentration at reaction site; then MM constraint
        _, conc_O2 = self.concentrations(np.nan, soil_m, porosity)
        mm_O2 = conc_O2 / (self.get('km_O2') + conc_O2) # (dimensionless)
        # Convert g C m-2 to g C m-3, then denominator from m-3 to cm-3
        rh = (rh / self.constants.soil_depth_m) / 1e6
        # Estimate each soil C pool (differentiated by base rate or alpha)
        substrate_by_pool = []
        for alpha in (alpha0, alpha1, alpha2):
            v_max_i = self.v_max(soil_t, alpha, ea) # Calculate base rate
            v_max_i = (v_max_i * 1e6) # Convert Mg C to g C
            empirical_C = np.divide(
                np.multiply(rh, km_s),
                np.multiply(sx_coef, np.subtract( # Hourly to daily Vmax
                    np.multiply(v_max_i * 24, mm_O2), rh)))
            # This empirical approach may result in C values < 0; assign 0
            substrate_by_pool.append(
                np.nanpercentile(
                    np.where(empirical_C < 0, 0, empirical_C), perc, axis = 0))
        return substrate_by_pool

    def concentrations(self, substrate, soil_m, porosity):
        '''
        For a single C pool, returns substrate and oxygen concentrations.
        Units for C substrate are g C cm-3 (same as input argument) and units
        for O2 are dimensionless--essentially, it's `d_gas` times the proportion
        of total volume (soil + water + air) occupied by O2.

        Parameters
        ----------
        substrate : numpy.ndarray
            Soil C substrate (g cm-3)
        soil_m : numpy.ndarray
            Soil moisture (%)
        porosity : numpy.ndarray
            Total porosity, between [0, 1]

        Returns
        -------
        tuple
            `(conc_Sx, conc_O2)` where `conc_Sx` is the concentration of
            substrate, `conc_O2` is the concentration of O2
        '''
        air_frac_O2 = 0.2095 # L O2 L-1 air (20.95%)
        p = self.get('p')
        d_liq = self.get('d_liq')
        d_gas = self.get('d_gas')
        # Calculate substrate concentration at reaction site
        conc_Sx = np.multiply(substrate,
            np.multiply(p * d_liq, np.power(soil_m / 100, 3)))
        # Calculate oxygen concentration at reaction site; then MM constraint
        a = (porosity - (soil_m / 100))
        conc_O2 = d_gas * air_frac_O2 * np.power(np.where(a < 0, 0, a), (4/3))
        return (conc_Sx, conc_O2)

    def get(self, parameter):
        '''
        Retrieves a parameter value whether it is fixed or a free parameter.
        This provides flexibility for versions of the model that may differ
        in free versus fixed parameters. Constants take precedence--if a value
        was fit but a constant value is found, the constant is used.

        Parameters
        ----------
        parameter : str
            Name of the parameter to retrieve
        '''
        if hasattr(self.constants, parameter):
            return getattr(self.constants, parameter)
        return self.params[self.labels.index(parameter)]

    def total_respiration(self, *args, **kwargs):
        '''
        Calculates the sum of respiration in each C pool. See `respiration()`.

        Returns
        -------
        numpy.ndarray
        '''
        rh = self.respiration(*args, **kwargs)
        respiration = np.zeros(rh[0].shape)
        for i in range(0, len(rh)):
            np.add(respiration, rh[i], out = respiration)
        return respiration

    def v_max(self, soil_t, alpha, ea = None):
        '''
        For a single C pool, returns maximum rate on enzymatic reaction in
        Mg C cm-3 hr-1. NOTE: Units of base rate (megagrams, Mg) are chosen
        to improve convergence in model fitting.

        Parameters
        ----------
        soil_t : numpy.ndarray
            Soil temperature (°K)
        alpha : numpy.ndarray
            Base rate/ pre-exponential factor (Mg C cm-3 hr-1)
        ea : numpy.ndarray
            Activation energy (kJ mol-1)

        Returns
        -------
        numpy.ndarray
        '''
        ea = self.get('ea') if ea is None else ea
        r_gas = 8.314472e-3 # Universal gas constant (kJ K-1 mol-1)
        return np.multiply( # Mg C cm-3 hr-1
            alpha, np.exp(-np.divide(ea, np.multiply(r_gas, soil_t))))


class DAMMDecompositionModel(AbstractDAMM):
    '''
    The DAMM decomposition model as reported by Davidson et al. (2012), with
    some changes: Support for multiple soil C pools; Additional free
    parameters (not bragging about it, these have to be fit); and Changed the
    units of the pre-exponential factor to better condition optimization.

    Free parameters are:

    - `alpha`: Pre-exponential factor of enzymatic reaction with `S_x`
               (Mg C cm-3 hr-1), note this is *Megagrams* of C...
    - `ea`:    Activation energy of enzymatic reaction with `S_x` (kJ mol-1)
    - `km_s`:  Michaelis-Menten coefficient for subtrate, using the constant-
               value form (g C cm-3)
    - `p`:     Proportion of `C_total` that is soluble
    - `d_liq`: Diffusion coefficient of substrate in liquid phase
    - `d_gas`: Diffusion coefficient of `O_2` in air

    NOTE: CUE is potentially another free parameter, but it has no relevance
    in running DAMM, only in fitting the model with unknown C storage/
    substrate.
    '''
    parameter_names = (
        'alpha0', 'alpha1', 'alpha2', 'ea', 'km_s', 'p', 'd_liq', 'd_gas')

    def __init__(self, params = None, soil_depth_cm = 5, km_O2 = 0.121):
        self.constants = Namespace()
        self.constants.add('km_O2', km_O2)
        self.constants.add('soil_depth_cm', soil_depth_cm)
        self.constants.add('soil_depth_m', soil_depth_cm / 100)
        self.params = params
        if params is None:
            # Use parameters from Davidson et al. (2012)
            self.params = (53.8, 0, 0, 72.26, 9.95e-7, 4.14e-4, 3.17, 1.67, np.nan)

    def respiration(self, substrate, soil_m, soil_t, porosity):
        '''
        Calculates daily total RH for all soil pools, g C m-2 day-1.

        Parameters
        ----------
        substrate : numpy.ndarray
            Soil C substrate (g cm-3) in each pool (3-tuple)
        soil_m : numpy.ndarray
            Soil moisture (%)
        soil_t : numpy.ndarray
            Soil temperature (°K)
        porosity : numpy.ndarray
            Total porosity, between [0, 1]

        Returns
        -------
        tuple
            `(rh0, rh1, rh2)` numpy.ndarray instances, one for each of the
            C pools (Units: g m-2 day-1)
        '''
        assert len(substrate) == 3,\
            'Need a substrate value for each of 3 pools'
        alpha0, alpha1, alpha2, ea, km_s, p, d_liq = (
            self.params[i] for i in range(0, 7)
        )
        respiration = []
        for i, alpha_i in enumerate((alpha0, alpha1, alpha2)):
            v_max_i = self.v_max(soil_t, alpha_i, ea)
            sx_i, conc_O2 = self.concentrations(substrate[i], soil_m, porosity)
            # Calculate Michaelis-Menten coefficients
            mm_sx = sx_i / (km_s + sx_i) # Units (g C cm-3) cancel out (dimensionless)
            mm_O2 = conc_O2 / (self.constants.km_O2 + conc_O2) # (dimensionless)
            rh = v_max_i * mm_sx * mm_O2 # Mg C cm-3 hr-1; need g C m-2 day-1
            # First convert Mg C to g C, then cm-3 to cm-2, then from
            #   cm-2 to m-2, then from hourly to a daily flux (24 hours/ day)
            resp = (((1e6 * rh) * self.constants.soil_depth_cm) * 1e4) * 24
            respiration.append(np.where(resp < 0, 0, resp))
        return respiration


class DAMMDecompositionModel2(AbstractDAMM):
    '''
    The DAMM decomposition model as reported by Davidson et al. (2012), with
    some changes: Support for multiple soil C pools; Additional free
    parameters (not bragging about it, these have to be fit); and Changed the
    units of the pre-exponential factor to better condition optimization.
    This model assumes that the Michaelis-Menten coefficient for substrate is
    *not constant* w.r.t. temperature (the slope-intercept form of km_s). It
    also allows any free parameter to be specified as a constant.

    Free parameters are:

    - `alpha`:      Pre-exponential factor of enzymatic reaction with `S_x`
                    (Mg C cm-3 hr-1), note this is *Megagrams* of C...
    - `ea`:         Activation energy of enzymatic reaction with `S_x`
                    (kJ mol-1)
    - `km_s_int`:   Intercept of Michaelis-Menten (MM) coefficient for
                    substrate (g C cm-3)
    - `km_s_slope`: Slope of Michaelis-Menten (MM) coefficient for
                    substrate (g C cm-3 K-1)
    - `p`:          Proportion of `C_total` that is soluble
    - `d_liq`:      Diffusion coefficient of substrate in liquid phase
    - `d_gas`:      Diffusion coefficient of `O_2` in air
    - `km_O2`:      Half-saturation (MM) coefficient for diffusion of O2

    NOTE: CUE is potentially another free parameter, but it has no relevance
    in running DAMM, only in fitting the model with unknown C storage/
    substrate.
    '''
    parameter_names = (
        'alpha0', 'alpha1', 'alpha2', 'ea', 'km_s_int', 'km_s_slope',
        'p', 'd_liq', 'd_gas', 'km_O2')

    def __init__(self, params, soil_depth_cm = 5, **kwargs):
        self.constants = Namespace()
        self.constants.add('soil_depth_cm', soil_depth_cm)
        self.constants.add('soil_depth_m', soil_depth_cm / 100)
        self.labels = list(self.parameter_names).copy()
        self.params = params
        for key, value in kwargs.items():
            self.constants.add(key, value)
            # Don't allow constants to appear in the parameters list
            if key in self.labels:
                # May need to re-build parameters list
                self.labels.remove(key)
        for name in self.parameter_names:
            assert name in self.labels or hasattr(self.constants, name),\
                'Required parameter "%s" must be specified either as a constant nor a free parameter' % name

    def respiration(self, substrate, soil_m, soil_t, porosity):
        '''
        Calculates daily total RH for all soil pools, g C m-2 day-1.

        Parameters
        ----------
        substrate : numpy.ndarray
            Soil C substrate (g cm-3) in each pool (3-tuple)
        soil_m : numpy.ndarray
            Soil moisture (%)
        soil_t : numpy.ndarray
            Soil temperature (°K)
        porosity : numpy.ndarray
            Total porosity, between [0, 1]

        Returns
        -------
        tuple
            (rh0, rh1, rh2) numpy.ndarray instances, one for each of the
            C pools (Units: g m-2 day-1)
        '''
        assert len(substrate) == 3,\
            'Need a substrate value for each of 3 pools'
        alpha0, alpha1, alpha2, ea, km_s_int, km_s_slope = (
            self.params[i] for i in range(0, 6)
        )
        respiration = []
        for i, a_i in enumerate((alpha0, alpha1, alpha2)):
            v_max_i = self.v_max(soil_t, a_i, ea)
            sx_i, conc_O2 = self.concentrations(
                np.array(substrate[i]), soil_m, porosity)
            # Calculate Michaelis-Menten coefficients
            km_s = km_s_int + (km_s_slope * soil_t)
            mm_sx = sx_i / (km_s + sx_i) # Units (g C cm-3) cancel out
            mm_O2 = conc_O2 / (self.get('km_O2') + conc_O2) # (dim.less)
            rh = v_max_i * mm_sx * mm_O2 # Mg C cm-3 hr-1
            # First convert Mg C to g C, then cm-3 to cm-2, then from
            #   cm-2 to m-2, then from hourly to a daily flux (24 hours/ day)
            resp = (((1e6 * rh) * self.constants.soil_depth_cm) * 1e4) * 24
            respiration.append(np.where(resp < 0, 0, resp))
        return respiration


def g_m2_to_g_cm3(value, soil_depth_cm = 5):
    '''
    Converts flux/ SOC stock from g m-2 to g cm-3.

    Parameters
    ----------
    value : int or float
        Value in g m-2
    soil_depth_cm : int
        Depth of the soil, in centimeters

    Returns
    -------
    float
        Value in g cm-3
    '''
    return (value / (soil_depth_cm / 100)) / 1e6


def g_cm3_to_g_m2(value, soil_depth_cm = 5):
    '''
    Converts flux/ SOC stock from g cm-3 to g m-2.

    Parameters
    ----------
    value : int or float
        Value in g cm-3
    soil_depth_cm : int
        Depth of the soil, in centimeters

    Returns
    -------
    float
        Value in g m-2
    '''
    return (value * soil_depth_cm) * 1e4


if __name__ == '__main__':
    # Test model implementation matches the description by Davidson et al.
    #   (2012, Table 3) by running...
    # Here we also show that the code is vectorized.
    substrate = (np.array(0.048).repeat(4).reshape((2,2)),
    np.zeros((2,2)), np.zeros((2,2)))
    soil_temp = np.array(273.15 + 37.7).repeat(4).reshape((2,2))
    soil_moisture = np.array(50).repeat(4).reshape((2,2))
    porosity = np.array(1 - (0.8 / 2.52)).repeat(4).reshape((2,2))

    damm = DAMMDecompositionModel(soil_depth_cm = 10)
    rh = damm.respiration(substrate, soil_moisture, soil_temp, porosity)
    assert np.all(rh[0].round(2) == 19.04), 'DAMMDecompositionModel: Failed test'
    print('DAMMDecompositionModel: Passed test')

    # km_s_int = 0 # Back-solved for km_s_slope given km_s = 9.95e-7
    # km_s_slope = 3.2009e-09
    damm = DAMMDecompositionModel2(
        #        alpha        E_a   km_s_slope        p D_liq D_gas
        params = (53.8, 0, 0, 72.26, 0, 3.2e-9, 4.14e-4, 3.17, 1.67, np.nan),
        km_O2 = 0.121, soil_depth_cm = 10)
    rh = damm.respiration(substrate, soil_moisture, soil_temp, porosity)
    assert np.all(rh[0].round(2) == 19.04), 'DAMMDecompositionModel2: Failed test'
    print('DAMMDecompositionModel2: Passed test')

    # Vectorization in parameters
    damm = DAMMDecompositionModel2(
        params = (
            np.array([53.8, 56.81]), np.array([0, 8.297]), np.array([0, 6.27e-2]),
            np.array([72.26, 71.13]), np.array([0, -6.28e-3]), np.array([3.2e-9, 2.34e-5]),
            np.array([4.14e-4, 0.126]), np.array([3.17, 23.309]), np.array([1.67, 4.648]),
            np.array([0.121, 8.367e-2])),
        soil_depth_cm = 10)
    rh = damm.respiration([0.048, 0, 0], 50, 310.85, 0.6825)
    assert np.all(rh[0].round(2) == 19.04), 'DAMMDecompositionModel2: Failed test'
    print('DAMMDecompositionModel2: Passed test')
