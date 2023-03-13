'''
Specialized scientific functions for biogeophysical variables and L4C model
processes.
'''

import numpy as np
from functools import partial
from scipy.ndimage import generic_filter
from scipy.linalg import solve_banded
from scipy.sparse import dia_matrix
from pyl4c import suppress_warnings
from pyl4c.data.fixtures import HDF_PATHS, BPLUT
from pyl4c.utils import get_pft_array, subset
from pyl4c.stats import ols, ols_variance, linear_constraint

def arrhenius(
        tsoil, beta0: float, beta1: float = 66.02, beta2: float = 227.13):
    r'''
    The Arrhenius equation for response of enzymes to (soil) temperature,
    constrained to lie on the closed interval [0, 1].

    $$
    f(T_{SOIL}) = \mathrm{exp}\left[\beta_0\left( \frac{1}{\beta_1} -
        \frac{1}{T_{SOIL} - \beta_2} \right) \right]
    $$

    Parameters
    ----------
    tsoil : numpy.ndarray
        Array of soil temperature in degrees K
    beta0 : float
        Coefficient for soil temperature (deg K)
    beta1 : float
        Coefficient for ... (deg K)
    beta2 : float
        Coefficient for ... (deg K)

    Returns
    -------
    numpy.ndarray
        Array of soil temperatures mapped through the Arrhenius function
    '''
    a = (1.0 / beta1)
    b = np.divide(1.0, np.subtract(tsoil, beta2))
    # This is the simple answer, but it takes on values >1
    y0 = np.exp(np.multiply(beta0, np.subtract(a, b)))
    # Constrain the output to the interval [0, 1]
    return np.where(y0 > 1, 1, np.where(y0 < 0, 0, y0))


def bias_correction_parameters(
        series, npoly: int = 1, cutoff: float = 1, var_cutoff: float = None,
        add_intercept: bool = True):
    '''
    Calculate the bias correction parameters for two overlapping time series,
    nominally the Nature Run and L4C Operational products, of a given
    variable using quantile mapping. For example, can correct the bias in
    Nature Run (2000-2017) against the L4C Ops record (2015-Present) by
    fitting bias correction parameters for the overlap period 2015-2017.
    Model can be specified:

        y = alpha + X beta_0 + X^2 beta_1 + ...

    NOTE: Because Nature Run and L4C Ops compare very well in some locations,
    a degree-1 polynomial (straight line) is fit first (regardless of npoly);
    if this solution produces corrections that are <1 gC m^-2, the degree-1
    solution is used. In some areas, there is a strong linear correspondence
    between most measurements but a small number have a super-linear
    relationship that is poorly fit by a degree-2 polynomial; in these cases
    (where model variance of the degree-2 fit is > var_cutoff), the degree-1
    solution is used. Forcing the line of best fit through the origin (with
    intercept=False) is also not recommended.

    Parameters
    ----------
    series : numpy.ndarray
        A (t x 2) NumPy array where t rows correspond to t time steps and
        each column is a product; the first column is the reference product
        or dependent variable in the linear bias correction.
    npoly : int
        Degree of the polynomial to use in bias correction (Default: 1)
    cutoff : float
        Cutoff for the degree-1 bias correction, in data units (e.g.,
        1 g C m-2 day-1); defaults to 1.0, i.e., the residual after correction
        must be greater than 1 g C m-2 day-1, which is the average impact of
        L4SM versus model-only observations. If this cutoff is exceeded, the
        degree-1 solution is returned.
    var_cutoff : float or None
        Cutoff in variance for higher-order solutions; if the residual model
        variance exceeds this threshold for the degree-N solution, then return
        the degree (N-1) solution (Default: None)
    add_intercept : bool
        True to add a the y-intercept term (Default: True)

    Returns
    -------
    numpy.ndarray
        A vector of length N + 1 where N is the degree of the polynomial
        fit requested
    '''
    def xmat(x, npoly):
        # Creates the design/ model matrix for a polynomial series
        # Add a column for each power of the requested polynomial series
        x = np.repeat(x.reshape((t, 1)), npoly, axis = 1)
        for i in range(1, npoly):
            # Calculate X^n for n up to N powers
            x[:,i] = np.power(x[:,0], npoly + 1)
        return x

    def fit(x, y, npoly):
        # Fits the model using OLS
        # If all of the Y values are NaN
        if np.all(np.isnan(y)): return np.ones((npoly + 1,)) * np.nan
        try:
            return ols(xmat(x, npoly), y, add_intercept)
        except np.linalg.linalg.LinAlgError:
            return np.ones((npoly + 1,)) * np.nan

    # Sort the input series from low -> high
    t = series.shape[0]
    y = np.sort(series[:,0])
    x = np.sort(series[:,1])

    # For some pixels, the time series has zero variance, and this can produce
    #   unstable OLS estimates (e.g., zero slope)
    if np.var(y) == 0 or np.var(x) == 0:
        # Return coefficients: (0, 1, 0, ..., 0)
        return np.hstack(((0, 1), list(0 for i in range(1, npoly))))

    if np.var(y) == 0 and np.var(x) == 0:
        # Intercept (mean) is the only necessary predictor
        return np.hstack(((1, 0), list(0 for i in range(1, npoly))))

    fit1 = np.hstack(
        (fit(x, y, npoly = 1), list(0 for i in range(1, npoly))))
    if npoly == 1:
        return fit1

    # First, try a degree-1 polynomial (straight-line) fit; if the bias
    #   correction slope is such that the correction is < 1 gC/m^-2,
    #   which is similar to the average impact of L4SM vs. model-only
    #   observations, then use the degree-1 fit parameters
    if x.mean() - (fit1[1] * x.mean()) < cutoff:
        return fit1

    # Second, starting with the simpler model, check if progressively more
    #   complicated models (up to a maximum of npoly) really do fit the data
    #   better; if not, or if the model variance is above a cutoff, use the
    #   next most-complicated model (last_model)
    last_model = fit1 # Starting with the simplest model...
    for p in range(2, npoly + 1):
        model = fit(x, y, npoly = p)
        # Calculates unbiased estimate of model variance
        model_var = ols_variance(xmat(x, p), y, model, add_intercept)
        # Without a cutoff for guidance, if the model variance of the degree-1
        #   fit is lower than that of the degree-2 fit...
        if var_cutoff is None:
            if model_var > ols_variance(
                    xmat(x, 1), y, last_model[0:p], add_intercept):
                return last_model
        else:
            if model_var > var_cutoff:
                return last_model

        last_model = model

    # Unless a simpler model was better, return coefficients for the requested
    #   polynomial degree
    return model


def climatology365(series, dates, ignore_leap = True):
    '''
    Computes a 365-day climatology for different locations from a time series
    of length T. The climatology could then be indexed using ordinals
    generated by `ordinals365()`. Setting `ignore_leap = False` may be useful
    if the time series has regular dropouts; e.g., for MODIS 8-day composite
    data, there are only 46 days each year with valid data.

    Parameters
    ----------
    series : numpy.ndarray
        T x ... array of data
    dates : list or tuple
        Sequence of datetime.datetime or datetime.date instances
    ignore_leap : bool
        True to convert DOY to (DOY-1) in leap years, effectively ignoring
        Leap Day (Default); if False, DOY numbers are unchanged

    Returns
    -------
    numpy.ndarray
    '''
    @suppress_warnings
    def calc_climatology(x):
        return np.array([
            np.nanmean(x[ordinal == day,...], axis = 0)
            for day in range(1, 366)
        ])
    # Get first and last day of the year (DOY)
    ordinal = np.array([
        # Finally, subtract 1 from each day in a leap year after Leap Day
        (doy - 1) if (
            ignore_leap and (dates[i].year % 4 == 0) and doy >= 60) else doy
        for i, doy in enumerate([
            # Next, fill in 0 wherever Leap Day occurs
            0 if (dates[i].year % 4 == 0 and doy == 60) else doy
            for i, doy in enumerate([
                # First, convert datetime.datetime to ordinal day-of-year (DOY)
                int(dt.strftime('%j')) for dt in dates
            ])
        ])
    ])
    return calc_climatology(series)


def daynight_partition(arr_24hr, updown, reducer = 'mean'):
    '''
    Partitions a 24-hour time series array into daytime and nighttime values,
    then calculates the mean in each group. Daytime is defined as when the sun
    is above the horizon; nighttime is the complement.

    Parameters
    ----------
    arr_24hr : numpy.ndarray
        A size (24 x ...) array; the first axis must have 24 elements
        corresponding to the measurement in each hour
    updown: numpy.ndarray
        A size (2 x ...) array, compatible with arr_24hr, where the first axis
        has the hour of sunrise and sunset, in that order, for each element
    reducer : str
        One of "mean" or "sum" indicating whether an average or a total of the
        daytime/ nighttime values should be calculated; e.g., for "mean", the
        hourly values from daytime hours are added up and divided by the
        length of the day (in hours).

    Returns
    -------
    numpy.ndarray
        A size (2 x ...) array where the first axis enumerates the daytime and
        nighttime mean values, respectively
    '''
    assert reducer in ('mean', 'sum'),\
        'Argument "reducer" must be one of: "mean", "sum"'
    # Prepare single-valued output array
    arr_daytime = np.zeros(arr_24hr.shape[1:])
    arr_nighttime = arr_daytime.copy()
    daylight_hrs = arr_daytime.copy().astype(np.int16)
    # Do sunrise and sunset define an interval? (Sunset > Sunrise)?
    inside_interval = np.apply_along_axis(lambda x: x[1] > x[0], 0, updown)
    # Or is the sun never up?
    never_up = np.logical_and(updown[0,...] == -1, updown[1,...] == -1)
    # Iteratively sum daytime VPD and temperature values
    for hr in range(0, 24):
        # Given only hour of sunrise/set on a 24-hour clock...
        #   if sun rises and sets on same day: SUNRISE <= HOUR <= SUNSET;
        #   if sun sets on next day: either SUNRISE <= HOUR or HOUR <= SUNSET;
        sun_is_up = np.logical_or( # Either...
            np.logical_and(inside_interval, # ...Rises and sets same day
                np.logical_and(updown[0,...] <= hr, hr <= updown[1,...])),
            np.logical_and(~inside_interval, # ...Sets on next day
                np.logical_or(updown[0,...] <= hr, hr <= updown[1,...])))
        # For simplicity, compute a 24-hour mean even if the sun never rises;
        #   there's no way to know what the "correct" daytime value is
        mask = np.logical_or(never_up, sun_is_up)
        np.add(np.where(
            mask, arr_24hr[hr,...], 0), arr_daytime, out = arr_daytime)
        np.add(np.where(
            ~mask, arr_24hr[hr,...], 0), arr_nighttime, out = arr_nighttime)
        # Keep track of the denominator (hours) for calculating the mean;
        #   note that this over-estimates actual daylight hours by 1 hour
        #   but results in the correct denominator for the sums above
        np.add(np.where(mask, 1, 0), daylight_hrs, out = daylight_hrs)
    arr_24hr = None
    # Calculate mean quantities
    if reducer == 'mean':
        arr_daytime = np.divide(arr_daytime, daylight_hrs)
        arr_nighttime = np.divide(arr_nighttime, 24 - daylight_hrs)
        # For sites where the sun is always above/ below the horizon, set missing
        #   nighttime values to zero
        arr_nighttime[~np.isfinite(arr_nighttime)] = 0
    return np.stack((arr_daytime, arr_nighttime))


def degree_lengths(phi, a = 6378137, b = 6356752.3142):
    '''
    Returns the approximate length of degrees of latitude and longitude.
    Source:

        https://en.wikipedia.org/wiki/Latitude

    phi : Number
        Latitude, in degrees
    a : Number
        Radius of the Earth (major axis) in meters
    b : Number
        Length of minor axis of the Earth in meters

    Returns
    -------
    tuple
        Length of a degree of (longitude, latitude), respectively
    '''
    e2 = ((a**2) - (b**2)) / (a**2)
    # Approximate length of a degree of latitude
    lat_m = 111132.954 - (559.822 * np.cos(2 * np.deg2rad(phi))) +\
        (1.175 * np.cos(4 * np.deg2rad(phi)))
    lng_m = (np.pi * a * np.cos(np.deg2rad(phi))) / (
        180 * np.sqrt(1 - (e2 * np.sin(np.deg2rad(phi))**2)))
    return (lng_m, lat_m)


def e_mult(params, tmin, vpd, smrz, ft):
    '''
    Calculate environmental constraint multiplier for gross primary
    productivity (GPP), E_mult, based on current model parameters. The
    expected parameter names are "LUE" for the maximum light-use
    efficiency; "smrz0" and "smrz1" for the lower and upper bounds on root-
    zone soil moisture; "vpd0" and "vpd1" for the lower and upper bounds on
    vapor pressure deficity (VPD); "tmin0" and "tmin1" for the lower and
    upper bounds on minimum temperature; and "ft0" for the multiplier during
    frozen ground conditions.

    Parameters
    ----------
    params : dict
        A dict-like data structure with named model parameters
    tmin : numpy.ndarray
        (T x N) vector of minimum air temperature (deg K), where T is the
        number of time steps, N the number of sites
    vpd : numpy.ndarray
        (T x N) vector of vapor pressure deficit (Pa), where T is the number
        of time steps, N the number of sites
    smrz : numpy.ndarray
        (T x N) vector of root-zone soil moisture wetness (%), where T is the
        number of time steps, N the number of sites
    ft : numpy.ndarray
        (T x N) vector of the (binary) freeze-thaw status, where T is the
        number of time steps, N the number of sites (Frozen = 0, Thawed = 1)

    Returns
    -------
    numpy.ndarray
    '''
    # Calculate E_mult based on current parameters
    f_tmin = linear_constraint(params['tmin0'], params['tmin1'])
    f_vpd  = linear_constraint(params['vpd0'], params['vpd1'], 'reversed')
    f_smrz = linear_constraint(params['smrz0'], params['smrz1'])
    f_ft   = linear_constraint(params['ft0'], 1.0, 'binary')
    return f_tmin(tmin) * f_vpd(vpd) * f_smrz(smrz) * f_ft(ft)


def k_mult(params, tsoil, smsf):
    '''
    Calculate environmental constraint multiplier for soil heterotrophic
    respiration (RH), K_mult, based on current model parameters. The expected
    parameter names are "tsoil" for the Arrhenius function of soil temperature
    and "smsf0" and "smsf1" for the lower and upper bounds of the ramp
    function on surface soil moisture.

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

    Returns
    -------
    numpy.ndarray
    '''
    f_tsoil = partial(arrhenius, beta0 = params['tsoil'])
    f_smsf  = linear_constraint(params['smsf0'], params['smsf1'])
    return f_tsoil(tsoil) * f_smsf(smsf)


def litterfall_casa(lai, years, dt = 1/365):
    '''
    Calculates daily litterfall fraction after the CASA model (Randerson et
    al. 1996). Computes the fraction of evergreen versus deciduous canopy and
    allocates a constant daily fraction (out of the year) for evergreen canopy
    but a varying daily fraction for deciduous, where the fraction varies with
    "leaf loss," a function of leaf area index (LAI). Canopies are assumed to
    be a mix of evergreen and deciduous, so the litterfall fraction is a sum
    of these two approaches.

    Randerson, J. T., Thompson, M. V, Malmstrom, C. M., Field, C. B., &
      Fung, I. Y. (1996). Substrate limitations for heterotrophs: Implications
      for models that estimate the seasonal cycle of atmospheric CO2.
      *Global Biogeochemical Cycles,* 10(4), 585–602.

    The approach here is a bit different from Randerson et al. (1996) because
    we re- calculate the evergreen fraction each year; however, this is a
    reasonable elaboration that, incidentally, accounts for potential changes
    in the evergreen-vs-deciduous mix of the canopy. The result is an array
    of daily litterfall fractions, i.e., the result multiplied by the annual
    NPP sum (for a given site and year) obtains the daily litterfall.

    Parameters
    ----------
    lai : numpy.ndarray
        The (T x N) leaf-area index (LAI) array, for T time steps and N sites
    years : numpy.ndarray
        A length-T 1D array indexing the years, e.g., [2001, 2001, 2001, ...];
        used to identify which of T time steps belong to a year, so that
        litterfall fractions sum to one over a year
    dt : float
        The fraction of a year that each time step represents, e.g., for daily
        time steps, should be close to 1/365 (Default: 1/365)

    Returns
    -------
    numpy.ndarray
        The fraction of available inputs (e.g., annual NPP) that should be
        allocated to litterfall at each time step
    '''
    def leaf_loss(lai):
        # Leaf loss function from CASA, a triangular averaging function
        #   centered on the current date, where the right limb of the
        #   triangle is subtracted from the left limb (leading minus
        #   lagged LAI is equated to leaf loss)
        ll = generic_filter(
            lai, lambda x: (0.5 * x[0] + x[1]) - (x[3] + 0.5 * x[4]),
            size = 5, mode = 'mirror')
        return np.where(ll < 0, 0, ll) # Leaf loss cannot be < 0

    # Get leaf loss at each site (column-wise)
    ll = np.apply_along_axis(leaf_loss, 0, lai)
    ll = np.where(np.isnan(ll), 0, ll) # Fill NaNs with zero leaf loss
    unique_years = np.unique(years).tolist()
    unique_years.sort()
    for each_year in unique_years:
        # For those dates in this year...
        idx = years == each_year
        # Calculate the evergreen fraction (ratio of min LAI to mean LAI over
        #   the course of a year)
        efrac = np.apply_along_axis(
            lambda x: np.nanmin(x) / np.nanmean(x), 0, lai[idx,:])
        # Calculate sum of 1/AnnualNPP (Evergreen input) plus daily leaf loss
        #   fraction (Deciduous input); Evergreen canopies have constant daily
        #   inputs
        ll[idx,:] = (efrac * dt) + (1 - efrac) * np.divide(
            ll[idx,:], ll[idx,:].sum(axis = 0))
    return ll


def mean_residence_time(
        hdf, units = 'years', subset_id = None, nodata = -9999):
    '''
    Calculates the mean residence time (MRT) of soil organic carbon (SOC)
    pools as the quotient of SOC stock size and heterotrophic respiration
    (RH). Chen et al. (2013, Global and Planetary Change), provide a formal
    equation for mean residence time: (SOC/R_H).

    Parameters
    ----------
    hdf : h5py.File
        The HDF5 file / h5py.File object
    units : str
        Either "years" (default) or "days"
    subset_id : str
        (Optional) Can provide keyword designating the desired subset area
    nodata : float
        (Optional) The NoData or Fill value (Default: -9999)

    Returns
    -------
    tuple
        Tuple of: subset array, xoff, yoff, i.e., (numpy.ndarray, Int, Int)
    '''
    assert units in ('days', 'years'), 'The units argument must be one of: "days" or "years"'
    soc_field = HDF_PATHS['SPL4CMDL']['4']['SOC']
    rh_field = HDF_PATHS['SPL4CMDL']['4']['RH']
    if subset_id is not None:
        # Get X- and Y-offsets while we're at it
        soc, xoff, yoff = subset(
            hdf, soc_path, None, None, subset_id = subset_id)
        rh, _, _ = subset(
            hdf, rh_path, None, None, subset_id = subset_id)
    else:
        xoff = yoff = 0
        soc = hdf[soc_path][:]
        rh = hdf[rh_path][:]

    # Find those areas of NoData in either array
    mask = np.logical_or(soc == nodata, rh == nodata)
    mrt = np.divide(soc, rh)
    if units == 'years':
        # NOTE: No need to guard against NaNs/ NoData here because of mask
        mrt = np.divide(mrt, 365.0)
    np.place(mrt, mask, nodata) # Put NoData values back in
    return (mrt, xoff, yoff)


def ordinals365(dates):
    '''
    Returns a length-T sequence of ordinals on [1,365]. Can be used for
    indexing a 365-day climatology; see `climatology365()`.

    Parameters
    ----------
    dates : list or tuple
        Sequence of datetime.datetime or datetime.date instances

    Returns
    -------
    list
    '''
    return [
        t - 1 if (year % 4 == 0 and t >= 60) else t
        for t, year in [(int(t.strftime('%j')), t.year) for t in dates]
    ]


def rescale_smrz(smrz0, smrz_min, smrz_max = 100):
    '''
    Rescales root-zone soil-moisture (SMRZ); original SMRZ is in percent
    saturation units. NOTE: Although Jones et al. (2017) write "SMRZ_wp is
    the plant wilting point moisture level determined by ancillary soil
    texture data provided by L4SM..." in actuality it is just `smrz_min`.

    Parameters
    ----------
    smrz0 : numpy.ndarray
        (T x N) array of original SMRZ data, in percent (%) saturation units
        for N sites and T time steps
    smrz_min : numpy.ndarray or float
        Site-level long-term minimum SMRZ (percent saturation)
    smrz_max : numpy.ndarray or float
        Site-level long-term maximum SMRZ (percent saturation); can optionally
        provide a fixed upper-limit on SMRZ; useful for calculating SMRZ100.

    Returns
    -------
    numpy.ndarray
    '''
    if smrz_min.ndim == 1:
        smrz_min = smrz_min[np.newaxis,:]
    assert smrz0.ndim == 2,\
        'Expected smrz0 to be a 2D array'
    assert smrz0.shape[1] == smrz_min.shape[1],\
        'smrz_min should have one value per site'
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


def soc_analytical_spinup(litterfall, k_mult, fmet, fstr, decay_rates):
    r'''
    Using the solution to the differential equations governing change in the
    soil organic carbon (SOC) pools, calculates the steady-state size of each
    SOC pool.

    The analytical steady-state value for the metabolic ("fast") pool is:
    $$
    C_{met} = \frac{f_{met} \sum NPP}{R_{opt} \sum K_{mult}}
    $$

    The analytical steady-state value for the structural ("medium") pool is:
    $$
    C_{str} = \frac{(1 - f_{met})\sum NPP}{R_{opt}\, k_{str} \sum K_{mult}}
    $$

    The analytical steady-state value for the recalcitrant ("slow") pool is:
    $$
    C_{rec} = \frac{f_{str}\, k_{str}\, C_{str}}{k_{rec}}
    $$

    Parameters
    ----------
    litterfall : numpy.ndarray
        Average daily litterfall, a (N x ...) array
    k_mult : numpy.ndarray
        The K_mult climatology, i.e., a (365 x N x ...) array of the long-term
        average K_mult value at each of N sites (optionally, with 81 1-km
        subgrid sites, e.g., 365 x N x 81)
    fmet : numpy.ndarray
        The f_metabolic model parameter, as an (N x ...) array
    fstr : numpy.ndarray
        The f_structural model parameter, as an (N x ...) array
    decay_rates : numpy.ndarray
        The optimal decay rates for each SOC pool, as a (3 x N x ...) array

    NOTE: If a 3 or more axes are used, those axes must match for all arrays;
    i.e., if (x ...) is used, it must be the same for all.

    Returns
    -------
    tuple
        A 3-element tuple, each element the steady-state values for that pool,
        i.e., `(metabolic, structural, recalcitrant)`
    '''
    # NOTE: litterfall is average daily litterfall (see upstream where we
    #   divided by 365), so, to obtain annual sum, multiply by 365
    c0 = np.divide(
        fmet * (litterfall * 365),
        decay_rates[0,...] * np.sum(k_mult, axis = 0))
    c1 = np.divide(
        (1 - fmet) * (litterfall * 365),
        decay_rates[1,...] * np.sum(k_mult, axis = 0))
    # NOTE: k_mult disappears because it is in both the numerator and
    #   denominator
    c2 = np.divide(fstr * decay_rates[1,...] * c1, decay_rates[2,...])
    c0[~np.isfinite(c0)] = 0
    c1[~np.isfinite(c1)] = 0
    c2[~np.isfinite(c2)] = 0
    return (c0, c1, c2)


def soc_numerical_spinup(
        soc, litterfall, k_mult, fmet, fstr, decay_rates, threshold = 0.1,
        verbose = False):
    '''
    Numerical spin-up of C pools.

    Parameters
    ----------
    soc : numpy.ndarray
        SOC in each pool in g C m-3 units, a (3 x N x ...) array
    litterfall : numpy.ndarray
        Daily litterfall in g C m-2 units, a (N x ...) array
    k_mult : numpy.ndarray
        The K_mult climatology, i.e., a (365 x N x ...) array of the long-term
        average K_mult value at each of N sites (with ... 1-km subgrid sites)
    fmet : numpy.ndarray
        The f_metabolic model parameter, as an (N x ...) array
    fstr : numpy.ndarray
        The f_structural model parameter, as an (N x ...) array
    decay_rates : numpy.ndarray
        The optimal decay rates for each SOC pool, as a (3 x N x ...) array
    threshold : float
        Goal for the NEE tolerance check; i.e., change in NEE between
        climatological years should be less than or equal to the threshold
        for all pixels (Default: 0.1 g C m-2 yr-1)
    verbose : bool
        True to print messages to the screen

    Returns
    -------
    tuple
        2-element tuple of `((soc0, soc1, soc2), tol)` where the first
        element is a 3-tuple of the SOC in each pool; second element is the
        final tolerance
    '''
    tsize = k_mult.shape[0] # Whether 365 (days) or T days
    tol = np.inf
    i = 0
    # Jones et al. (2017) write that goal is NEE tolerance <=
    #   1 g C m-2 year-1, but we can do better
    if verbose:
        print('Iterating...')
    while not np.all(abs(tol) <= threshold):
        diffs = np.zeros(k_mult.shape)
        for t in range(0, tsize):
            rh = k_mult[t,...] * decay_rates * soc
            # Calculate change in C pools (g C m-2 units)
            dc0 = np.subtract(np.multiply(litterfall, fmet), rh[0])
            dc1 = np.subtract(np.multiply(litterfall, 1 - fmet), rh[1])
            dc2 = np.subtract(np.multiply(fstr, rh[1]), rh[2])
            soc[0] += dc0
            soc[1] += dc1
            soc[2] += dc2
            diffs[t,:] += (dc0 + dc1 + dc2)
        # Calculate total annual change in NEE ("mean tolerance") at each
        #   site over the year
        if i > 0:
            # Tolerance goes to zero as each successive year brings fewer
            #   changes in NEE
            tol = last_year - np.nansum(diffs, axis = 0)
        last_year = np.nansum(diffs, axis = 0)
        tol = np.where(np.isnan(tol), 0, tol)
        # Calculate mean absolute tolerance across sites
        if verbose:
            print('[%d] Mean (Max) abs. tolerance: %.4f (%.4f)' % (
                i, np.abs(tol).mean(), np.abs(tol).max()))
        i += 1
    return ((soc[0], soc[1], soc[2]), tol)


def soc_numerical_spinup2(
        soc, litterfall, k_mult, fmet, fstr, decay_rates, cue,
        threshold = 0.1, verbose = False):
    '''
    Numerical spin-up of C pools; here, the "tolerance" of spin-up is equal
    the annual NEE sum.

    Parameters
    ----------
    soc : numpy.ndarray
        SOC in each pool in g C m-3 units, a (3 x N x ...) array
    litterfall : numpy.ndarray
        Daily litterfall in g C m-2 units, a (N x ...) array
    k_mult : numpy.ndarray
        The K_mult climatology, i.e., a (365 x N x ...) array of the long-term
        average K_mult value at each of N sites (with ... 1-km subgrid sites)
    fmet : numpy.ndarray
        The f_metabolic model parameter, as an (N x ...) array
    fstr : numpy.ndarray
        The f_structural model parameter, as an (N x ...) array
    decay_rates : numpy.ndarray
        The optimal decay rates for each SOC pool, as a (3 x N x ...) array
    cue : numpy.ndarray
        The carbon use efficiency (CUE), a (N x ...) array
    threshold : float
        Goal for the NEE tolerance check; i.e., change in NEE between
        climatological years should be less than or equal to the threshold
        for all pixels (Default: 0.1 g C m-2 yr-1)
    verbose : bool
        True to print messages to the screen

    Returns
    -------
    tuple
        2-element tuple of `((soc0, soc1, soc2), tol)` where the first
        element is a 3-tuple of the SOC in each pool; second element is the
        final tolerance
    '''
    tsize = k_mult.shape[0] # Whether 365 (days) or T days
    tol = np.inf
    i = 0
    # Jones et al. (2017) write that goal is NEE tolerance <=
    #   1 g C m-2 year-1, but we can do better
    if verbose:
        print('Iterating...')
    while not np.all(abs(tol) <= threshold):
        nee = np.zeros(k_mult.shape)
        for t in range(0, tsize):
            rh = k_mult[t] * decay_rates[0] * soc
            # Calculate change in C pools (g C m-2 units)
            dc0 = np.subtract(np.multiply(litterfall, fmet), rh[0])
            dc1 = np.subtract(np.multiply(litterfall, 1 - fmet), rh[1])
            dc2 = np.subtract(np.multiply(fstr, rh[1]), rh[2])
            soc[0] += dc0
            soc[1] += dc1
            soc[2] += dc2
            # Adjust structural RH pool for material transferred to recalcitrant
            rh[1] = rh[2] * (1 - fstr)
            # Compute (mean daily) GPP as the (mean daily NPP):CUE ratio, then
            #   compute RA as (GPP - NPP)
            gpp = litterfall / cue
            # While it looks like we can optimize above+below, we'll need to
            #   re-use "gpp" later to calculate NEE ("diffs")
            ra = gpp - litterfall
            nee[t] = (ra + rh.sum(axis = 0)) - gpp
        if i > 0:
            # Tolerance goes to zero as each successive year brings fewer
            #   changes in NEE
            tol = last_year - np.nansum(nee, axis = 0)
        last_year = np.nansum(nee, axis = 0)
        tol = np.where(np.isnan(tol), 0, tol)
        # Calculate mean absolute tolerance across sites
        if verbose:
            print('[%d] Mean (Max) abs. tolerance: %.4f (%.4f)' % (
                i, np.abs(tol).mean(), np.abs(tol).max()))
        i += 1
    return ((soc[0], soc[1], soc[2]), tol)


def tridiag_solver(tri, r, kl = 1, ku = 1, banded = None):
    '''
    Solution to the tridiagonal equation by solving the system of equations
    in sparse form. Creates a banded matrix consisting of the diagonals,
    starting with the lowest diagonal and moving up, e.g., for matrix:

        A = [[10.,  2.,  0.,  0.],
             [ 3., 10.,  4.,  0.],
             [ 0.,  1.,  7.,  5.],
             [ 0.,  0.,  3.,  4.]]
        banded = [[ 3.,  1.,  3.,  0.],
                  [10., 10.,  7.,  4.],
                  [ 0.,  2.,  4.,  5.]]

    The banded matrix is what should be provided to the optoinal "banded"
    argument, which should be used if the banded matrix can be created faster
    than `scipy.sparse.dia_matrix()`.

    Parameters
    ----------
    tri : numpy.ndarray
        A tridiagonal matrix (N x N)
    r : numpy.ndarray
        Vector of solutions to the system, Ax = r, where A is the tridiagonal
        matrix
    kl : int
        Lower bandwidth (number of lower diagonals) (Default: 1)
    ku : int
        Upper bandwidth (number of upper diagonals) (Default: 1)
    banded : numpy.ndarray
        (Optional) Provide the banded matrix with diagonals along the rows;
        this can be faster than scipy.sparse.dia_matrix()

    Returns
    -------
    numpy.ndarray
    '''
    assert tri.ndim == 2 and (tri.shape[0] == tri.shape[1]),\
        'Only supports 2-dimensional square matrices'
    if banded is None:
        banded = dia_matrix(tri).data
    # If it is necessary, in a future implementation, to extract diagonals;
    #   this is a starting point for problems where kl = ku = 1
    # n = tri.shape[0]
    # a, b, c = [ # (n-1, n, n-1) refer to the lengths of each vector
    #     sparse[(i+1),(max(0,i)):j]
    #     for i, j in zip(range(-1, 2), (n-1, n, n+1))
    # ]
    return solve_banded((kl, ku), np.flipud(banded), r)


def vpd(qv2m, ps, temp_k):
	r'''
    Calculates vapor pressure deficit (VPD); unfortunately, the provenance
    of this formula cannot be properly attributed. It is taken from the
    SMAP L4C Science code base, so it is exactly how L4C calculates VPD.

    $$
    \mathrm{VPD} = 610.7 \times \mathrm{exp}\left(
    \frac{17.38 \times T_C}{239 + T_C}
    \right) - \frac{(P \times [\mathrm{QV2M}]}{0.622 + (0.378 \times [\mathrm{QV2M}])}
    $$

    Where P is the surface pressure (Pa), QV2M is the water vapor mixing
    ratio at 2-meter height, and T is the temperature in degrees C (though
    this function requires units of Kelvin when called).

    NOTE: A variation on this formula can be found in the text:

    Monteith, J. L. and M. H. Unsworth. 1990.
    Principles of Environmental Physics, 2nd. Ed. Edward Arnold Publisher.

    See also:
        https://glossary.ametsoc.org/wiki/Mixing_ratio

    Parameters
    ----------
    qv2m : numpy.ndarray or float
        QV2M, the water vapor mixing ratio at 2-m height
    ps : numpy.ndarray or float
        The surface pressure, in Pascals
    temp_k : numpy.ndarray or float
        The temperature at 2-m height in degrees Kelvin

    Returns
    -------
    numpy.ndarray or float
        VPD in Pascals
    '''
	temp_c = temp_k - 273.15 # Convert temperature to degrees C
	avp = np.divide(np.multiply(qv2m, ps), 0.622 + (0.378 * qv2m))
	x = np.divide(17.38 * temp_c, (239 + temp_c))
	esat = 610.7 * np.exp(x)
	return np.subtract(esat, avp)
