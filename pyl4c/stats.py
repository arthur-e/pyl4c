'''
Various statistical functions. Note that many have an `add_intercept` argument
and that this is True by default, which means the X matrix will have a column
of ones added for the calculation of the intercept. If you are already using
a design/ model matrix with an intercept term, be sure to set
`add_intercept = False`.
'''

from collections import Counter
import math
import numpy as np

summary_statistics = {
    'boxplot': {
        'nanmin': np.nanmin,
        'nanmax': np.nanmax,
        'nanmedian': np.nanmedian,
        'q1': lambda x: np.percentile(x[~np.isnan(x)], 25),
        'q2': lambda x: np.percentile(x[~np.isnan(x)], 75)
    }
}

def detrend(y, x = None, fill = False):
    '''
    Detrends a 1D series (linearly). Equivalently, returns the residuals
    of an OLS regression where the (transpose of the) design matrix is:

        1 1 1 1 ... 1
        0 1 2 3 ... N

    This removes the linear (straight line) component, e.g., of a time
    series with equal-size time steps.

    Parameters
    ----------
    y : numpy.ndarray
        The 1D series to detrend
    x : numpy.ndarray
        (Optional) The linear series describing the trend; usually a series
        of consecutive integers, e.g.: 1, 2, 3, ...
    fill : bool
        (Optional) True to fill NaNs with mean value (Default: False)

    Returns
    -------
    numpy.ndarray
        The detrended y values
    '''
    assert y.ndim == 1, 'Series to detrend must be a 1D vector'
    n, m = (y.size, 1)
    if x is not None:
        if x.ndim == 2:
            n, m = x.shape

    # Return all NaNs if the input array is all NaNs
    if np.all(np.isnan(y)):
        return np.repeat(np.nan, n, axis = 0)

    # Optionally, fill NaNs with the mean
    if fill:
        y = y.copy()
        nan_mask = np.isnan(y)
        y[np.isnan(y)] = np.nanmean(y)

    x = np.arange(0, n) if x is None else x
    beta = ols(x, y, add_intercept = True)
    yhat = np.hstack(
        (np.ones((n,)).reshape((n, 1)), x.reshape((n, m)))) @ beta
    if fill:
        return np.where(nan_mask, np.nan, np.subtract(y, yhat))
    return np.subtract(y, yhat)


def entropy(seq, base = 2):
    '''
    Returns the Shannon entropy of an input sequence of elements. Default
    units are bits but other units can be returned by changing the base.
    All calculations are with base-2 logarithms; change in base is done
    through multiplying by the constant factor `log_b(a)`` to change from
    base `a` to base `b`. Adapted from [1].

    NOTE: An estimate of the upper limit or "optimal" entropy for base `b`
    with `N` possible symbols can be obtained:

            math.log(n, b)

    1. http://rosettacode.org/wiki/Entropy#Python

    Parameters
    ----------
    seq : list or tuple or str
        Sequence of elements over which entropy is calculated
    base : int or float
        Base of the output units, e.g., 2 for "bits," e (2.718...) for "nats,"
        and 10 for "bans" (Default: 2)

    Returns
    -------
    float
    '''
    # Trivial case, all symbols are the same
    if np.all(np.equal(seq[0], seq)):
        return 0.0 # Avoid a "-0.0" return
    p, lns = Counter(seq), float(len(seq))
    e = -sum( count / lns * np.log2(count / lns) for count in p.values())
    return e if base == 2 else e * math.log(2, base)


def harmonic_ols(x, y, period = 12):
    r'''
    Returns OLS estimates for harmonic series of X, i.e., the matrix `A` of
    the equation `y = Ax + b` is a linear combination of sines and cosines.
    $$
    y_{i,t} = \alpha_i +
    \beta_{i,1}\, \mathrm{cos}\left(\frac{2\pi}{T}x_{i,t}\right) +
    \beta_{i,2}\,\mathrm{sin}\left(\frac{2\pi}{T}x_{i,t}\right) + \varepsilon_{i,t}
    $$

    Parameters
    ----------
    x : numpy.ndarray
        The independent variable, must be 1D
    y : numpy.ndarray
        The dependent variable, must be 1D

    Returns
    -------
    numpy.ndarray
        The solution to the least-squares problem
    '''
    assert x.ndim == 1, 'Array x must be 1D'
    x0 = x.reshape((x.shape[0], 1))
    # Create initial design/ model matrix (without intercept), as this is
    #   augmented by ols()
    a = ((2 * np.pi) / period) # Harmonic coefficient
    xm = np.hstack((x0, np.cos(a * x0), np.sin(a * x0)))
    return ols(xm, y, add_intercept = True)


def linear_constraint(xmin, xmax, form = None):
    '''
    Returns a linear ramp function, for deriving a value on [0, 1] from
    an input value `x`:

        if x >= xmax:
            return 1
        if x <= xmin:
            return 0
        return (x - xmin) / (xmax - xmin)

    Parameters
    ----------
    xmin : int or float
        Lower bound of the linear ramp function
    xmax : int or float
        Upper bound of the linear ramp function
    form : str
        Type of ramp function: "reversed" decreases as x increases;
        "binary" returns xmax when x == 1; default (None) is increasing
        as x increases.

    Returns
    -------
    function
    '''
    assert form == 'binary' or np.any(xmax >= xmin),\
        'xmax must be greater than/ equal to xmin'
    if form == 'reversed':
        return lambda x: np.where(x >= xmax, 0,
            np.where(x < xmin, 1, 1 - np.divide(
                x - xmin, xmax - xmin)))
    if form == 'binary':
        return lambda x: np.where(x == 1, xmax, xmin)
    return lambda x: np.where(x >= xmax, 1,
        np.where(x < xmin, 0, np.divide(x - xmin, xmax - xmin)))


def ols(x, y, add_intercept = True, use_qr_decomp = True):
    '''
    Returns ordinary least squares (OLS) estimates for X. If X is univariate
    (1D), returns the slope of the line between Y and X as well as the
    "y-intercept" or the intersection of this line with the vertical axis.

    Parameters
    ----------
    x : numpy.ndarray
        The independent variable(s); should N x M where M is the number of
        variables
    y : numpy.ndarray
        The dependent variable, must be 1D
    add_intercept : bool
        True to add a y-intercept term (Default: True)
    use_qr_decomp : bool
        True to use QR decomposition to obtain solution (Default: True)

    Returns
    -------
    numpy.ndarray
        The solution to the least-squares problem
    '''
    assert y.ndim == 1, 'Array y must be 1D'
    n = x.shape[0] # Num. of samples
    m = 1 if x.ndim == 1 else x.shape[1] # Num. of covariates
    # Create design/ model matrix
    xm = x.reshape((n, m)) # Without y-intercept term, unless...
    if add_intercept:
        xm = np.hstack((np.ones((n,)).reshape((n, 1)), x.reshape((n, m))))
    if xm.shape[1] > n:
        raise ValueError('System of equations is rank deficient')
    # Generally better to use QR decomposition to obtain \hat{\beta}
    if use_qr_decomp:
        q, r = np.linalg.qr(xm)
        fit = np.linalg.inv(r) @ q.T @ y
    else:
        fit = np.linalg.inv(xm.T @ xm) @ xm.T @ y
    return fit


def ols_variance(x, y, beta = None, add_intercept = True):
    r'''
    Returns the unbiased estimate of OLS model variance:
    $$
    SSE / (n - p)
    $$

    Where:
    $$
    SSE = (y - X\beta )' (y - X\beta )
    $$

    SSE is known as the sum of squared errors of prediction and is equivalent
    to the residual sum of squares (RSS).

    Parameters
    ----------
    x : numpy.ndarray
        The independent variable(s); should be N x M where M is the number of
        variables
    y : numpy.ndarray
        The dependent variable, must be 1D
    beta : numpy.ndarray
        (Optional) Coefficient estimates, an M-dimensional vector
    add_intercept : bool
        True to add a y-intercept term (Default: True)

    Returns
    -------
    float
        The sum-of-squared-errors of prediction
    '''
    n = x.shape[0] # Num. of samples
    p = 1 if x.ndim == 1 else x.shape[1] # Num. of covariates
    p += 1 if add_intercept else 0
    # Alternatively, use xm and beta calculated in sum_of_squares()...
    # return ((y - xm @ beta).T @ (y - xm @ beta)) / (n - p)
    return sum_of_squares(x, y, beta, add_intercept, which = 'sse') / (n - p)


def rmsd(x1, x2, n = None, weights = None):
    r'''
    Returns the root mean-squared deviation (RMSD) between two continuously
    varying random quantities:
    $$
    RMSD(\hat{x}, x) = \sqrt{n^{-1}\sum_i^N (\hat{x}_i - x_i)^2}
    $$

    Where `N` (or `n`) is the number of samples (e.g., model cells or time
    steps).

    NOTE: `NoData` should be filled with `np.nan` prior to calling this
    function; it is assumed that both vectors have the same missingness.

    Parameters
    ----------
    x1 : numpy.ndarray
        A 1D or 2D numeric vector
    x2 : numpy.ndarray
        A 1D or 2D numeric vector
    n : int
        (Optional) The number of samples, for normalizing; if not provided,
        calculated as the number of non-NaN samples
    weights : numpy.ndarray
        Weights array of a shape that can be broadcast to match both x1 and x2

    Returns
    -------
    float
    '''
    assert isinstance(n, int) or n is None, 'Argument "n" must be of integer type'
    assert x1.ndim <= 2 and x2.ndim <= 2, 'No support for more than 2 axes'
    diffs = np.subtract(x1, x2)
    # Optionally weight the residuals first
    if weights is not None:
        if weights.ndim == 1:
            weights = weights.reshape((1, *weights.shape))
        diffs = diffs * weights
    diffs = diffs[~np.isnan(diffs)]
    n = n if n is not None else diffs.shape[0]
    return np.sqrt(np.divide(np.sum(np.power(diffs, 2)), n))


def sum_of_squares(x, y, beta = None, add_intercept = True, which = 'ssr'):
    '''
    Calculates the sum-of-squared errors, sum-of-squared regressors, or
    total sum-of-squared errors based on the observed and predicted responses.
    Will calculate sum-of-squares for univariate linear regression if vector
    "x" is the independent variable and "y" is the dependent variable.

    Parameters
    ----------
    x : numpy.ndarray
        Observed response or independent variable;
    y : numpy.ndarray
        Predicted response or dependent variable;
    beta : numpy.ndarray
        (Optional) To avoid re-fitting (or fitting incorrectly) the OLS
        regression, can provide the coefficients of the OLS regression
    add_intercept : bool
        True to add an intercept term when fitting OLS regression
    which : str
        Which sum-of-squares quantity to calculate (Default: 'ssr');
        should be one of: ('sse', 'ssr', 'sst')

    Returns
    -------
    float
    '''
    assert y.ndim == 1, 'Array y must be 1D'
    if beta is None:
        beta = ols(x, y, add_intercept)
    n = x.shape[0] # Num. of samples
    m = p = 1 if x.ndim == 1 else x.shape[1] # Num. of covariates
    # Create design/ model matrix
    xm = x.reshape((n, m)) # Without y-intercept term, unless...
    if add_intercept:
        p += 1
        xm = np.hstack((np.ones((n,)).reshape((n, 1)), x.reshape((n, m))))

    yhat = xm @ beta # Calculate predicted values
    if which == 'sse':
        # Alternatively: ((y - xm @ beta).T @ (y - xm @ beta))
        return np.sum(np.power(np.subtract(y, yhat), 2))

    elif which == 'ssr':
        return np.sum(np.power(np.subtract(yhat, np.nanmean(y)), 2))

    elif which == 'sst':
        if add_intercept:
            return np.sum(np.power(np.subtract(y, np.nanmean(y)), 2))
        else:
            return np.sum(np.power(y, 2))

    else:
        raise NotImplementedError('Argument "which" not understood')


def t_statistic(x, y, beta_hat, beta = 0, add_intercept = True):
    r'''
    Calculates t-statistics for each of the predictors of beta_hat; by
    default, the true value of beta is assumed to be zero (i.e., null
    hypothesis is true). Calculated as:
    $$
    T_{n-p} = \frac{\hat{\beta }_j - \beta_j}{\mathrm{s.e.}(\hat{\beta }_j)}\sim t_{n-p}
    $$

    Parameters
    ----------
    x : numpy.ndarray
        The independent variable(s); should be N x M where M is the number
        of variables
    y : numpy.ndarray
        The dependent variable, must be 1D
    beta_hat : numpy.ndarray
        The estimated value(s) for beta
    beta : numpy.ndarray
        (Optional) The true value(s) for beta
    add_intercept : bool
        True to add a y-intercept term (Default: True)

    Returns
    -------
    float
    '''
    assert x.ndim <= 2, 'Array x must be 1D or 2D only'
    model_var = ols_variance(x, y, beta_hat, add_intercept = add_intercept) # Var(beta)
    beta_hat_se = np.sqrt(np.diag(np.linalg.inv(x.T @ x) * model_var))
    return np.subtract(beta_hat, beta) / beta_hat_se
