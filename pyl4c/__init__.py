'''
This is a collection of Python tools for managing, analyzing, and visualizing
data from the NASA Soil Moisture Active Passive (SMAP) Level 4 Carbon (L4C)
mission. In particular:

- Working with data in EASE-Grid 2.0 projection (`pyl4c.ease2`;)
- Converting HDF5 geophysical variables to GeoTIFF format (`pyl4c.spatial`);
- Creating statistical summaries of SMAP L4C variables or other raster arrays (`pyl4c.utils`);
- Reproducing L4C operational model logic (`pyl4c.science`);
- Down-scaling L4C flux and SOC state variables (`pyl4c.apps.resample`);
- Calibrating the L4C model (`pyl4c.apps.calibration`);
- Running the L4C model (`pyl4c.apps.l4c`);
- Aligning and summarizing SMAP L4C variables with TransCom regions (`pyl4c.lib.transcom`);

**Key things to note:**

- File paths for your system may need to be updated in order to access
    essential ancillary datasets; see `pyl4c.data.fixtures`.
'''

__pdoc__ = {}
__pdoc__['tests'] = False

from collections import Counter
import warnings
import numpy as np

class Namespace(object):
    '''
    Dummy class for holding attributes.
    '''
    def __init__(self):
        pass

    def add(self, label, value):
        '''
        Adds a new attribute to the Namespace instance.

        Parameters
        ----------
        label : str
            The name of the attribute
        value : None
            Any kind of value to be stored
        '''
        setattr(self, label, value)


def equal_or_nan(x, y):
    '''
    A variation on numpy.equal() that also returns True where respective
    inputs are NaN.

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    '''
    return np.where(
        np.logical_or(np.isnan(x), np.isnan(y)), True, np.equal(x, y))


def haversine(p1, p2, radius = 6371e3):
    '''
    Haversine formula for great circle distance, in meters. Accurate for
    points separated near and far but for small distances the accuracy is
    improved by providing a different radius of the sphere, say 6356.7523 km
    for polar regions or 6378.1370 km for equatorial regions. Default is the
    mean earth radius.

    NOTE: Distance returned is in the same units as radius.

    Parameters
    ----------
    p1 : tuple or list
        Sequence of two floats, longitude and latitude, respectively
    p2 : tuple or list
        Same as p1 but for the second point
    radius : int or float
        Radius of the sphere to use in distance calculation
        (Default: 6,371,000 meters)

    Returns
    -------
    float
    '''
    x1, y1 = map(np.deg2rad, p1)
    x2, y2 = map(np.deg2rad, p2)
    dphi = np.abs(y2 - y1) # Difference in latitude
    dlambda = np.abs(x2 - x1) # Difference in longitude
    angle = 2 * np.arcsin(np.sqrt(np.add(
        np.power(np.sin(dphi / 2), 2),
        np.cos(y1) * np.cos(y2) * np.power(np.sin(dlambda / 2), 2)
    )))
    return float(angle * radius)


def pft_dominant(pft_map_array):
    '''
    Returns the PFT class dominant among a (1-km) subarray; for example,
    for each 9-km EASE-Grid 2.0 pixel, returns the PFT class that is dominant
    among the 1-km sub-array pixels.

    Parameters
    ----------
    pft_map_array : numpy.ndarray
        An M x N array specifying the PFT code for every pixel

    Returns
    -------
    list
        An M-element list of the dominant PFT among N pixels
    '''
    return [ # Skip invalid PFT codes
        list(filter(lambda x: x[0] in range(1, 9), count))[0][0]
        for count in [ # Count 1-km cells by PFT
            a.most_common() for a in np.apply_along_axis(
                lambda x: Counter(x.tolist()), 1, pft_map_array)
        ]
    ]


def pft_selector(pft_map, pft):
    '''
    For a given PFT class, returns the tower sites, as rank indices, that
    represent that PFT. Exceptions are made according to the L4C calibration
    protocol, e.g., sites with any amount of Deciduous Needleleaf (DNF) in
    their 1-km subgrid are considered to represent the DNF PFT class.

    Parameters
    ----------
    pft_map : numpy.ndarray
        An (N x M) array where N is the number of sites and M is the number
        of replicates within each site; e.g., for SMAP L4C, M=81 corresponding
        with the 81 cells of the 1-km subgrid for each eddy covariance flux
        tower site.
    pft : int
        The integer number of the PFT to select

    Returns
    -------
    numpy.ndarray
    '''
    idx = pft_map.shape[0]
    if pft == 3:
        return np.apply_along_axis(lambda x: x == 3, 1, pft_map).any(axis = 1)
    return np.equal(pft, [ # Skip invalid PFT codes
        count[1][0] if count[0][0] not in range(1, 9) else count[0][0]
        for count in [ # Count 1-km cells by PFT
            a.most_common() for a in np.apply_along_axis(
                lambda x: Counter(x.tolist()), 1, pft_map)
        ]
    ])


def suppress_warnings(func):
    'Decorator to suppress NumPy warnings'
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return inner
