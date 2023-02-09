'''
Functions for affine transformations between geographic coordinates (WGS84)
and EASE-Grid 2.0 row-column coordinates.

NOTE: The `ease2_from_wgs84()` and `ease2_to_wgs84()` functions should refer
to the center of a grid cell, which can be confirmed by:

    >>> ease2_from_wgs84(ease2_to_wgs84((0, 0), 'M09'), 'M01')
    (4, 4)
'''

import numpy as np
from functools import partial
from itertools import product
from pyl4c import haversine
from pyl4c.data.fixtures import EASE2_GRID_PARAMS

EARTH_RADIUS = 6378137.0 # WGS84 sphere
WGS84_ECCENTRICITY = 0.081819190843


def ease2_coords(grid: str, in_1d: bool = True):
    '''
    Returns EASE-Grid 2.0 coordinate arrays; coordinates are WGS84 longitude-
    or latitude.

    Parameters
    ----------
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.
    in_1d : bool
        True to produce 1D arrays (Default: True)

    Returns
    -------
    tuple
        Two elements, the longitude and latitude coordinate arrays:
        (numpy.ndarray, numpy.ndarray)
    '''
    nrows, ncols = EASE2_GRID_PARAMS[grid]['shape']
    rows = np.arange(0, nrows)
    cols = np.arange(0, ncols)
    e2ll = partial(ease2_to_wgs84, grid = grid)
    x_coords = np.array(list(map(lambda c: e2ll((0, c))[0], cols.tolist())))
    y_coords = np.array(list(map(lambda r: e2ll((r, 0))[1], rows.tolist())))
    if in_1d:
        return (x_coords, y_coords)
    return ( # Otherwise, return two 2D arrays with duplicates
        np.repeat(x_coords.reshape((1, ncols)), nrows, axis = 0),
        np.repeat(y_coords.reshape((nrows, 1)), ncols, axis = 1))


def ease2_nested_cells(coords, k: int, grid = 'M01'):
    '''
    Finds the row-column coordinates of all EASE-Grid 2.0 cells nested within
    a larger cell. The finer or target grid resolution is specified by `grid`;
    it is assumed that the WGS84 `coords` provided refer to the center of a
    larger cell.

    Parameters
    ----------
    coords : list or tuple
        The longitude-latitude pair for which to find the nearest grid cells
    k : int
        The search radius; corresponds to the number of rows and
            columns away from the defined coordinates
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.

    Returns
    -------
    list
    '''
    # Translate WGS84 to row-column indices
    row, col = ease2_from_wgs84(coords, grid)
    potential_rows = np.arange(row - k, row + k + 1, 1)
    potential_cols = np.arange(col - k, col + k + 1, 1)
    return list(product(potential_rows, potential_cols))


def ease2_search_radius(coords, k: int, grid = 'M09'):
    '''
    Finds the nearest EASE-Grid 2.0 grid cells using a search radius of k grid
    cells; returns a list of grid cell indices sorted by distance ascending.
    The row-column indices that correspond to the affine transformation of the
    given coordinates will always be the nearest, but this can be useful for
    finding the next-nearest row-column coordinates. See
    `pyl4c.ease2.ease2_nested_cells()`.

    Parameters
    ----------
    coords : list or tuple
        The longitude-latitude pair for which to find the nearest grid cells
    k : int
        The search radius; corresponds to the number of rows and
            columns away from the defined coordinates
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.

    Returns
    -------
    list
    '''
    choices_idx = ease2_nested_cells(coords, k, grid)
    choices_wgs84 = [ # Convert from row-column to longitude-latitude
        ease2_to_wgs84(pair, grid) for pair in choices_idx
    ]
    dists = [ # Calculate every pair-wise distance
        haversine(coords, choices_wgs84[i]) for i in range(0, len(choices_idx))
    ]
    # Return indices sorted by distance ascending
    return np.array(choices_idx)[np.argsort(dists),:].tolist()


def ease2_from_wgs84(coords, grid: str = 'M09', exact: bool = False):
    '''
    Given a longitude-latitude coordinate pair, derive EASE-Grid 2.0
    row-column coordinates. This is translated directly from the gridutil C
    code (easegrid_wgs84_transforms.c) which, in turn, was translated from
    Brodzik's IDL code. NOTE: Only supports the global EASE-Grid 2.0
    projection for now. NOTE: The `ease2_from_wgs84()` and `ease2_to_wgs84()`
    functions are only consistent (i.e., reversible) if `exact=True` is used.

    Parameters
    ----------
    coords : list or tuple
        Sequence of two floats, the longitude and latitude, respectively
    grid : str
        String, the EASE-Grid 2.0 designation: M01, M09, etc.
    exact : bool
        True to return grid cell indices as floating point instead of integer
        type (Default: False)

    Returns
    -------
    tuple
        The (row, column) coordinates corresponding to the given
        latitude-longitude coordinates
    '''
    # Unpack longitude and latitude
    user_lng, user_lat = coords

    # Define projection parameters for WGS84 and the EASE-Grid 2.0 projections
    e2 = np.power(WGS84_ECCENTRICITY, 2)
    epsilon = 1e-6

    # Unique to global EASE-Grid 2.0 projection
    ref_lat = ref_lng = 0.0
    ref_lat2 = 30.0
    sin_phi1 = np.sin(np.deg2rad(ref_lat2))
    cos_phi1 = np.cos(np.deg2rad(ref_lat2))
    kz = cos_phi1 / np.sqrt(1 - e2 * sin_phi1 * sin_phi1)

    # Get grid shape and resolution
    nrows, ncols = EASE2_GRID_PARAMS[grid]['shape']
    resolution = EASE2_GRID_PARAMS[grid]['resolution']
    r0 = (ncols - 1) / 2.0 # Column mapped to reference latitude
    s0 = (nrows - 1) / 2.0 # Row mapped to reference latitude(?)

    # Start transformation of user coordinates
    dlng = user_lng - ref_lng
    assert abs(dlng) <= 180, 'Longitude error'

    # Convert to radians
    phi = np.deg2rad(user_lat)
    lam = np.deg2rad(dlng)

    q = (1.0 - e2) * ((np.sin(phi) / (1.0 - e2 * np.sin(phi) * np.sin(phi))) - (1.0 / (2.0 * WGS84_ECCENTRICITY)) * np.log((1.0 - WGS84_ECCENTRICITY * np.sin(phi)) / (1.0 + WGS84_ECCENTRICITY * np.sin(phi))))
    qp = 1.0 - ((1.0 - e2) / (2.0 * WGS84_ECCENTRICITY) * np.log((1.0 - WGS84_ECCENTRICITY) / (1.0 + WGS84_ECCENTRICITY)))

    # Unique to global EASE-Grid 2.0 projection
    x = EARTH_RADIUS * kz * lam
    y = EARTH_RADIUS * (q / (2.0 * kz))

    result = (s0 - (y / resolution), r0 + (x / resolution))
    if exact:
        return result

    return tuple(map(lambda c: int(round(c)), result))


def ease2_to_wgs84(coords, grid = 'M09'):
    '''
    Given a row-column coordinate pair from an EASE-Grid 2.0, derive the
    corresponding WGS84 longitude-latitude coordinates. This is translated
    directly from the gridutil C code (easegrid_wgs84_transforms.c) which, in
    turn, was translated from Brodzik's IDL code. NOTE: Only supports the
    global EASE-Grid 2.0 projection for now. NOTE: The `ease2_from_wgs84()`
    and `ease2_to_wgs84()` functions are only consistent (i.e., reversible) if
    `exact=True` is used (in `ease2_from_wgs84()`).

    Parameters
    ----------
    coords : list or tuple
        Sequence of two integers, the row and column coordinates
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.

    Returns
    -------
    tuple
        The (longitude, latitude) coordinates corresponding to the given
        row-column coordinates
    '''
    e2 = np.power(WGS84_ECCENTRICITY, 2)
    e4 = np.power(WGS84_ECCENTRICITY, 4)
    e6 = np.power(WGS84_ECCENTRICITY, 6)
    epsilon = 1e-6

    # Unique to global EASE-Grid 2.0 projection
    ref_lat = ref_lng = 0.0
    ref_lat2 = 30.0
    sin_phi1 = np.sin(np.deg2rad(ref_lat2))
    cos_phi1 = np.cos(np.deg2rad(ref_lat2))
    kz = cos_phi1 / np.sqrt(1 - e2 * sin_phi1 * sin_phi1)

    # Convert (row, column) to (x, y) coordinates for this projection
    x, y = translate_row_col_to_ease2(coords, grid)

    qp = (1.0 - e2) * ((1.0 / (1.0 - e2)) - (1.0 / (2.0 * WGS84_ECCENTRICITY)) * np.log((1.0 - WGS84_ECCENTRICITY) / (1.0 + WGS84_ECCENTRICITY)))

    # Unique to global EASE-Grid 2.0 projection
    beta = np.arcsin(2.0 * y * kz / (EARTH_RADIUS * qp))
    lam  = x / (EARTH_RADIUS * kz)

    phi = beta \
        + (((e2 / 3.0) + ((31.0 / 180.0) * e4) \
              + ((517.0 / 5040.0) * e6)) * np.sin(2.0 * beta)) \
        + ((((23.0 / 360.0) * e4) \
              + ((251.0 / 3780.0) * e6)) * np.sin(4.0 * beta)) \
        + (((761.0 / 45360.0) * e6) * np.sin(6.0 * beta))

    return ((180.0 * (lam / np.pi)) + ref_lng, (180.0 * phi / np.pi))


def translate_row_col_to_ease2(coords, grid = 'M09'):
    '''
    Returns the EASE-Grid 2.0 coordinates at the given row-column index.

    Parameters
    ----------
    coords : list or tuple
        Sequence of two integers, the row and column coordinates
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.

    Returns
    -------
    tuple
        The EASE-Grid 2.0 (X, Y) coordinates corresponding to the given
        row-column coordinates
    '''
    # Unpack row, column coordinates
    user_row, user_col = coords

    # Get grid shape and resolution
    nrows, ncols = EASE2_GRID_PARAMS[grid]['shape']
    resolution = EASE2_GRID_PARAMS[grid]['resolution']
    r0 = (ncols - 1) / 2.0 # Column mapped to reference latitude
    s0 = (nrows - 1) / 2.0 # Row mapped to reference latitude(?)

    # Convert (row, column) to (x, y) coordinates for this projection
    x = (user_col - r0) * resolution
    y = (s0 - user_row) * resolution
    return (x, y)
