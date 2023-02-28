'''
Convenience functions for working with SMAP L4C data in HDF5 arrays.
NOTE: All of the functions beginning with `get_` require access to ancillary
data files.
'''

import csv
import glob
import io
import os
import h5py
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pyl4c import haversine
from pyl4c.data.fixtures import ANCILLARY_DATA_PATHS, HDF_PATHS, EASE2_GRID_PARAMS, SUBSETS_BBOX
from pyl4c.ease2 import ease2_coords, ease2_from_wgs84, ease2_search_radius

class MockL4CGranule(h5py.File):
    '''
    A mock for a typical L4C Ops granule. Should have all the right data
    fields at the right sizes for a typical L4C Ops granule read in as
    an h5py.File instance.

    Parameters
    ----------
    file : file
        (Optional) The File object to write; defaults to an in-memory
        `BytesIO` instance
    grid : str
        (Optional) The EASE-Grid 2.0 size: "M01" or "M09"
    mode : str
        (Optional) The file access mode (Default: "w")
    coords : tuple
        (Optional) 2D coordinate arrays to set on the granule's
        "GEO/longitude" and "GEO/latitude" fields
    data : dict
        (Optional) If provided, should be a Dictionary with keys for each
        desired dataset (e.g., "NEE" or "SOC") and any correctly shaped
        2D array as values
    pft_mean_fields : tuple or list
        Sequence of field names (e.g., `['RH', 'GPP']`) for which the PFT mean
        fields (e.g., `RH/rh_pft1_mean`) should be created
    '''
    def __init__(
            self, file = None, grid = 'M09', mode = 'w', coords = None,
            data = None, pft_mean_fields = list()):
        # Although file = io.BytesIO() in the function signature, above,
        #   should accomplish the same thing, it generates an OSError that
        #   cannot be reproduced in interactive mode unless the "file"
        #   is explicitly set here and io.BytesIO() is *not* invoked above
        if file is None:
            file = io.BytesIO()

        super(MockL4CGranule, self).__init__(file, mode)

        # Create a field for each L4C variable
        shp = EASE2_GRID_PARAMS[grid]['shape']
        for field in ('NEE', 'GPP', 'RH', 'SOC'):
            key = '%s/%s_mean' % (field, field.lower()) # e.g., "SOC/soc_mean"
            if data is None:
                init_data = np.ones(shp) * np.nan
            else:
                # Generate a NaN-valued array if no data provided
                init_data = data.get(key, np.ones(shp) * np.nan)

            # Create, e.g., "SOC/soc_mean" field
            self.create_dataset(key, shp, dtype = 'float32', data = init_data)
            # Optionally, create, e.g., "SOC/soc_pft1_mean" field
            if field in pft_mean_fields:
                for pft in range(1, 9):
                    key = '%s/%s_pft%d_mean' % (field, field.lower(), pft)
                    init_data = data.get(key, np.ones(shp) * np.nan)
                    self.create_dataset(
                        key, shp, dtype = 'float32', data = init_data)

        if coords is None:
            x_coords, y_coords = ease2_coords(grid = 'M09', in_1d = False)
        else:
            x_coords, y_coords = coords
            assert x_coords.ndim == 2 and y_coords.ndim == 2, 'Must provide 2D coordinate arrays'
        self.create_dataset('GEO/longitude', shp, dtype = 'float32',
            data = x_coords)
        self.create_dataset('GEO/latitude', shp, dtype = 'float32',
            data = y_coords)


def composite(
        *arrays, reducer = 'mean', target_band = 0, nodata = -9999.0,
        dtype = np.float32, processes = 1):
    '''
    Composites multiple raster arrays in a single band; is extremely fast when
    used with a Process Pool, but does not support normalization and will not
    composite more than one band at a time.

    NOTE: This function does NOT calculate memory use and will NOT guard
    against memory overflow.

    Parameters
    ----------
    arrays : *numpy.ndarray
        The input arrays to composite
    reducer : str
        The name of the reducer function, either: "median", "min", "max",
        "mean", or "std"
    target_band : int
        The index of the band to composite
    nodata : int or float
        The NoData value to ignore in compositing
    dtype : type
        The data type to enforce in the output
    processes : int
        (Optional) Number of processes to use

    Returns
    -------
    numpy.ndarray
    '''
    if reducer not in ('sum', 'median', 'min', 'max', 'mean', 'std'):
        raise ValueError('Invalid reducer name')

    assert all([isinstance(a, np.ndarray) for a in arrays]), 'Every element of "arrays" must be an instance of np.ndarray'
    shp = arrays[0].shape
    assert all(map(lambda x: x == shp, [r.shape for r in arrays])), 'Raster arrays must have the same shape'
    # For single-band arrays...
    if arrays[0].ndim < 3:
        shp = (1, shp[0], shp[1])
        arrays = list(map(lambda r: r.reshape(shp), arrays))

    # Stack the arrays in a continuous, single-band "tapestry" using
    #   vstack(), then cut out the arrays concatenated in this way into
    #   separate bands using reshape()
    b = target_band
    stack = np.vstack(list(map(
        lambda r: r[b,...].reshape((1, shp[1], shp[2])), arrays)))
    # Insert nan into NoData locations
    stack = np.where(stack == nodata, np.nan, stack)
    # Find the reducer function based on its name (should be a "nan" version)
    reducer_func = partial(getattr(np, 'nan%s' % reducer), axis = 0)
    # Avoid traceback masking and memory errors with ProcessPoolExecutor
    #   when only a single process is requested
    if processes == 1:
        all_results = reducer_func(stack)
    else:
        # Get index ranges for each process to work on
        work = partition(stack, num_processes = processes, axis = 1)
        with ProcessPoolExecutor(max_workers = processes) as executor:
            all_results = executor.map(
                reducer_func, [stack[:,i:j] for i, j in work])

    # Stack each reduced band (and reshape to multi-band image)
    result = np.concatenate(list(all_results), axis = 0)\
        .reshape((1, shp[1], shp[2]))
    return np.where(np.isnan(result), nodata, result)


def composite_hdf(hdf_file_glob, field, subset_id = 'CONUS', **kwargs):
    '''
    A convenience wrapper for `composite()`, enabling its use with HDF5 files
    specified by path names. Accepts additional `**kwargs` as keywoard arguments
    to `pyl4c.utils.composite()`.

    NOTE: This function does NOT calculate memory use and will NOT guard
    against memory overflow.

    Parameters
    ----------
    hdf_file_glob : str
        A file path expression for glob.glob()
    field : str
        The hierarchical path to a variable within each HDF5 file
    subset_id : str
        A well-known identifier for a bounding-box subset

    Returns
    -------
    numpy.ndarray
    '''
    assert subset_id in SUBSETS_BBOX.keys()
    assert len(glob.glob(hdf_file_glob)) > 0, 'No files found matching GLOB regular expression'
    arrays = []
    for path in glob.glob(hdf_file_glob):
        with h5py.File(path, 'r') as hdf:
            assert field in hdf.keys(), 'Variable "%s" not found in HDF5 file' % field
            arrays.append(subset(hdf, field, subset_id = subset_id)[0])
    return composite(*arrays, **kwargs)


def get_ease2_coords(grid, in_1d = True):
    '''
    DEPRECATED: Use `pyl4c.ease2.ease2_coords()` instead.
    Returns a tuple of 1D arrays, the X- and Y-coordinates of the desired
    EASE-Grid 2.0, from reading the ancillary data file.

    Parameters
    ----------
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.
    in_1d : bool
        True to return only 1D arrays, avoiding redundant rows/ columns
        (Default)

    Returns
    -------
    tuple
        X- and Y-coordinate arrays: (numpy.ndarray, numpy.ndarray)
    '''
    return ease2_coords(grid, in_1d)


def get_ease2_slice_idx(grid, subset_id):
    '''
    Returns a tuple `((xmin, xmax), (ymin, ymax))` of the indices that can be
    used to slice a corresponding EASE-Grid 2.0 in order to extract the
    desired bounding box.

    Parameters
    ----------
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.
    subset_id : str
        Keyword designating the desired subset and its corresponding bounding
        box

    Returns
    -------
    tuple
        Tuple of tuples: `((xmin, xmax), (ymin, ymax))`
    '''
    assert grid in EASE2_GRID_PARAMS.keys(),\
        'Could not understand grid argument; must be one of: %s' % ', '.join(EASE2_GRID_PARAMS.keys())
    x_coords, y_coords = ease2_coords(grid, in_1d = True)
    return get_slice_idx_by_bbox(x_coords, y_coords, subset_id)


def get_ease2_slice_offsets(grid, subset_id):
    '''
    Returns the X and Y offsets of an EASE-Grid 2.0 subset, so that an output
    raster array can be aligned properly; see
    `pyl4c.spatial.array_to_raster()`.

    Parameters
    ----------
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.
    subset_id : str
        Keyword designating the desired subset and its corresponding bounding box

    Returns
    -------
    tuple
        The X- and Y-offset `(xoff, yoff)`
    '''
    assert grid in EASE2_GRID_PARAMS.keys(),\
        'Could not understand grid argument; must be one of: %s' % ', '.join(EASE2_GRID_PARAMS.keys())
    slice_idx = get_ease2_slice_idx(grid = grid, subset_id = subset_id)
    return (slice_idx[0][0], slice_idx[1][0])


def get_pft_array(grid, subset_id = None, subset_bbox = None, slice_idx = None):
    '''
    Returns an array of PFT codes on the given EASE-Grid 2.0 for the entire
    globe or a specified subset.

    Arguments
    ---------
    grid : str
        The EASE-Grid 2.0 designation: M01, M09, etc.
    subset_id : str
        (Optional) Instead of `subset_bbox`, can provide keyword designating
        the desired subset area
    subset_bbox : tuple
        (Optional) Instead of `subset_id`, can provide any arbitrary bounding
        box, i.e., sequence of coordinates: `<xmin>, <ymin>, <xmax>, <ymax>`;
        these should be in WGS84 (longitude-latitude) coordinates
    slice_idx : list
        (Optional) Sequence of row-column slice indices to use, for subsetting
        the array; provide as: `[(row1, row2), (col1, col2)]` where the array
        is to be subset as: `array[row1:row2,col1:col2]`

    Returns
    -------
    numpy.ndarray
    '''
    assert grid in EASE2_GRID_PARAMS.keys(),\
        'Could not understand grid argument; must be one of: %s' % ', '.join(EASE2_GRID_PARAMS.keys())
    msg = 'Expected only one of the following arguments: subset_id, subset_bbox, slice_idx'
    assert not all((subset_id is not None, subset_bbox is not None)), msg
    assert not all((subset_bbox is not None, slice_idx is not None)), msg
    m = int(grid[-1]) # Pop off the grid size in km
    anc_hdf_path = ANCILLARY_DATA_PATHS['smap_l4c_ancillary_data_file_path']
    field = ANCILLARY_DATA_PATHS['smap_l4c_%dkm_ancillary_data_lc_path' % m]
    xf = ANCILLARY_DATA_PATHS['smap_l4c_%dkm_ancillary_data_x_coord_path' % m]
    yf = ANCILLARY_DATA_PATHS['smap_l4c_%dkm_ancillary_data_y_coord_path' % m]
    with h5py.File(anc_hdf_path, 'r') as hdf:
        if subset_id is not None:
            pft_array, _, _ = subset(
                hdf, field, hdf[xf][0,:], hdf[yf][:,0], subset_id = subset_id)
            return pft_array
        elif subset_bbox is not None:
            pft_array, _, _ = subset(
                hdf, field, hdf[xf][0,:], hdf[yf][:,0], subset_bbox = subset_bbox)
            return pft_array
        elif slice_idx is not None:
            y0, y1 = slice_idx[0]
            x0, x1 = slice_idx[1]
            return hdf[field][y0:y1,x0:x1]
        else:
            return hdf[field][:]


def get_slice_idx_by_bbox(
        x_coords, y_coords, subset_id = None, subset_bbox = None):
    '''
    Returns a tuple `((xmin, xmax), (ymin, ymax))` of the indices that can be
    used to slice a corresponding EASE2 grid in order to extract the
    desired bounding box. The returned indices correspond to the approximate
    locations of the bounding box (defined in geographic space) in array
    index space; i.e., if `array` is the EASE2 grid of interest, then the
    following slice will extract an area within the bounding box (bbox):

        array[ymin:ymax, xmin:xmax]

    NOTE: Coordinate arrays (`x_coords`, `y_coords`) must be given in the same
    units as the BBOX definition in `SUBSET_BBOX`, i.e., decimal degrees.

    Parameters
    ----------
    x_coords : numpy.ndarray
        A 1D array of X coordinate values
    y_coords : numpy.ndarray
        A 1D array of Y coordinate values
    subset_id : str
        (Optional) Instead of subset_bbox, can provide keyword designating
        the desired subset area
    subset_bbox : tuple
        (Optional) Instead of subset_id, can provide any arbitrary bounding
        box, i.e., sequence of coordinates: `<xmin>, <ymin>, <xmax>, <ymax>`;
        these should be in WGS84 (longitude-latitude) coordinates
    '''
    if subset_id is not None and subset_bbox is not None:
        raise ValueError('Should provide only one argument: Either subset_id or subset_bbox')
    assert x_coords.ndim == 1 and y_coords.ndim == 1, 'Must provide 1D coordinate arrays only'

    bb = subset_bbox
    if subset_id is not None:
        bb = SUBSETS_BBOX[subset_id]

    # Get min, max indices of X coords within bbox; note that we +1 because
    #   ending slice indices are non-inclusive in Python
    if x_coords[-1] > x_coords[0]:
        # If X-coordinates are sorted smallest to largest...
        x_slice_idx = [
            np.where(x_coords >= bb[0])[0].min(),
            np.where(x_coords <= bb[2])[0].max() + 1
        ]
    else:
        x_slice_idx = [
            np.where(x_coords <= bb[0])[0].min(),
            np.where(x_coords >= bb[2])[0].max() + 1
        ]

    # Get min, max indices of Y coords within bbox
    if y_coords[-1] > y_coords[0]:
        # If Y-coordinates are sorted smallest to largest...
        y_slice_idx = [
            np.where(y_coords <= bb[3])[0].max() + 1,
            np.where(y_coords >= bb[1])[0].min()
        ]
    else:
        y_slice_idx = [
            np.where(y_coords <= bb[3])[0].min(),
            np.where(y_coords >= bb[1])[0].max() + 1
        ]

    x_slice_idx.sort() # Necessary b/c slicing is always small:large number
    y_slice_idx.sort()
    return (tuple(x_slice_idx), tuple(y_slice_idx))


def get_xy_coords(hdf_or_nc, in_1d = True):
    '''
    Returns a tuple (longitude, latitude) where the elements are coordinate
    arrays of longitude and latitude. These are needed for, e.g., plotting
    the geophysical data on a global geographic grid. This is convenience
    function for extracting the longitude-latitude coordinates based on the
    filename and our knowledge of where these data are stored. NOTE: This may
    seem like a hack, but is the easiest solution to the fundamental problem
    of inconsistent variable paths; inconsistent naming between dataset IDs
    and filenames; and incomplete documentation of each within the HDF5 file.

    Parameters
    ----------
    hdf_or_nc : h4py.File or netcdf4.Dataset
        Either: an HDF5 file / h5py.File object OR a NetCDF file
    in_1d : bool
        True (Default) to return 1D arrays; if False, returns 2D arrays where
        the coordinates are duplicated along one axis
    '''
    try:
        # HDF5 files
        if hasattr(hdf_or_nc, 'keys') and hasattr(hdf_or_nc, 'filename'):
            if 'GPP' in hdf_or_nc.keys() and 'NEE' in hdf_or_nc.keys():
                d = HDF_PATHS['SPL4CMDL']['4'] # NOTE: We assume Version 4
            elif 'Geophysical_Data' in hdf_or_nc.keys():
                if not 'sm_surface' in hdf_or_nc['Geophysical_Data'].keys():
                    raise ValueError()
                d = HDF_PATHS['SPL4SMGP']['4']
            else:
                raise ValueError()
            x = hdf_or_nc[d['longitude']]
            y = hdf_or_nc[d['latitude']]
        # NetCDF files
        else:
            assert hasattr(hdf_or_nc, 'variables'), 'Assumed NetCDF file has no variables'
            if 'lon' in hdf_or_nc.variables and 'lat' in hdf_or_nc.variables:
                x = hdf_or_nc.variables['lon']
                y = hdf_or_nc.variables['lat']

    except ValueError:
        # Let's assume we know what the X-Y coordinate array keys are
        if 'cell_lon' in hdf_or_nc.keys() and 'cell_lat' in hdf_or_nc.keys():
            x = hdf_or_nc['cell_lon']
            y = hdf_or_nc['cell_lat']
        else:
            raise NotImplementedError('The filename was not recognized as a product with known longitude-latitude data')

    x = x[0,:] if in_1d and len(x.shape) > 1 else x[:]
    y = y[:,0] if in_1d and len(y.shape) > 1 else y[:]
    return (x, y)


def index(array, indices):
    '''
    Fast array indexing by subsetting the array first. If there are multiple
    indices, anywhere in the array, that need to be returned, this provides
    a speed-up by creating a smaller array to index.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing data at (row, column) indices
    indices : tuple or list
        2-element sequence of (row, column) indices, in order

    Returns
    -------
    list
        List of indexed values
    '''
    assert len(indices) == 2,\
        'Must provide 2-element sequence of (row coordinates, column coordinates)'
    assert len(indices[0]) == len(indices[1]),\
        'Array indices must be the same length!'
    assert array.ndim <= 2, 'Array must have 2 or fewer axes'
    midx = np.ravel_multi_index(indices, array.shape)
    return array.ravel()[midx]


def partition_generator(n_elements, n_parts = 1):
    '''
    A Generator that yields slice indices for equal-sized partitions of a
    1D array or sequence. See also: `partition()`.

    Parameters
    ----------
    n_elements : int
        The number of elements in array/ sequence
    n_parts : int
        The number of partitions desired

    Returns
    -------
    generator
    '''
    p = 0
    parts = np.linspace(0, n_elements, n_parts + 1, dtype = int)
    while p < n_parts:
        yield (
            int(parts[p]),
            # If it is the final index, add 1
            int(parts[p+1] if (p != n_parts - 1) else (parts[p+1] + 1))
        )
        p += 1


def partition(array, num_processes, axis = 0):
    '''
    Creates index ranges for partitioning an array to work on over multiple
    processes.

    Parameters
    ----------
    array : numpy.ndarray
        The 2-dimensional array to partition
    num_processes : int
        The number of processes desired
    axis : int
        The axis to break apart into chunks

    Returns
    -------
    list
    '''
    N = array.shape[axis]
    return list(partition_generator(N, num_processes))


def subset(
        hdf_or_nc, field, x_coords = None, y_coords = None, subset_id = None,
        subset_bbox = None):
    '''
    Returns a subset array from the HDF, for the desired variable, where the
    array corresponds to an area defined by a known bounding box, e.g., the
    continental United States (CONUS).

    NOTE: `x_coords` and `y_coords` (hierarchical paths) will be inferred from
    the filename if not provided at all.

    Parameters
    ----------
    hdf_or_nc : h5py.File or netcdf4.Dataset
        Either: an HDF5 file / h5py.File object OR a NetCDF file
    field : str
        Hierarchical path to the desired variable
    x_coords : numpy.ndarray or str
        (Optional) 1D NumPy array of X coordinates OR hierarchical path to
        the variable representing X coordinates, e.g., longitude values
    y_coords : numpy.ndarray or str
        (Optional) 1D NumPy array of Y coordinates OR hierarchical path to
        the variable representing Y coordinates, e.g., latitude values
    subset_id : str
        (Optional) Instead of subset_bbox, can provide keyword designating
        the desired subset area
    subset_bbox : tuple or list
        (Optional) Instead of subset_id, can provide any arbitrary bounding
        box, i.e., sequence of coordinates: `<xmin>, <ymin>, <xmax>, <ymax>`;
        these should be in WGS84 (longitude-latitude) coordinates

    Returns
    -------
    tuple
        Tuple of: subset array, xoff, yoff; (numpy.ndarray, Int, Int)
    '''
    assert (subset_id is None and subset_bbox is not None) or (subset_id is not None and subset_bbox is None), 'Should provide only one argument: Either subset_id or subset_bbox'
    assert isinstance(hdf_or_nc, h5py.File) or hasattr(hdf_or_nc, 'variables'), 'An HDF5 or NetCDF file is required; cannot subset a stand-alone array'
    assert (x_coords is None and y_coords is None) or (isinstance(x_coords, str) and isinstance(y_coords, str)) or (isinstance(x_coords, np.ndarray) and isinstance(y_coords, np.ndarray)), 'The x_coords and y_coords arguments must have matching type'
    assert (not isinstance(x_coords, np.ndarray)) or (x_coords.ndim == 1), 'The x_coords and y_coords arguments must be 1D arrays, otherwise pass a String or None'

    # Check that we have a 2D array to work with
    if hasattr(hdf_or_nc, 'variables'):
        shp = hdf_or_nc.variables[field].shape
    else:
        assert field in hdf_or_nc.keys(), 'Field name "%s" not found' % field
        shp = hdf_or_nc[field].shape
    assert len(shp) == 2, 'HDF5 or NetCDF data array indexed by "%s" must be a 2D array' % field

    # If a hierarchical path to the X and Y coordinate variables was not
    #   given, then infer the paths from the filename
    if x_coords is None:
        x_coords, y_coords = get_xy_coords(hdf_or_nc, in_1d = True)
    if isinstance(x_coords, str):
        # NOTE: Only need first row of X-coordinate array, first column of
        #   Y-coordinate array, because they are duplicated thereafter
        if hasattr(hdf_or_nc, 'variables'):
            x_coords = hdf_or_nc.variables[x_coords][0,:] # NetCDF
            y_coords = hdf_or_nc.variables[y_coords][0,:]
        else:
            x_coords = hdf_or_nc[x_coords][0,:] # HDF5
            y_coords = hdf_or_nc[y_coords][:,0]

    x_idx, y_idx = get_slice_idx_by_bbox(
        x_coords, y_coords, subset_id, subset_bbox)
    xmin, xmax = x_idx # Unpack the slice range indexes
    ymin, ymax = y_idx
    if hasattr(hdf_or_nc, 'variables'):
        return (hdf_or_nc.variables[field][ymin:ymax, xmin:xmax], xmin, ymin)
    return (hdf_or_nc[field][ymin:ymax, xmin:xmax], xmin, ymin)


def sample(array, indices):
    '''
    Samples the value in an array at each (row, column) position in a sequence
    of row-column index pairs.

    Parameters
    ----------
    array : numpy.ndarray
    indices : tuple
        Tuple of (x, y) coordinate pairs; no z-level should be be provided
        (2D coordinate pairs only)

    Returns
    -------
    numpy.ndarray
    '''
    indices_by_axis = [
        int(i[0]) for i in indices], [int(i[1]) for i in indices
    ]
    # If row, column indices are not sorted, we'll need another list comprehension
    try:
        return array[indices_by_axis]
    except TypeError:
        return [array[idx[0], idx[1]] for idx in indices]


def summarize(
        data_array, summaries, scale = 1, data_mask = None, nodata = -9999):
    '''
    Calculates statistical summar[ies] of an input data array over all values.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array of data values that we wish to summarize, e.g., gridded land
        surface temperatures
    summaries : dict
        A Dictionary of `{name: function}` where function is some vectorized
        function that acts over the values associated with a class label,
        including NaNs, and returns a single number
    scale : int or float
        Optional scaling parameter to apply to the input array values, e.g.,
        if the array values are (spatial) rates and should be scaled by the
        (equal) area of the grid cell
    nodata : int or float
        NoData or Fill value(s) to ignore; can pass a sequence of multiple
        values (Default: -9999)

    Returns
    -------
    dict
    '''
    assert data_array.ndim <= 2 or (data_array.ndim == 3 and data_array.shape[0] == 1), 'Can only work with 1-band raster arrays'
    if data_array.ndim == 3:
        data_array = data_array[0,...] # Unwrap 1-band raster arrays

    # Fill in NaN where there is NoData
    data_array = np.where(np.isin(data_array, nodata), np.nan, data_array)
    stats = dict([(k, None) for k in summaries.keys()])
    for stat_name, func in summaries.items():
        # NOTE: Runs faster if dtype of accumulator is *not* set
        stats[stat_name] = func(np.multiply(data_array, scale))
    return stats


def summarize_by_class(
        data_array, class_array, summaries, scale = 1, ignore = (0,),
        nodata = -9999):
    '''
    Calculates statistical summar[ies] of an input data array for each class
    label in an input class array.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array of data values that we wish to summarize, e.g., gridded land
        surface temperatures
    class_array : numpy.ndarray
        Array of class labels that will be used to summarize input data,
        e.g., watersheds or ecoregions; should NOT contain any NaNs as this
        will cause the function to hang
    summaries : dict
        A Dictionary of `{name: function}` where function is some vectorized
        function that acts over the values associated with a class label,
        including NaNs, and returns a single number
    scale : int or float
        Optional scaling parameter to apply to the input array values, e.g.,
        if the array values are (spatial) rates and should be scaled by the
        (equal) area of the grid cell
    ignore : tuple
        Class labels (values in the class_array) to ignore
    nodata : int or float
        NoData or Fill value to ignore (Default: -9999)

    Returns
    -------
    dict
        A nested Python dictionary with a key-value pair for each class,
        where the value is another dictionary with a key-value pair for each
        summary statistic.
    '''
    assert data_array.ndim <= 2 or (data_array.ndim == 3 and data_array.shape[0] == 1), 'Can only work with 1-band raster arrays'
    if data_array.ndim == 3:
        data_array = data_array[0,...] # Unwrap 1-band raster arrays
    assert data_array.shape == class_array.shape, 'Input data_array does not match shape of the class_array'
    # Fill in NaN where there is NoData
    data_array = np.where(np.isin(data_array, nodata), np.nan, data_array)
    # In case None was passed to "ignore," replace with empty list
    ignore = ignore if ignore is not None else []
    assert not np.any(np.isnan(class_array)),\
        'Class array should not contain NaNs; use "ignore" argument instead'
    classes = set(np.unique(class_array[~np.isnan(class_array)])) # Create, e.g., {1: {}, 2: {}, ...}
    stats = dict([(k, dict()) for k in classes.difference(ignore)])
    for code in classes.difference(ignore):
        query = np.where(np.isin(class_array, code), data_array, np.nan)
        stats[code] = summarize(query, summaries, scale, nodata = nodata)
    return stats


def summarize_hdf_by_pft(
        hdf, field, summaries, scale = 1, pft_codes = range(1, 9),
        subset_id = 'CONUS', x_coords = None, y_coords = None,
        nodata = -9999):
    '''
    Calculates statistical summar[ies] of an input data array for each Plant
    Functional Type (PFT) class.

    Parameters
    ----------
    hdf : h5py.File
        The HDF5 file / h5py.File object
    field : str
        One of: "SOC", "NEE", "GPP", or "RH"
    summaries : dict
        A Dictionary of `{name: function}` where function is some vectorized
        function that acts over the values associated with a class label,
        including NaNs, and returns a single number
    scale : int or float
        Optional scaling parameter to apply to the input array values, e.g.,
        if the array values are (spatial) rates and should be scaled by the
        (equal) area of the grid cell
    pft_codes   The PFT codes to create summaries for; defaults to all
                of them, (i.e. Default: 1-8 inclusive)
    subset_id   Keyword designating the desired subset area
                (Default: CONUS)
    x_coords    (Optional) A 1D NumPy array of X coordinate values, used
                in subsetting and can be automatically discovered
    y_coords    (Optional) A 1D NumPy array of Y coordinate values, used
                in subsetting and can be automatically discovered
    nodata      NoData or Fill value to ignore (Default: -9999)
    '''
    assert field in ('SOC', 'NEE', 'GPP', 'RH'), 'Only possible to summarize one of: "SOC", "NEE", "GPP", or "RH"'
    assert (subset_id in SUBSETS_BBOX.keys()) or (subset_id is None), 'Named subset_id not a recognized geographic subset; see SUBSETS_BBOX in this module'
    # Warn user against using nansum() or sum() functions
    if any([k.rfind('sum') > 0 for k in summaries.keys()]):
        print('WARNING: Totals or sums may be biased low because subgrid heterogeneity is not recognized; use total_hdf_by_pft() instead')

    field_names = [
        '%s/%s_pft%d_mean' % (field, field.lower(), p) for p in pft_codes
    ]
    stats = dict([(p, dict()) for p in field_names])
    for name in field_names:
        if subset_id is not None:
            array, x, y = subset(
                hdf, name, x_coords, y_coords, subset_id)
        else:
            array = hdf[name][:]

        stats[name] = summarize(array, summaries, scale, nodata = nodata)

    return stats


def total_hdf_by_pft(
        hdf, counts, field, scale = 1, pft_codes = range(1, 9),
        max_count = 81, subset_id = 'CONUS', x_coords = None, y_coords = None,
        nodata = -9999):
    '''
    Calculates statistical summar[ies] of an input data array for each Plant
    Functional Type (PFT) class. The problem with calculating *totals* for
    PFT means is that we need to know the area proportion of a given PFT
    within each 9-km cell in order to correctly sum over that area. This
    function correctly scales the 9-km PFT mean value by the provided `scale`
    argument and then scales this value by the proportion of 1-km subgrid
    cells that match the given PFT. This produces accurate sums of spatial
    rates; i.e., the `scale` parameter converts a PFT mean to a (biased)
    total, then the `counts` are used to scale that total by the area
    proportion of the given PFT. The summary function used is np.nansum.

    Parameters
    ----------
    hdf : h5py.File
        The HDF5 file / h5py.File object
    counts : tuple or list
        A sequence of NumPy arrays, each array corresponding to the count of
        1-km subcells matching a certain PFT; these should be provided in
        order, i.e., first element is the count of PFT 1 subcells, second
        element for PFT 2, and on.
    field : str
        One of: "SOC", "NEE", "GPP", or "RH"
    scale : int or float
        Optional scaling parameter to apply to the input array values, e.g.,
        if the array values are (spatial) rates and should be scaled by the
        (equal) area of the grid cell
    pft_codes : tuple or list
        The PFT codes to create summaries for; defaults to all of them,
        (i.e. Default: 1-8 inclusive)
    max_count : int
        The maximum number of 1-km subcells that any given PFT can total
        within a 9-km area; no reason why this shouldn't be 81 (9 x 9),
        probably.
    subset_id : str
        (Optional) Instead of subset_bbox, can provide keyword designating
        the desired subset area
    x_coords : numpy.ndarray or str
        (Optional) 1D NumPy array of X coordinates OR hierarchical path to
        the variable representing X coordinates, e.g., longitude values
    y_coords : numpy.ndarray or str
        (Optional) 1D NumPy array of Y coordinates OR hierarchical path to
        the variable representing Y coordinates, e.g., latitude values
    nodata : int or float
        NoData or Fill value to ignore (Default: -9999)
    '''
    assert len(counts) == len(pft_codes),\
        'Length of counts and pft_codes must be equal'
    assert field in ('SOC', 'NEE', 'GPP', 'RH'),\
        'Only possible to summarize one of: "SOC", "NEE", "GPP", or "RH"'
    assert (subset_id in SUBSETS_BBOX.keys()) or (subset_id is None),\
        'Named subset_id not a recognized geographic subset; see SUBSETS_BBOX in this module'
    summaries = {'nansum': np.nansum}
    stats = dict()
    for i, pft in enumerate(pft_codes):
        # Get the name of the target field; set up summary stats dict()
        fieldname = '%s/%s_pft%d_mean' % (field, field.lower(), pft)
        stats[fieldname] = dict()

        if subset_id is not None:
            array, _, _ = subset(
                hdf, fieldname, x_coords, y_coords, subset_id)
        else:
            array = hdf[fieldname][:]

        assert array.shape == counts[i].shape, 'Counts array must be the same size as data array'
        # Scale the data array by the proportion of the subgrid cells that
        #   match the given PFT class
        stats[fieldname] = summarize(
            np.multiply( # NOTE: We scale data array ahead of time...
                np.divide(counts[i], max_count),
                np.multiply(array, scale)),
            # ...So scale must be fixed at 1
            summaries, scale = 1, nodata = nodata)

    return stats
