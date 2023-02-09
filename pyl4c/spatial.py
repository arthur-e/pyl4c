'''
Helper functions and data (EASE-Grid 2.0 parameters) for converting SMAP L4C
data from its native HDF5 format into or out of various GIS-friendly formats.
It may seem arbitrary that the subset() function is in the `utils` module and
not here, but the main distinction is that `spatial` module includes functions
that depend on the GDAL library (osgeo.gdal) but `utils` does not and should
not.

Corner coordinates for EASE-Grid 2.0 obtained from [1] and EPSG codes
obtained from [2]. The same `proj4` string is used for all EASE2 grids; see [3].

    # proj4 string
    +proj=cea +lat_0=0 +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m

Note also that `array_to_raster()`, `array_to_raster_clone()`, `as_array()`,
and `dump_raster()` are copied from my (MIT-licensed) `unmixing` library [4].

1. https://nsidc.org/ease/ease-grid-projection-gt
2. https://nsidc.org/data/ease
3. https://doi.org/10.3390/ijgi3031154
4. https://github.com/arthur-e/unmixing/
'''

import tempfile
import numpy as np
import pyproj
from osgeo import gdal, gdalconst, gdal_array, osr
from affine import Affine
from pyl4c.data.fixtures import EASE2_GRID_PARAMS
from pyl4c.epsg import EPSG
from pyl4c.utils import sample

def array_to_raster(
        a, gt, wkt, xoff = None, yoff = None, dtype = None, nodata = -9999):
    '''
    Creates a raster from a given array, with optional x- and y-offsets
    if the array was clipped.

    Parameters
    ----------
    a : numpy.ndarray
        NumPy array to convert to a raster dataset
    gt : tuple
        A GDAL GeoTransform tuple
    wkt : str
        Spatial reference system (SRS) as Well-Known Text projection string
    xoff : int
        The offset in the x-direction; should be provided when clipped
    yoff : int
        The offset in the y-direction; should be provided when clipped
    dtype : type
        The data type to coerce on the array
    nodata : int or float
        The NoData or Fill value (Default: -9999)

    Returns
    -------
    gdal.Dataset
    '''
    if dtype is not None:
        a = a.astype(dtype)
    try:
        rast = gdal_array.OpenNumPyArray(a)
    except AttributeError:
        # For backwards compatibility with older version of GDAL
        rast = gdal.Open(gdal_array.GetArrayFilename(a))
    except:
        rast = gdal_array.OpenArray(a)
    rast.SetGeoTransform(gt)
    rast.SetProjection(wkt)
    if nodata is not None:
        for band in range(1, rast.RasterCount + 1):
            rast.GetRasterBand(band).SetNoDataValue(nodata)
    if xoff is not None and yoff is not None:
        # Bit of a hack; essentially, re-create the raster but with the
        #   correct X and Y offsets (don't know how to do this without the
        #   use of CopyDatasetInfo())
        return array_to_raster_clone(a, rast, xoff, yoff)
    return rast


def array_to_raster_clone(a, proto, xoff = None, yoff = None):
    '''
    Creates a raster from a given array and a prototype raster dataset, with
    optional x- and y-offsets if the array was clipped.

    Parameters
    ----------
    a : numpy.ndarray
        NumPy array to convert to a raster dataset
    proto : gdal.Dataset
        A prototype dataset
    xoff : int or None
        The offset in the x-direction; should be provided when clipped
    yoff : int or None
        The offset in the y-direction; should be provided when clipped

    Returns
    -------
    gdal.Dataset
    '''
    try:
        rast = gdal_array.OpenNumPyArray(a)
    except AttributeError:
        # For backwards compatibility with older version of GDAL
        rast = gdal.Open(gdal_array.GetArrayFilename(a))
    except:
        rast = gdal_array.OpenArray(a)
    kwargs = dict()
    if xoff is not None and yoff is not None:
        kwargs = dict(xoff=xoff, yoff=yoff)
    # Copy the projection info and metadata from a prototype dataset
    if type(proto) == str:
        proto = gdal.Open(proto)
    gdal_array.CopyDatasetInfo(proto, rast, **kwargs)
    return rast


def as_array(file_path, band_axis = True, subset_idx = None, nodata = None):
    '''
    A convenience function for opening a raster as an array and accessing its
    spatial information; takes a single string argument.

    NOTE: subset_idx should be provided as either a flat sequence or nested
    tuples. For a flat sequence (GDAL format):

        (<xoff>, <yoff>, <xmax>, <ymax>)

    Where `<xoff>, <yoff>` are the offsets in the X- and Y-directions (column
    and row directions, starting from the top-left corner. The last two
    numbers indicate the ending column and row indices.

    For nested tuples (for backwards compatibility):

        ((<xoff>, <xmax>), (<yoff>, <ymax>))

    For instance, if the offsets equal zero and the `<xmax>, <ymax>` terms are
    equal to the total number of columns and rows, respectively, then the
    entire raster array is read (i.e., might as well set `subset_idx = None`).

    Parameters
    ----------
    file_path : str
        The path of the raster file to open as an array
    band_axis : bool
        True to include a band axis, even for single-band rasters
    subset_idx : tuple or list
        (Optional) Can provide any arbitrary bounding box sequence as
        row-column indices (not geographic coordinates) to get a spatial
        subset instead of the entire raster array.
    nodata : int or float
        (Optional) If not None, fills this value with NaN (numpy.nan)

    Returns
    -------
    tuple
        Sequence of (numpy.ndarray, geotransform tuple, projection str)
    '''
    ds = gdal.Open(file_path)
    if len(ds.GetSubDatasets()) > 0:
        raise ValueError(
            'This dataset has multiple layers; you must specify a valid subdataset, for example: %s' % ds.GetSubDatasets()[0][0])
    if subset_idx is not None:
        bb = subset_idx
        # If nested sequence is passed
        if hasattr(subset_idx[0], '__len__'):
            bb = [bb[0][0], bb[1][0], bb[0][1], bb[1][1]]
        # Default is for GDAL-style bounds (<xmin>, <ymin>, <xmax>, <ymax>)
        assert all(map(lambda x: x >= 0, bb)),\
            'Elements of subset_idx must be slice indices >/= 0'
        arr = ds.ReadAsArray(
            *map(int, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])))
    else:
        arr = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    ds = None
    # Make sure that single-band rasters have a band axis
    if band_axis and len(arr.shape) < 3:
        shp = arr.shape
        arr = arr.reshape((1, shp[0], shp[1]))
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return (arr, gt, wkt)


def as_raster(file_path, subset_bbox = None):
    '''
    A convenience function for opening a raster and accessing its spatial
    information; takes a single string argument.

    NOTE: subset_bbox should be provided as either a flat sequence or nested
    tuples. For a flat sequence (GDAL format):

        (<xmin>, <ymin>, <xmax>, <ymax>)

    For nested tuples (for backwards compatibility):

        ((<xmin>, <xmax>), (<ymin>, <ymax>))

    Parameters
    ----------
    file_path : str
        The path of the raster file to open as a gdal.Dataset
    subset_bbox : tuple or list
        (Optional) The bounding-box coordinates, as geographic coordinate
        pairs, to use in extracting a spatial subset

    Returns
    -------
    tuple
        Tuple of: (gdal.Dataset, geotransform tuple, projection str)
    '''
    ds = gdal.Open(file_path)
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    if subset_bbox is not None:
        bb = subset_bbox
        # If nested sequence is passed
        if hasattr(subset_bbox[0], '__len__'):
            bb = [bb[0][0], bb[1][0], bb[0][1], bb[1][1]]

        # NOTE: Because SRS coordinate system is "flipped" vertically
        #   w.r.t. pixel coordinate system, we flip ymax and ymin;
        # See: https://www.perrygeo.com/python-affine-transforms.html
        ymax, xmin = xy_to_pixel(bb[0:2], gt)
        ymin, xmax = xy_to_pixel(bb[2:4], gt)
        array, gt, wkt = as_array( # Read in as array
            file_path, band_axis, (xmin, ymin, xmax, ymax))
        ds = array_to_raster(array, gt, wkt, xmin, ymin)
        # Get updated GeoTransform, with correct upper-left coordinate pair
        gt = ds.GetGeoTransform()
    return (ds, gt, wkt)


def binary_reclass(a, target_input = (0,), output = (0, 1)):
    '''
    For an input raster array, produce a binary output array, replacing all
    cases of `target_input` values with outputs[0] and all other cases with
    output[1].

    Parameters
    ----------
    a : numpy.ndarray
        A NumPy array to reclassify
    target_input : tuple or list or numpy.ndarray
        Vector of input values to replace with output[0]
    output : tuple or list or numpy.ndarray
        2-element vector of values which will replace input values

    Returns
    -------
    numpy.ndarray
    '''
    assert len(output) == 2, 'Only two output values should be provided'
    return np.where(np.isin(a, target_input), output[0], output[1])


def bounds(rast):
    '''
    Returns the minimum and maximum coordinate values in the sequence expected
    by, e.g., the `-te` switch in various GDAL utiltiies:
    `(xmin, ymin, xmax, ymax)`. Copied from my MIT-licensed library
    `gdal_extent.py`.

    Parameters
    ----------
    rast : gdal.Dataset

    Returns
    -------
    tuple
        The bounds of the raster, i.e., `(xmin, ymin, xmax, ymax)`
    '''
    gt = rast.GetGeoTransform()
    xsize = rast.RasterXSize # Size in the x-direction
    ysize = rast.RasterYSize # Size in the y-direction
    xr = abs(gt[1]) # Resolution in the x-direction
    yr = abs(gt[-1]) # Resolution in the y-direction
    return (gt[0], gt[3] - (ysize * yr), gt[0] + (xsize * xr), gt[3])


def crs_transform(from_epsg, to_epsg):
    '''
    Creates a coordinate reference system (CRS) transformation. The resulting
    `CoordinateTransformation` instance should be used, e.g.:

        crs_transform(4326, 6933).TransformPoint(latitude, longitude)

    Parameters
    ----------
    from_epsg : int
        The EPSG code for the originating CRS
    to_epsg : int
        The EPSG code for the target (output) CRS

    Returns
    -------
    gdal.osr.CoordinateTransformation
    '''
    source = osr.SpatialReference()
    target = osr.SpatialReference()
    source.ImportFromEPSG(from_epsg)
    target.ImportFromEPSG(to_epsg)
    return osr.CoordinateTransformation(source, target)


def dump_raster(
        rast, rast_path, driver = 'GTiff', gdt = None, nodata = None,
        compress = False):
    '''
    Creates a raster file from a given `gdal.Dataset` instance.

    Parameters
    ----------
    rast : gdal.Dataset
        A gdal.Dataset; does NOT accept NumPy array
    rast_path : str
        The path of the output raster file
    driver : str
        The name of the GDAL driver to use, which determines output file type
        (Default: "GTiff")
    gdt : int
        The GDAL data type to use, e.g., see gdal.GDT_Float32
    nodata : int or float
        The NoData value; defaults to -9999.
    compress : bool
        True to apply lossless compression to the output; requires that the
        "GTiff" driver is used (GeoTIFF output)
    '''
    if gdt is None:
        gdt = rast.GetRasterBand(1).DataType
    driver = gdal.GetDriverByName(driver)
    if compress and driver.ShortName == 'GTiff':
        _rast_path = rast_path
        tmp = tempfile.NamedTemporaryFile()
        rast_path = tmp.name
    sink = driver.Create(
        rast_path, rast.RasterXSize, rast.RasterYSize, rast.RasterCount, int(gdt))
    assert sink is not None,\
        'Cannot create dataset; there may be a problem with the output path you specified'
    sink.SetGeoTransform(rast.GetGeoTransform())
    sink.SetProjection(rast.GetProjection())
    for b in range(1, rast.RasterCount + 1):
        dat = rast.GetRasterBand(b).ReadAsArray()
        sink.GetRasterBand(b).WriteArray(dat)
        sink.GetRasterBand(b).SetStatistics(*map(np.float64,
            [dat.min(), dat.max(), dat.mean(), dat.std()]))
        if nodata is None:
            nodata = rast.GetRasterBand(b).GetNoDataValue()
            if nodata is None:
                nodata = -9999
        sink.GetRasterBand(b).SetNoDataValue(np.float64(nodata))
    sink.FlushCache()
    if not compress or driver.ShortName != 'GTiff':
        return # Done
    # Optionally, compress the output file
    opts = gdal.TranslateOptions(
        format = 'GTiff', creationOptions = ['COMPRESS=LZW'])
    # Note that "_rast_path" is the true output path, "rast_path" just temp
    gdal.Translate(_rast_path, rast_path, options = opts)


def ease2_coords_approx(grid, in_1d = True, srs_epsg = 4326, precision = 8):
    '''
    Returns approximate EASE-Grid 2.0 coordinate arrays. They are
    "approximate" because, although they are calculated correctly using affine
    transformations, they do not correspond to the coordinate arrays provided
    in typical EASE-Grid 2.0 products, namely SMAP L4C. They only correspond
    with these arrays to 2nd or 3rd decimal place in geographic coordinates
    (i.e., with `srs_epsg = 4326`).

    Parameters
    ----------
    grid : str
        Output grid size: "M01" or "M09"
    in_1d : bool
        True to produce 1D arrays (Default: True)
    srs_epsg : int
        The EPSG code of the desired output SRS (Default: 4326)
    precision : int
        The decimal precision of output coordinates (Default: 8)

    Returns
    -------
    tuple
        Two coordinate arrays, X and Y: (numpy.ndarray, numpy.ndarray)
    '''
    srs0 = osr.SpatialReference()
    srs0.ImportFromWkt(EPSG[EASE2_GRID_PARAMS[grid]['epsg']])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(srs_epsg)
    native = Affine.from_gdal(*EASE2_GRID_PARAMS[grid]['geotransform'])
    target = osr.CoordinateTransformation(srs0, srs)
    shp = EASE2_GRID_PARAMS[grid]['shape']
    rows = np.arange(0, shp[0]) + 0.5 # To get center of a pixel
    cols = np.arange(0, shp[1]) + 0.5
    l1, l2 = (1, 0) if gdal.VersionInfo()[0] == '3' else (0, 1)
    x_coords = np.array([
        tuple(map( # Convert from (row, column) to (easting, northing)...
            # Then to (latitude, longitude)
            lambda x: round(x, precision),
            target.TransformPoint(*np.array(native * (i, 0)))))[l1]
        for i in cols
    ])
    y_coords = np.array([
        tuple(map(
            lambda x: round(x, precision),
            target.TransformPoint(*np.array(native * (0, i)))))[l2]
        for i in rows
    ])
    if in_1d:
        return (x_coords, y_coords)
    return (
        np.repeat(x_coords.reshape((1, shp[1])), shp[0], axis = 0),
        np.repeat(y_coords.reshape((shp[0], 1)), shp[1], axis = 1))


def ease2_to_geotiff(
        array, output_path, grid, xoff = None, yoff = None, dtype = None,
        nodata = -9999, compress = False):
    '''
    Convenience function for writing an EASE-Grid 2.0 data array to a
    GeoTIFF raster file.

    Parameters
    ----------
    array : numpy.ndarray
        A NumPy data array
    output_path : str
        The output GeoTIFF raster file path
    grid : str
        One of: "M09", "M01"
    xoff : int
        The offset in the x-direction; should be provided when clipped
    yoff : int
        The offset in the y-direction; should be provided when clipped
    dtype : type
        The NumPy data type to enforce, e.g., np.int32
    nodata : int or float
        The NoData or Fill value
    compress : bool
        True to apply lossless compression to the output (Default: False)
    '''
    # Check that the provided array size/shape makes sense
    expected = EASE2_GRID_PARAMS[grid]['shape']
    if xoff is None and yoff is None:
        assert (array.ndim == 2 and expected == array.shape) or (array.ndim == 3 and expected == array.shape[1:3]),\
        'Array shape (%d, %d) does not matched expected shape (%d, %d); set xoff and yoff if this array is clipped' % (*array.shape[1:3], *expected)
    gt = EASE2_GRID_PARAMS[grid]['geotransform']
    wkt = EPSG[EASE2_GRID_PARAMS[grid]['epsg']]
    rast = array_to_raster(array, gt, wkt, xoff, yoff, dtype = dtype)
    dump_raster(
        rast, output_path, driver = 'GTiff', nodata = nodata,
        compress = compress)


def ease2_hdf_to_geotiff(
        hdf, field, output_path, grid = 'M09', dtype = None, nodata = None):
    '''
    Dumps a single field from an HDF5 file into a GeoTIFF file. The grid
    argument refers to which EASE-Grid 2.0 is used:

    - M01: 1-km Global grid
    - M09: 9-km Global grid

    Parameters
    ----------
    hdf : h5py.File
        The HDF5 file / h5py File object
    field : str
        The hierarchical path to the variable of interest
    output_path : str
        The output file path
    grid : str
        EASE-Grid 2.0 type: "M01", "M09", ...
    dtype : type
        The NumPy data type to coerce on the array
    nodata : int or float
        The NoData value or "fill" value
    '''
    gt = EASE2_GRID_PARAMS[grid]['geotransform']
    wkt = EPSG[EASE2_GRID_PARAMS[grid]['epsg']]
    # Get the NoData value from the dataset description, if possible
    if nodata is None and '_FillValue' in hdf[field].attrs:
        nodata = hdf[field].attrs['_FillValue']
    ease2_to_geotiff(hdf[field][:], output_path, grid, dtype, nodata)


def intersect_rasters(
        ref_raster, src_raster, nodata = -9999, nudge = (0, 0),
        method = gdalconst.GRA_Bilinear, gdt = gdalconst.GDT_Float32):
    '''
    Projects the source raster so that its top-left corner is aligned with
    the reference raster. Then, clips or pads the source raster so that it
    has the same number of rows and columns (covers the same extent at the
    same grid resolution) as the reference raster. Can also "nudge" a raster
    in one or both directions, which can help fix small differences in
    alignment.

    NOTE: If the reference raster's top-left corner is far left and/or above
    that of the source raster, the intersected raster may contain no data
    from the original raster, i.e., an empty raster will result.

    NOTE: If the `ref_raster` and `src_raster` have different projections, the
    result may contain no data (i.e., all `NoData`). Not sure why, as here we
    are projecting the data to match the reference raster. This method should
    be used with rasters that are "close" but not an exact match for SRS,
    resolution, and extent.

    Parameters
    ----------
    ref_raster : gdal.Dataset
        The reference raster
    src_raster : gdal.Dataset
        The source raster; the raster to be changed
    nodata : int or float
        The NoData value to fill in where the reference raster is larger
    nudge : tuple or list
        Sequence of two integers, the number of pixels to push a raster array
        in the X and Y directions, respectively
    method : int
        gdal.GRA_Bilinear or gdal.GRA_NearestNeighbour are recommended
        (Default: gdal.GRA_Bilinear)
    gdt : int
        The GDAL data type to use for the output raster
        (Default: gdalconst.GDT_Float32)

    Returns
    -------
    gdal.Dataset
    '''
    msg = '%s is not a gdal.Dataset!'
    assert hasattr(ref_raster, 'GetGeoTransform'), msg % '"ref_raster"'
    assert hasattr(src_raster, 'GetGeoTransform'), msg % '"src_raster"'
    # Check that raster has spatial projection information
    msg = '%s has no defined spatial reference!'
    assert ref_raster.GetProjection() != '', msg % '"ref_raster"'
    assert src_raster.GetProjection() != '', msg % '"src_raster"'

    gt = ref_raster.GetGeoTransform()
    wkt = ref_raster.GetProjection()
    gt0 = src_raster.GetGeoTransform()
    wkt0 = src_raster.GetProjection()

    # Create a new raster with the desired attributes
    width, height = (ref_raster.RasterXSize, ref_raster.RasterYSize)
    width0, height0 = (src_raster.RasterXSize, src_raster.RasterYSize)
    out_raster = gdal.GetDriverByName('MEM').Create('temp.file',
        width, height, src_raster.RasterCount, gdt)

    # Initialize and set the NoData value
    for i in range(1, src_raster.RasterCount + 1):
        b = out_raster.GetRasterBand(i)
        b.Fill(nodata)
        b.SetNoDataValue(nodata)

    # Set the desired geotransform, and projection
    out_raster.SetGeoTransform(gt)
    out_raster.SetProjection(wkt)

    # Re-project the source image; now the top-left corners are aligned
    gdal.ReprojectImage(src_raster, out_raster, wkt0, wkt, method)
    arr0 = out_raster.ReadAsArray()
    del src_raster # Delete original raster references
    del ref_raster
    del out_raster
    # Clip the extent of the src image if ref is smaller
    if arr0.ndim > 2: ch, cw = arr0.shape[1:]
    else: ch, cw = arr0.shape
    if (width <= width0): cw = width # Clip src to ref extents
    if (height <= height0): ch = height

    # Read rows, columns out to the extent of the output array
    if arr0.ndim > 2:
        arr = arr0[:,0:ch,0:cw]
    else:
        arr = arr0[0:ch,0:cw]

    # Finally, "nudge" rows and columns over and down, as needed
    nx, ny = nudge
    # Nudge rows in the y-direction
    if ny > 0:
        arr = np.vstack((np.ones(arr.shape)[:ny,:] * np.nan, arr))
    elif ny < 0:
        arr = np.vstack((arr, np.ones(arr.shape)[:abs(ny),:] * np.nan))

    # Nudge columns in the x-direction
    if nx > 0:
        arr = np.hstack((np.ones(arr.shape)[:,:nx] * np.nan, arr))
    elif ny < 0:
        arr = np.hstack((arr, np.ones(arr.shape)[:,:abs(nx)] * np.nan))

    return array_to_raster(arr, gt, wkt)


def intersect_rasters2(
        ref_raster, src_raster, nodata = -9999,
        method = gdalconst.GRA_Bilinear, gdt = gdalconst.GDT_Float32):
    '''
    Improvement upon `intersect_rasters()`, perhaps? This function is
    considerably shorter as it uses a wrapper for the command-line `gdalwarp`.
    Will need to test performance.

    NOTE: If the reference raster's top-left corner is far left and/or above
    that of the source raster, the intersected raster may contain no data
    from the original raster, i.e., an empty raster will result.

    NOTE: If the `ref_raster` and `src_raster` have different projections, the
    result may contain no data (i.e., all `NoData`). Not sure why, as here we
    are projecting the data to match the reference raster.

    Parameters
    ----------
    ref_raster : gdal.Dataset
        The reference raster
    src_raster : gdal.Dataset
        The source raster; the raster to be changed
    nodata : int or float
        The NoData value to fill in where the reference raster is larger
    method : int
        gdal.GRA_Bilinear or gdal.GRA_NearestNeighbour are recommended
        (Default: gdal.GRA_Bilinear)
    gdt : int
        The GDAL data type to use for the output raster
        (Default: gdalconst.GDT_Float32)

    Returns
    -------
    gdal.Dataset
    '''
    msg = '%s is not a gdal.Dataset!'
    assert hasattr(ref_raster, 'GetGeoTransform'), msg % '"ref_raster"'
    assert hasattr(src_raster, 'GetGeoTransform'), msg % '"src_raster"'
    # Check that raster has spatial projection information
    msg = '%s has no defined spatial reference!'
    assert ref_raster.GetProjection() != '', msg % '"ref_raster"'
    assert src_raster.GetProjection() != '', msg % '"src_raster"'
    gt = ref_raster.GetGeoTransform()
    x_res = y_res = np.abs(gt[1])
    wkt_proj4 = pyproj.CRS.from_wkt(ref_raster.GetProjectionRef()).to_proj4()
    return gdal.Warp(
        'temp.file', src_raster, format = 'MEM', xRes = x_res, yRes = y_res,
        dstSRS = wkt_proj4, outputBounds = bounds(ref_raster))


def project_equirectangular(
        src_raster, size_degrees, method = gdal.GRA_Bilinear, nodata = -9999,
        gdt = gdalconst.GDT_Float32, nrows = None, ncols = None):
    '''
    Projects a raster into an equirectangular geographic coordinate system
    (GCS) based on the WGS84 datum. Thanks to J. Gómez-Dans for the more
    general solution to calculating output raster size [1].

    NOTE: This will not work well for very small cell sizes, around 0.1
    degress on a side or less. It should generally be used for down-sampling
    (scaling up) only.

    1. https://jgomezdans.github.io/gdal_notes/reprojection.html

    Parameters
    ----------
    src_raster : gdal.Dataset
        Raster to be projected to a GCS
    size_degrees : int or float
        Output cell size (spatial resolution) in degrees
    method : int
        gdal.GRA_Bilinear or gdal.GRA_NearestNeighbour are recommended
        (Default: gdal.GRA_Bilinear)
    gdt : int
        Output GDAL data type (Default: 32-bit floating point)

    Returns
    -------
    gdal.Dataset
    '''
    assert src_raster.RasterCount == 1, 'Can only work with 1-band raster arrays'
    gt0 = src_raster.GetGeoTransform()
    wkt0 = osr.SpatialReference()
    wkt0.ImportFromWkt(src_raster.GetProjection())

    # Equirectangular (target) spatial reference system
    wkt = osr.SpatialReference()
    wkt.ImportFromEPSG(4326)

    # NOTE: God forbid, if this transformation stops working, note that as of
    #   GDAL 3.0+, TransformPoint() expects coordinates to be provided in
    #   the order that the CRS expects, which is not necessarily X,Y,Z order
    transform = osr.CoordinateTransformation(wkt0, wkt)
    # Work out the boundaries of the new dataset in the target projection
    (uly, ulx, ulz) = transform.TransformPoint(gt0[0], gt0[3])
    (lry, lrx, lrz) = transform.TransformPoint(
        gt0[0] + gt0[1]*src_raster.RasterXSize,
        gt0[3] + gt0[5]*src_raster.RasterYSize)
    # Prior to GDAL 3.x, WGS84 projection defined coordinates in (X,Y), i.e.,
    #   (longitude, latitude) order; now they are (latitude, longitude) order
    if gdal.__version__[0] == '2':
        # So, for older versions, just swap coordinate order
        uly, ulx = (ulx, uly)
        lry, lrx = (lrx, lry)

    if ncols is None:
        ncols = abs(int((lrx - ulx) / size_degrees)) # Get NEW raster width
        # For lat-long projections, this is a sign there may be a problem
        if wkt.IsGeographic() and ncols <= 1:
            # Need to consider West longitude a possibility
            ncols = abs(int((-lrx - ulx) / size_degrees))
    if nrows is None:
        # Neat math fact: Not an issue for southern latitude
        nrows = int(abs((uly - lry) / size_degrees)) # Get NEW raster height
    assert ncols != 0 and nrows != 0,\
        'Rows and/or columns are zero; check that size_degrees is appropriate'

    gt = gdal.AutoCreateWarpedVRT(src_raster, str(wkt0), str(wkt), method)\
        .GetGeoTransform()
    gt = (gt[0], size_degrees, gt[2], gt[3], gt[4], -size_degrees)
    # Alternatively, we can guess some of the parameters (geog. CRS example):
    # gt = (-ulx, size_degrees, gt0[2], uly, gt0[4], -size_degrees)
    rast_projected = gdal.GetDriverByName('MEM')\
        .Create('', ncols, nrows, 1, gdt) # 1 refers to the number of bands

    # Set the NoData value so this is ignored in reprojecting
    src_raster.GetRasterBand(1).SetNoDataValue(nodata)
    output_band = rast_projected.GetRasterBand(1)
    output_band.SetNoDataValue(nodata)
    output_band.Fill(nodata) # This is necessary to place NoData in the output

    # Set the output image SRS and warp warp!
    rast_projected.SetGeoTransform(gt)
    rast_projected.SetProjection(str(wkt))
    gdal.ReprojectImage(
        src_raster, rast_projected, str(wkt0), str(wkt), method)
    return rast_projected


def resample_ease2(
        src_raster, grid, nodata = -9999, method = gdalconst.GRA_Bilinear,
        gdt = gdalconst.GDT_Float32):
    '''
    Resamples an input raster from one nested EASE-Grid 2.0 to another; e.g.,
    from a 1-km grid to a 9-km grid or vice-versa. Uses GDAL for resampling,
    which is expected to produce the same result as a matrix operations that
    exploit grid nesting.

    Parameters
    ----------
    src_raster : gdal.Dataset
        The source raster to be resampled
    grid : str
        Desired output EASE-Grid 2.0 specification, e.g., "M09"
    nodata : int or float
        The NoData value to fill in where the reference raster is larger
    method : int
        gdal.GRA_Bilinear or gdal.GRA_NearestNeighbour are recommended
        (Default: gdal.GRA_Bilinear)
    gdt : int
        The GDAL data type to use for the output raster
        (Default: gdalconst.GDT_Float32)

    Returns
    -------
    gdal.Dataset
    '''
    assert hasattr(src_raster, 'GetGeoTransform'), '"src_raster" is not a gdal.Dataset!'
    assert src_raster.GetProjection() != '', '"src_raster" has no defined spatial reference!'
    # Get attributes of the original (input) raster
    gt0 = src_raster.GetGeoTransform()
    wkt0 = src_raster.GetProjection()
    # Get attributes of the output raster
    res = EASE2_GRID_PARAMS[grid]['geotransform'][1]
    height, width = EASE2_GRID_PARAMS[grid]['shape']
    gt = list(gt0)
    gt[1] = res
    gt[-1] = -res
    # Create the output raster as an in-memory raster file
    out_raster = gdal.GetDriverByName('MEM').Create('temp.file',
        width, height, src_raster.RasterCount, gdt)
    # Set the desired geotransform, and projection
    out_raster.SetGeoTransform(tuple(gt))
    out_raster.SetProjection(wkt0)
    # We're using this empty raster as a reference for projecting the source
    return intersect_rasters(out_raster, src_raster, nodata, method, gdt)


def sample_points(data, points, gt = None, ignore_srs_mismatch = False):
    '''
    Samples the value in a raster array at each location in a sequence of
    geographic coordinate pairs.

    Parameters
    ----------
    data : numpy.ndarray or gdal.Dataset
    points : tuple or list
        Sequence of (X, Y) coordinate pairs
    gt : tuple
        (Optional) GeoTransform tuple
    ignore_srs_mismatch : bool
        True to ignore any apparent differences in the SRS between "points"
        and raster GeoTransform

    Returns
    -------
    list
        A list of sampled raster values
    '''
    if hasattr(data, 'GetGeoTransform'):
        if gt is not None and not np.all(np.equal(gt, rast.GetGeoTransform())):
            raise ValueError('Input raster GeoTransform and user-provided "gt" argument do not match (should drop "gt" argument)')
        gt = rast.GetGeoTransform()
        transform = Affine.from_gdal(*gt)
    else:
        if gt is None:
            raise ValueError('Must specify a geotransform, gt, unless a gdal.Dataset is provided')
        transform = Affine.from_gdal(*gt)

    # Check that the point X coordinate is not too many orders of magnitude
    #   different from the minimum X coordinate of the input raster SRS; if
    #   so, there is likely a mismatch in SRS
    if abs(np.log10(abs(gt[0])) - np.log10(abs(points[0][0]))) >= 2:
        if not ignore_srs_mismatch:
            raise ValueError('Coordinate system of "points" and input data do not match')

    # If points are given as a sequence of pairs, transformation to
    #   row-column indices is straightforward
    if hasattr(points, '__len__'):
        indices = [ # And make sure to reverse to obtain row, column order
            tuple(reversed(tuple(
                map(int, ~transform * pair)))) for pair in points
        ]
    else:
        raise NotImplementedError('Argument "points" must be a sequence type')

    if hasattr(data, 'GetGeoTransform'):
        return sample(data.ReadAsArray(), indices)
    return sample(data, indices)


def xy_to_pixel(coords, gt, coords_srs_epsg = None, raster_srs_epsg = None):
    '''
    Translates a generic X-Y coordinate pair associated with a point into
    pixel coordinates, i.e., the row-column indices in a raster that contains
    that contains that point.

    Parameters
    ----------
    coords : tuple or list
        Coordinate pair, as a 2-element sequence
    gt : tuple
        GeoTransform for the associated raster
    coords_srs_epsg : int
        (Optional) EPSG code for the coordinate pair's SRS
    raster_srs_epsg : int
        (Optional) EPSG code for the associated raster

    Returns
    -------
    tuple
        A 2-element tuple; the row, column indices for the coordinate pair
    '''
    # Get an affine transformation to translate geographic coordinates to
    #   raster coordinates
    affine_transform = Affine.from_gdal(*gt)
    x, y = coords
    # Optionally, transform the geographic coordinates if needed
    if coords_srs_epsg is not None and raster_srs_epsg is not None:
        coords_srs = osr.SpatialReference()
        coords_srs.ImportFromEPSG(coords_srs_epsg)
        # Create a coordinate reference system transformation to translate
        #   latitude-longitude coordinates into EASE-Grid 2.0 coordinates
        if coords_srs.IsGeographic() == 1:
            coords = reversed(coords)
        x, y = crs_transform(coords_srs_epsg, raster_srs_epsg)\
            .TransformPoint(*coords, 0)[:-1] # Drop Z coordinate
    return tuple(map(int, reversed(~affine_transform * (x, y))))
