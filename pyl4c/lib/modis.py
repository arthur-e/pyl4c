'''
MODIS sinusoidal projection forward and backward coordinate transformations,
courtesy of Giglio et al. (2018), Collection 6 MODIS Burned Area Product
User's Guide, Version 1, Appendix B:

    https://modis-fire.umd.edu/files/MODIS_C6_BA_User_Guide_1.2.pdf
'''

import re
import numpy as np

SPHERE_RADIUS = 6371007.181 # Radius of ideal sphere, meters
TILE_LINES = {250: 5000, 500: 2400, 1000: 1200} # Num. lines by nominal res.
TILE_SIZE = 1111950 # Width and height of MODIS tile in projection plane
XMIN = -20015109 # Western limit of projection plane, meters
YMAX = 10007555 # Northern limit of projection plane, meters
VIIRS_METADATA = re.compile(
    r'.*XDim=(?P<xdim>\d+)'
    r'.*YDim=(?P<ydim>\d+)'
    r'.*UpperLeftPointMtrs=\((?P<ul>[0-9,\-\.]+)\)'
    r'.*LowerRightMtrs=\((?P<lr>[0-9,\-\.]+)\).*', re.DOTALL)
VIIRS_H5_ROOT = 'HDFEOS/GRIDS/VNP_Grid_VNP15A2H/Data Fields'

def dec2bin_unpack(x):
    '''
    Unpacks an arbitrary decimal NumPy array into a binary representation
    along a new axis. Assumes decimal digits are on the interval [0, 255],
    i.e., that only 8-bit representations are needed.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    '''
    # Make sure the bit representation is enumerated along a new axis, the
    #   very last axis
    axis = x.ndim
    # unpackbits() returns the bit representation in big-endian order, so we
    #   flip the array (with -8) to get litte-endian order
    return np.unpackbits(x[...,None], axis = axis)[...,-8:]


def geotransform(
        hdf, ps = 463.31271653, nrow = 2400, ncol = 2400,
        metadata = VIIRS_METADATA):
    '''
    Prescribe a geotransform tuple for the output GeoTIFF. For MODIS/VIIRS
    sinsuoidal projections, the lower right corner coordinates are "the only
    metadata that accurately reflect the extreme corners of the gridded image"
    (Myneni et al. 2018, VIIRS LAI/fPAR User Guide). So, instead of using the
    reported upper-left (UL) corner coordinates, it is more accurate to use
    the lower-right (LR) corner coordinates and calculate the position of the
    UL corner based on the width and height of the image and the pixel size.
    NOTE that a rather odd pixel size is required to get the correct results
    verified against the HDF-EOS-to-GeoTIFF (HEG) Conversion Tool; see also
    Giglio et al. (2018), "Collection 6 MODIS Burned Area Product User's
    Guide, Version 1" Table 1.

        https://modis-land.gsfc.nasa.gov/pdf/MODIS_C6_BA_User_Guide_1.2.pdf

    Parameters
    ----------
    hdf : h5py.File
    ps : int or float
        The pixel size; in units matching the linear units of the SRS
        (Default: 463.3127 meters)
    nrow : int
        Number of rows in the output image (Default: 2400 for MODIS/VIIRS)
    ncol : int
        Number of columns in the output image (Default: 2400 for MODIS/VIIRS)
    metadata : re.Pattern
        Compiled regex that captures important metadata fields

    Returns
    -------
    tuple
        (x_min, pixel_width, 0, y_max, 0, -pixel_height)
    '''
    meta = hdf['HDFEOS INFORMATION/StructMetadata.0'][()].decode()
    lr = VIIRS_METADATA.match(meta).groupdict()['lr'].split(',')
    return ( # Subtract distance (meters) from LR corner to obtain UR corner
        float(lr[0]) - (ncol * ps), ps, 0, float(lr[1]) + (nrow * ps), 0, -ps)


def mod15a2h_qc_fail(x):
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparLai_QC` band: Bad pixels have either `1` in the first bit ("Pixel not
    produced at all") or anything other than `00` ("clear") in bits 3-4.
    Output array is True wherever the array fails QC criteria. Compare to:

        np.vectorize(lambda v: v[0] == 1 or v[3:5] != '00')

    Parameters
    ----------
    x : numpy.ndarray
        Array where the last axis enumerates the unpacked bits
        (ones and zeros)

    Returns
    -------
    numpy.ndarray
        Boolean array with True wherever QC criteria are failed
    '''
    y = dec2bin_unpack(x)
    # Emit 1 = FAIL if these two bits are not == "00"
    c1 = y[...,3:5].sum(axis = -1).astype(np.uint8)
    # Emit 1 = FAIL if 1st bit == 1 ("Pixel not produced at all")
    c2 = y[...,0]
    # Intermediate arrays are 1 = FAIL, 0 = PASS
    return (c1 + c2) > 0


def modis_from_wgs84(coords):
    '''
    Given longitude-latitude coordinates, return the coordinates on the
    sinusoidal projection plane.

    Parameters
    ----------
    coords : tuple or list
        (Longitude, Latitude) coordinate pair

    Returns
    -------
    tuple
        (X, Y) coordinate pair in MODIS sinusoidal projection
    '''
    x, y = map(np.deg2rad, coords)
    return (SPHERE_RADIUS * x * np.cos(y), SPHERE_RADIUS * y)


def modis_to_wgs84(coords):
    '''
    Convert coordinates on the MODIS sinusoidal plane to longitude-latitude
    coordinates (WGS84).

    Parameters
    ----------
    coords : tuple or list
        (X, Y) coordinate pair in MODIS sinusoidal projection

    Returns
    -------
    tuple
        (Longitude, Latitude) coordinate pair
    '''
    x, y = coords
    lat = y / SPHERE_RADIUS
    lng = x / (SPHERE_RADIUS * np.cos(lat))
    return tuple(map(np.rad2deg, (lng, lat)))


def modis_tile_from_wgs84(coords):
    '''
    Given longitude-latitude coordinates, return the MODIS tile (H,V) that
    contains them.

    Parameters
    ----------
    coords : tuple or list
        (Longitude, Latitude) coordinate pair

    Returns
    -------
    tuple
        (H,V) tile identifier
    '''
    x, y = modis_from_wgs84(coords) # Get coordinates in the projection plane
    return (
        np.floor((x - XMIN) / TILE_SIZE),
        np.floor((YMAX - y) / TILE_SIZE))


def modis_row_col_from_wgs84(coords, nominal = 500):
    '''
    Given longitude-latitude coordinates, return the corresponding row-column
    coordinates within a MODIS tile. NOTE: You'll need to determine which
    MODIS tile contains this row-column index with `modis_tile_from_wgs84()`.

    Parameters
    ----------
    coords : tuple or list
        (X, Y) coordinate pair in MODIS sinusoidal projection
    nominal : int
        Nominal resolution of MODIS raster: 250 (meters), 500, or 1000

    Returns
    -------
    tuple
        (Row, Column) coordinates
    '''
    x, y = modis_from_wgs84(coords) # Get coordinates in the projection plane
    # Get actual size of, e.g., "500-m" MODIS sinusoidal cell
    res = TILE_SIZE / float(TILE_LINES[nominal])
    return (
        np.floor((((YMAX - y) % TILE_SIZE) / res) - 0.5),
        np.floor((((x - XMIN) % TILE_SIZE) / res) - 0.5),
    )


def modis_row_col_to_wgs84(coords, h, v, nominal = 500):
    '''
    Convert pixel coordinates in a specific MODIS tile to longitude-latitude
    coordinates.

    Parameters
    ----------
    coords : tuple or list
        (X, Y) coordinate pair in MODIS sinusoidal projection
    h : int
        MODIS tile "h" index
    v : int
        MODIS tile "v" index
    nominal : int
        Nominal resolution of MODIS raster: 250 (meters), 500, or 1000

    Returns
    -------
    tuple
        (Longitude, Latitude) coordinates
    '''
    r, c = coords
    lines = TILE_LINES[nominal]
    assert (0 <= r <= (lines - 1)) and (0 <= c <= (lines - 1)),\
        'Row and column indices must be in the range [0, %d]' % (lines - 1)
    # Get actual size of, e.g., "500-m" MODIS sinusoidal cell
    res = TILE_SIZE / float(TILE_LINES[nominal])
    x = ((c + 0.5) * res) + (h * TILE_SIZE) + XMIN
    y = YMAX - ((r + 0.5) * res) - (v * TILE_SIZE)
    return modis_to_wgs84((x, y))


def vnp15a2h_qc_fail(x):
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparLai_QC` band: Bad pixels have either `11` in the first two bits
    ("Fill Value") or anything other than `0` in the 3rd least-significant
    bits, which combines "Pixel not produced at all". For example, see decimal
    number 80:

        0101|0|000 where "000" is the combined (Fill bit | Retrieval quality)

    Parameters
    ----------
    x : numpy.ndarray
        Unsigned, 8-bit integer array

    Returns
    -------
    numpy.ndarray
        Boolean array
    '''
    y = dec2bin_unpack(x)
    # Emit 1 = FAIL if sum("11") == 2; "BiomeType" == "Filled Value"
    c1 = np.where(y[...,0:2].sum(axis = -1) == 2, 1, 0).astype(np.uint8)
    # Emit 1 = FAIL if 3rd bit == 1; "SCF_QC" == "Pixel not produced at all"
    c2 = y[...,5]
    # Intermediate arrays are 1 = FAIL, 0 = PASS
    return (c1 + c2) > 0


def vnp15a2h_cloud_fail(x):
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparExtra_QC` band (cloud QC band): Bad pixels have anything OTHER THAN
    `1` second least-significant bit; `00` and `01` being acceptable cloud QC
    flags ("Confident clear" and "Probably clear", respectively).

    Parameters
    ----------
    x : numpy.ndarray
        Unsigned, 8-bit integer array

    Returns
    -------
    numpy.ndarray
        Boolean array
    '''
    y = dec2bin_unpack(x)
    return y[...,-2] > 0
