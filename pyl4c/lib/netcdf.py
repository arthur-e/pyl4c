'''
Utilities for working with NetCDF files.
'''

import calendar
import datetime
import re
import numpy as np
import netCDF4
from osgeo import osr
from osgeo import gdal
from osgeo import gdalconst
from bisect import bisect_left
from scipy.io import netcdf_file
from pyl4c.epsg import EPSG
from pyl4c.spatial import EASE2_GRID_PARAMS, array_to_raster, dump_raster
from pyl4c.utils import get_slice_idx_by_bbox

TIME_RX = re.compile(r'^(?P<interval>seconds|hours|days|months|years) since (?P<epoch>[\d\-]+(?:\s{1}?(?:\d{2}\:\d{2}))?(?:\:\d{2})?)')
NDAYS_RX = re.compile(r'(\d+)\_day')

def nc_dim(variable):
    '''
    Returns the dimensions of a netCDF variable.

    Parameters
    ----------
    variable : scipy.io.netcdf.netcdf_variable

    Returns
    -------
    int
    '''
    if hasattr(variable, 'ndim'):
        return variable.ndim
    if hasattr(variable, 'dimensions'):
        return len(variable.dimensions)


def netcdf_array(
        nc, keys, cell_size = None, time_idx = 0, x_offset = -180,
        scale_and_offset = True):
    '''
    Extracts an equirectangular NetCDF data array as a NumPy array, with
    spatial reference system (SRS) information. NOTE: Both `cell_size` elements
    should be positive numbers.

    Parameters
    ----------
    nc : netCDF4.Dataset
        The NetCDF file (opened)
    keys : tuple
        A tuple of (variable, x_coords, y_coords) where variable is the
        variable name, x_coords the name of the X-coordinate array variable,
        y_coords the name of the Y-coordinate array variable
    cell_size : tuple or None
        (Optional) Tuple of the the cell size in both directions in degrees,
        e.g., (1, 1); will attempt to infer the cell size if not provided
    time_idx : int
        (Optional) Index of the "time" axis to extract as a raster
        (Default: 0)
    x_offset : int or float
        (Optional) Because of the absurd X-coordinates convention used in some
        NetCDF arrays, it may be necessary to "offset" the X coordinates; this
        is only used if it is detected that the maximum X coordinate value is
        >/= 180 degrees E (Default: -180);
    scale_and_offset : bool
        (Optional) True to apply the scale and offset to the data, if the
        attributes "scale_factor" and "add_offset" are found (Default); False
        to do nothing

    Returns
    -------
    tuple
        Tuple of `(array, gt, wkt)`
    '''
    def infer_cell_size(x_coords, y_coords):
        assert x_coords.ndim == 1,\
            "Can't infer cell size from coordinate arrays with more than one axis"
        # NOTE: Rounding coordinate arrays to 5 decimal places for comparison
        x_diff = np.round(np.array(x_coords)[1:] - np.array(x_coords)[0:-1], 4)
        y_diff = np.round(np.array(y_coords)[1:] - np.array(y_coords)[0:-1], 4)
        assert np.median(x_diff) == x_diff[0] and np.median(y_diff) == y_diff[0],\
            "Array cell size is not equal in both directions!"
        return (np.abs(np.median(x_diff)), np.abs(np.median(y_diff)))

    # Original (equirectangular) spatial reference system
    wkt0 = osr.SpatialReference()
    wkt0.ImportFromEPSG(4326)
    if cell_size is not None:
        x_res, y_res = cell_size
        assert x_res > 0 and y_res > 0, 'Argument cell_size must be greater than zero'
    else:
        x_res, y_res = infer_cell_size(
            np.array(nc.variables[keys[1]][:]),
            np.array(nc.variables[keys[2]][:]))

    x0 = np.array(nc.variables[keys[1]][:]).min()
    y0 = np.array(nc.variables[keys[2]][:]).max()
    # Geotransform params: (x_min, pixel_width, 0, y_max, 0, -pixel_height)
    gt0 = (x0, x_res, 0, y0, 0, -y_res)

    # NOTE: Because some NetCDF files have a "degrees east" convention with
    #   longitude increasing monotonically from 0 to 360, we must apply an
    #   offset to the X coordinates
    start_idx = None
    if np.array(nc.variables[keys[1]][:]).max() > 180:
        # Update affine transformation
        gt0 = (x0 + x_offset, x_res, 0, y0, 0, -y_res)
        # Transform to proper longitudes
        x_coords = (nc.variables[keys[1]][:] + x_offset)

        # Get X-coordinate origin, or index of east longitude closest to zero,
        #   unless all coordinates are in a single hemisphere
        test = np.logical_and(
            x_coords > 0, x_coords <= np.abs(x_coords).min())
        if np.any(test):
            idx = np.arange(0, x_coords.shape[0])
            start_idx = int(idx[x_coords == x_coords[test]])

    # Extract a NumPy array
    if hasattr(time_idx, 'index'):
        a, b = (min(time_idx), max(time_idx))
        arr = np.array(nc.variables[keys[0]][a:(b+1),...])
    else:
        # Make sure a single time slice as a band axis
        if nc_dim(nc.variables[keys[0]]) > 2:
            arr = np.array(nc.variables[keys[0]][time_idx,...])[np.newaxis,...]
        else:
            # Some netCDF variables may have no time/ band axis (i.e., static)
            arr = np.array(nc.variables[keys[0]][:])[np.newaxis,...]

    # If Y coordinates are in ascending order, flip the array
    lats = nc.variables[keys[2]][:].tolist()
    if lats.index(max(lats)) > lats.index(min(lats)):
        arr = np.flip(arr, axis = 1)

    # If necessary, re-sort data so west-most longitude is the first column
    if start_idx is not None:
        arr = np.concatenate((arr[...,start_idx:], arr[...,:start_idx]), axis = 2)

    scale = 1
    if scale_and_offset and hasattr(nc.variables[keys[0]], 'scale_factor'):
        scale = nc.variables[keys[0]].scale_factor

    offset = 0
    if scale_and_offset and hasattr(nc.variables[keys[0]], 'add_offset'):
        offset = nc.variables[keys[0]].add_offset

    return ((arr * scale) + offset, gt0, wkt0.ExportToWkt())


def netcdf_raster(
        nc, keys, cell_size = None, time_idx = 0, subset_bbox = None,
        in_nodata = None, out_nodata = -9999):
    '''
    Dumps a NetCDF variable to a GeoTIFF file. NOTE: Both `cell_size` elements
    should be positive numbers.

    Parameters
    ----------
    nc : netCDF4.Dataset
        The NetCDF file (opened)
    keys : tuple
        A tuple of (variable, x_coords, y_coords) where variable is the
        variable name, x_coords the name of the X-coordinate array variable,
        y_coords the name of the Y-coordinate array variable
    cell_size : tuple or None
        (Optional) Tuple of the the cell size in both directions in degrees,
        e.g., (1, 1); will attempt to infer the cell size if not provided
    time_idx : int
        (Optional) Index of the "time" axis to extract as a raster
        (Default: 0)
    subset_bbox : tuple or None
        (Optional) Subset bounding box
    in_nodata : int or float or None
        (Optional) The NoData value to ignore in the input
    out_nodata : int or float or None
        (Optional) The NoData value to set in the output (Default: -9999)

    Returns
    -------
    gdal.Dataset
    '''
    arr, gt0, wkt0 = netcdf_array(nc, keys, cell_size, time_idx)
    xmin = ymin = 0
    if subset_bbox is not None:
        x_coords = nc.variables[keys[1]][:]
        y_coords = nc.variables[keys[2]][:]
        if np.array(nc.variables[keys[1]][:]).max() > 180:
            # Correct for weird NetCDF easting
            x_coords = nc.variables[keys[1]][:] - 180
        x_idx, y_idx = get_slice_idx_by_bbox(
            x_coords, y_coords, subset_bbox = subset_bbox)
        xmin, xmax = x_idx
        ymin, ymax = y_idx
        if arr.ndim == 2:
            arr = arr[ymin:ymax, xmin:xmax]
        else:
            arr = arr[:, ymin:ymax, xmin:xmax]

    # Optional: Mask user-specified NoData value
    if in_nodata is not None:
        arr = np.where(arr == in_nodata, np.nan, arr)

    # Mask any NoData/ fill value defined by the dataset
    if hasattr(nc.variables[keys[0]], 'missing_value'):
        arr = np.where(
            arr == nc.variables[keys[0]].missing_value, np.nan, arr)

    # Create a gdal.Dataset from the array
    if out_nodata is not None:
        return array_to_raster(
            np.where(
                np.isnan(arr), out_nodata, arr), gt0, str(wkt0), xmin, ymin)

    return array_to_raster(arr, gt0, str(wkt0), xmin, ymin)


def parse_time_units(netcdf_time_axis):
    '''
    Returns the time interval size (e.g., "days") and the epoch
    (e.g., "1970-01-01") from a given netCDF units declaration for the time
    axis (e.g., "days since 1970-01-01").

    Parameters
    ----------
    netcdf_time_axis : netCDF4._netCDF4.Variable
        Variable that describes the time axis

    Returns
    -------
    tuple
        Tuple of (str, datetime.datetime)
    '''
    units_string = netcdf_time_axis.units
    if isinstance(units_string, bytes):
        units_string = bytes(units_string).decode()

    interval, epoch_str = TIME_RX.match(units_string).groups()
    epoch_str = epoch_str.replace('-1-1', '-01-01')
    if epoch_str.rfind('00:00:00') > 0:
        epoch = datetime.datetime.strptime(epoch_str, '%Y-%m-%d %H:%M:%S')
    elif epoch_str.rfind('00:00') > 0:
        epoch = datetime.datetime.strptime(epoch_str, '%Y-%m-%d %H:%M')
    else:
        epoch = datetime.datetime.strptime(epoch_str, '%Y-%m-%d')
    return (interval, epoch)


def spatial_average(
        nc, keys, reducer, t_labels = None, subset_bbox = None, nodata = -9999):
    '''
    Calculates a spatial average across a NetCDF time series variable.

    Parameters
    ----------
    nc : netCDF4.Dataset
        The NetCDF file (opened)
    keys : tuple
        A tuple of (variable, x_coords, y_coords) where variable is the
        variable name, x_coords the name of the X-coordinate array variable,
        y_coords the name of the Y-coordinate array variable
    reducer : function
        A vectorized function that returns a scalar for an input vector of
        values; should ignore np.nan values
    t_labels : tuple or list
        (Optional) A vector of time index labels
    subset_bbox : tuple
        (Optional) To specify a subset area for the spatial average (i.e.,
        instead of averaging over the entire NetCDF array domain), pass a
        bounding box argument
    nodata : int or float
        The NoData value to ignore (Default: -9999)

    Returns
    -------
    list
        Of the form `[(index, value), ...]` for each time step
    '''
    if subset_bbox is not None:
        x_coords = nc.variables[keys[1]][:]
        y_coords = nc.variables[keys[2]][:]
        if np.array(nc.variables[keys[1]][:]).max() > 180:
            # Correct for weird NetCDF easting
            x_coords = nc.variables[keys[1]][:] - 180
        x_idx, y_idx = get_slice_idx_by_bbox(
            x_coords, y_coords, subset_bbox = subset_bbox)
        xmin, xmax = x_idx
        ymin, ymax = y_idx

    scale = 1
    if hasattr(nc.variables[keys[0]], 'scale_factor'):
        scale = nc.variables[keys[0]].scale_factor

    offset = 0
    if hasattr(nc.variables[keys[0]], 'add_offset'):
        offset = nc.variables[keys[0]].add_offset

    num_epochs = nc.variables[keys[0]].shape[0]
    time_series = []
    for i, t in enumerate(range(0, num_epochs)):
        if subset_bbox is not None:
            arr = nc.variables[keys[0]][:][t, ymin:ymax, xmin:xmax]

        else:
            arr = nc.variables[keys[0]][:][t,...]

        index = t
        if t_labels is not None:
            index = t_labels[i]

        time_series.append((
            # Filter out NoData; use summary function that ignores NaNs
            index, reducer( # Multiply by scale, add offset
                np.where(arr == nodata, np.nan, (arr * scale) + offset))
        ))

    return time_series


def time_series(netcdf_time_axis, has_leap = None):
    '''
    Constructs a series of datetime.datetime elements based on a netCDF
    time axis; assumes that time steps are evenly spaced (except in leap
    years) and ignores fractional time steps.

    NOTE: This is EXTREMELY difficult to get right because of awful
    implementation of time axes in the typical netCDF dataset. Among other
    challenges... Apparently, netCDF time axes use the convention that if the
    epoch starts at midnight, the first day is the "zeroth" day; e.g., CLM5.0
    states its time axis is "days since 1700-01-01 00:00:00" but the first
    time step is 31.0; it seems highly unlike this monthly time series would
    start in February (Jan. 1 + 31 days == Feb. 1), so they must be using the
    *last day* of each month (e.g., Jan 0 + 31 days == Jan. 31).

    Parameters
    ----------
    netcdf_time_axis : netCDF4._netCDF4.Variable
        Variable that describes the time axis
    has_leap : bool or None
        Set True to force recognition of leap years; if None, will attempt
        to automatically determine whether leap years should be recognized
        based on the field metadata (Default: None)

    Returns
    -------
    list
        `[*datetime.date|datetime.datetime]`
    '''
    steps = netcdf_time_axis[:]
    interval, epoch = parse_time_units(netcdf_time_axis)
    convention = getattr(netcdf_time_axis, 'calendar', '')
    if hasattr(convention, 'decode'):
        convention = convention.decode('utf-8')

    # Attempt to determine leap year recognition
    if has_leap is None:
        has_leap = convention in ('', 'standard', 'gregorian')

    first_step = steps[1] - steps[0]
    # Basically, we can tolerate different step sizes for intervals that
    #   can be passed to datetime.timedelta, which assumes leap years;
    #   in all other cases, we can't calculate time steps correctly
    if not has_leap:
        if not np.all(np.array(steps[1:]) - np.array(steps[:-1]) == first_step):
            assert interval in ('seconds', 'hours', 'days'),\
                'Time series is not evenly spaced and leap years do not explain this discrepancy'

    ldays = 1 if has_leap else 0
    ndays = 365
    if NDAYS_RX.match(convention) is not None:
        # Some datasets oddly have, e.g., "360_day" conventions
        ndays = int(NDAYS_RX.match(convention).groups()[0])

    # Because timedelta can't process intervals higher than "days"...
    if interval == 'years':
        # Simple years series starting January 1 if units are years
        years = np.arange(epoch.year, epoch.year + steps[-1] + 1)
        time_axis = [datetime.date(int(y), 1, 1) for y in years]
    elif interval == 'months':
        time_axis = []
        m_range = (np.arange(0, len(steps)) + epoch.month) % 12
        m_range = np.where(m_range == 0, 12, m_range)
        for t, t_value in enumerate(steps):
            m = m_range[t]
            y = epoch.year + np.floor(t_value / 12)
            time_axis.append(datetime.date(int(y), int(m), epoch.day))
    # But if time intervals are in "seconds," "hours," or "days"
    elif interval in ('seconds', 'hours', 'days') and has_leap:
        time_axis = [
            epoch +
                datetime.timedelta(**{interval: int(t)}) for t in steps
        ]
    # But timedelta assumes leap years are in use, so if the time series
    #   doesn't use leap years, we must construct it manually
    else:
        time_axis = []
        this_year = epoch.year
        mdays = np.cumsum(calendar.mdays)
        for t in steps:
            if interval == 'seconds':
                y = epoch.year + (t / (ndays * 24 * 60 * 60))
            if interval == 'hours':
                y = epoch.year + (t / (ndays * 24))
            if interval in ('seconds', 'hours'):
                m = bisect_left(mdays, ((y - epoch.year) * ndays) % ndays)
                d = (((y - epoch.year) * ndays) % ndays) - mdays[m - 1]
            if interval == 'days':
                y = epoch.year + (t / ndays) # e.g., t / 365
                # Incidentally, the left "sorting index" is the month number
                m = bisect_left(mdays, t % ndays)
                m += 1 if m == 0 else 0
                d = (t % ndays) - mdays[m - 1] # And days are what is left over
                d += 1 if d == 0 else 0
            time_axis.append(datetime.datetime(int(y), int(m), int(d)))
    return time_axis
