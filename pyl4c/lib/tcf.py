'''
Utilities related to the TCF code developed by Joe Glassy, Lucas Jones. Use:

    $ python pyl4c/lib/tcf.py run /anx_lagr3/laj/smap/natv72/prelaunch/land/Y2000/D*
        --output_dir="/home/user/"


TODO:

- Support for reading spatial subsets; a kind of reverse sparse row-column
    mapping, e.g.:

        np.repeat(np.arange(0, 3856).reshape((1, 3856)), 1624, axis = 0)

Ideas for an alternative sparse array subsetting that can be done without
storing a large ancillary array:

- Count distance, in elements, between valid cells in the raveled array;
  the number of stored distances increases with the sparseness of the
  array; 0 indicates the adjacent cell (raveled) is valid.
'''

import datetime
import glob
import os
import re
import h5py
import numpy as np
from itertools import product
from cached_property import cached_property
from pyl4c.data.fixtures import ANCILLARY_DATA_PATHS, HDF_PATHS, EASE2_GRID_PARAMS
from pyl4c.utils import summarize, summarize_by_class

FIELD_MAP = { # Mapping TCF field names to L4C Ops field names
    'Ctot': HDF_PATHS['SPL4CMDL']['4']['SOC'],
    'Ctot*': HDF_PATHS['SPL4CMDL']['4']['SOC*'],
    'gpp': HDF_PATHS['SPL4CMDL']['4']['GPP'],
    'gpp*': HDF_PATHS['SPL4CMDL']['4']['GPP*'],
    'nee': HDF_PATHS['SPL4CMDL']['4']['NEE'],
    'nee*': HDF_PATHS['SPL4CMDL']['4']['NEE*'],
    'rhtot': HDF_PATHS['SPL4CMDL']['4']['RH'],
    'rhtot*': HDF_PATHS['SPL4CMDL']['4']['RH*'],
    'Tmult': 'EC/tmult_mean',
    'Wmult': 'EC/wmult_mean',
    'Emult': 'EC/emult_mean'
}

TYPE_MAP = {
    'flt32': np.float32,
    'flt16': np.float16,
    'int16': np.int16,
    'uint16': np.uint16,
    'int8': np.int8,
    'uint8': np.uint8
}

REV_FIELD_MAP = dict([(v, k) for k, v in FIELD_MAP.items()])
REV_TYPE_MAP = dict([(v, k) for k, v in TYPE_MAP.items()])


class MetaSparseArray(type):
    '''
    NOTE: These are potentially expensive, but necessary, operations; they are
    exposed at the global scope because I don't want successive instances of
    `SparseArray()` to do this lookup, and allocate more memory, each time.
    '''
    @property
    def sparse_row_idx(cls):
        if getattr(cls, '_sparse_row_idx', None) is None:
            cls._sparse_row_idx = np.fromfile(ANCILLARY_DATA_PATHS['smap_l4c_9km_sparse_row_index'],
                dtype = np.uint16)

        return cls._sparse_row_idx

    @property
    def sparse_col_idx(cls):
        if getattr(cls, '_sparse_col_idx', None) is None:
            cls._sparse_col_idx = np.fromfile(ANCILLARY_DATA_PATHS['smap_l4c_9km_sparse_col_index'],
                dtype = np.uint16)

        return cls._sparse_col_idx


class SparseArray(object, metaclass = MetaSparseArray):
    '''
    Represents a "sparse" or "land" array such as that output from the
    TCF C codebase.

    Parameters
    ----------
    source : str or numpy.ndarray
        Either a file system path to a stored output array in "sparse" form
        (assumes that the filename incorporates the data type, e.g.,
        `foo.flt32` or `foo.uint8`) or an extant `numpy.ndarray` (either
        sparse or not).
    grid : str
        The EASE-Grid 2.0 designation, e.g., "M09" for a 9-km grid or "M01"
        for a 1-km grid
    dtype : type
        The NumPy data type used to represent numbers (Default: None); must
        be specified when reading a NumPy or HDF5 array
    field : str
        Name of the HDF5 field to read
    '''
    def __init__(self, source, grid, dtype = None, field = None):
        self._dtype = dtype
        self._grid = grid
        self._field = field
        self._nodata = None

        # Incorporate information from the filename
        if isinstance(source, str):
            self.filename = source
            self._deflated = True
            self._ftype = self.filename.split('.')[-1]
            # Attempt to read the array data
            assert os.path.exists(source), 'File not found: %s' % source
            assert self._ftype in TYPE_MAP.keys(), 'Unrecognized file extension; could not determine data type'
            self.data = np.fromfile(source, dtype = TYPE_MAP[self._ftype])
        elif isinstance(source, np.ndarray):
            self.filename = None
            self._deflated = source.ndim == 1
            self._ftype = None
            assert self._dtype is not None,\
                'Must indicate the data type when reading an array!'
            assert source.ndim <= 2, 'No support for more than 2 axes'
            self.data = source
        elif isinstance(source, h5py.File):
            # NOTE: Trying to deflate the HDF5 array while reading it is
            #   inefficient because it requires looping over multiple index
            #   operations; it will never be faster than deflating the entire
            #   array one time
            assert self._field is not None, "__init__() missing 1 required positional argument: 'field'"
            self.filename = source.filename
            self._ftype = None
            self.data = source[self._field][:]
            self._deflated = self.data.ndim == 1
            assert self._dtype is not None,\
                'Must indicate the data type when reading an array!'
        else:
            raise TypeError('Must provide either a binary file path or a numpy.ndarray instance')

    def __nested_src_idx__(self, x, y, ny):
        # Serialization offset for nested grid elements, e.g., 0 through 8
        #   for a 1-km grid nested in a 9-km grid
        return (x * ny) + y

    def __nested_dst_idx__(self, x, y, m, n, nn, ny):
        # Serialization offset for a grid nested within a 9-km grid
        # x, y correspond to 9-km grid indices; m, n to nested grid indices;
        #   nn is number of nested columns (e.g., 9 for 1-km grid) and ny
        #   is the number of columns in the overall 2D grid (e.g., 34704)
        return (((x * nn) + m) * ny) + (y * nn) + n

    def __inflate_1km_grid__(self, nodata = -9999):
        # This is SLOW; it shouldn't be used in production, but just in case
        #   a pure Python version of mkgrid for 1-km grids is ever needed...
        shp = EASE2_GRID_PARAMS[self._grid]['shape']
        # NOTE: 1-km destination array starts raveled
        result = np.ones((shp[0] * shp[1],)) * nodata
        # Source array has first 81 pixels, then second 81 pixels, etc...
        nested_grid = product(range(0, 9), range(0, 9))
        for i in range(0, self.__class__.sparse_row_idx.shape[0]):
            row = self.__class__.sparse_row_idx[i]
            col = self.__class__.sparse_col_idx[i]
            for m, n in nested_grid:
                # This is a translation of the function from spland.c
                #   in the l4c-utils/mkgrid C library
                src = self.__nested_src_idx__(
                    i, self.__nested_src_idx__(m, n, 9), 81)
                dst = self.__nested_dst_idx__(row, col, m, n, 9, shp[1])
                result[dst] = self.data[src]

    @property
    def shape(self):
        "Alias for the array data's shape"
        return self.data.shape

    @classmethod
    def get_deflated_idx(cls, coords):
        '''
        Translates (row, column) coordinates in the inflated array to the
        corresponding 1D index in the deflated (sparse) array.

        Parameters
        ----------
        coords : tuple or list
            Sequence of (row, column) integer indices

        Returns
        -------
        numpy.ndarray
        '''
        row, col = coords
        return np.arange(0, cls.sparse_row_idx.shape[0])[
            np.logical_and(cls.sparse_col_idx == col, cls.sparse_row_idx == row)
        ]

    @classmethod
    def sparse_selector(cls, slice_idx, grid = 'M09'):
        '''
        Returns a 1D array with values in `[np.nan, 1]` where 1 is assigned to
        those cells inside the desired spatial subset of the corresponding
        2D array. This allows efficient summarization, for instance, of the
        sparse (1D) data array without having to inflate it.

        Parameters
        ----------
        slice_idx : tuple or list
            Sequence of nested bounding box coordinates in pixel space, e.g.:
            `((xmin, xmax), (ymin, ymax))`
        grid : str
            EASE-Grid 2.0 designation describing the size of the grid that
            should be restored, e.g., "M09" or "M01"

        Returns
        -------
        numpy.ndarray
        '''
        shp = EASE2_GRID_PARAMS[grid]['shape']
        x_idx, y_idx = slice_idx
        xmin, xmax = x_idx
        ymin, ymax = y_idx

        # If the array is already deflated, we need to generate a
        #   corresponding 1D array to "select" values
        # Get a grid of NaNs, then fill in 1 where the subset fits
        selector_deflated = np.ones(cls.sparse_row_idx.shape) * np.nan
        selector = np.ones(shp) * np.nan
        selector[ymin:ymax, xmin:xmax] = 1
        # Exactly like self.inflate() but selector is the "data'"
        for i in range(0, cls.sparse_row_idx.shape[0]):
            row = cls.sparse_row_idx[i]
            col = cls.sparse_col_idx[i]
            selector_deflated[i] = selector[row, col]

        return selector_deflated

    @classmethod
    def ordinal_to_date(cls, year, ordinal):
        '''
        Returns a `datetime.date()` instance based on year, ordinal day.

        Parameters
        ----------
        year : int
        ordinal : int
            Day of the year on interval [1, 366]

        Returns
        -------
        datetime.date
        '''
        assert ordinal <= 366, 'Ordinal out of range [1, 366]'
        assert year in range(1800, 2100), 'Year out of range [1800, 2100]'
        # In leap years, for days after February 29, subtract 1 so that
        #   we obtain the "regular" date
        if year % 4 == 0:
            if ordinal == 60:
                return datetime.date(year, 2, 29)
                if ordinal > 60:
                    ordinal -= 1
                    d = datetime.date.fromordinal(ordinal)
                    return datetime.date(year, d.month, d.day)

    def deflate(self, nodata = -9999):
        '''
        Inverse of `SparseArray.inflate()`; converts a 2D array to a "sparse"
        or "land" representation.

        Parameters
        ----------
        nodata : int or float
            The NoData value to output in resulting array
        '''
        # If the array is already deflated but a different NoData value was
        #   requested, update that value
        if self._deflated:
            if self._nodata is not None and nodata != self._nodata:
                self.data = np.where(
                    self.data == self._nodata, nodata, self.data)
            else:
                return None # Otherwise, return nothing

        # Initialize an empty array (fill with NoData)
        result = np.ones(
            self.__class__.sparse_row_idx.shape, self._dtype) * nodata
        for i in range(0, self.__class__.sparse_row_idx.shape[0]):
            row = self.__class__.sparse_row_idx[i]
            col = self.__class__.sparse_col_idx[i]
            result[i] = self.data[row, col]
        self._deflated = True
        self.data = result
        result = None

    def inflate(self, nodata = -9999):
        '''
        Converts from a "sparse" or "land" array to an EASE-Grid 2.0 array. The
        sparse format was developed by Joe Glassy and Lucas Jones. Special files
        are needed to index the sparse data correctly (it is not simply a raveled
        array filtered to land pixels).

        Parameters
        ----------
        nodata : int or float
            The NoData value to output in resulting array
        '''
        # If the array is already inflated but a different NoData value was
        #   requested, update that value
        if not self._deflated:
            if self._nodata is not None and nodata != self._nodata:
                self.data = np.where(
                    self.data == self._nodata, nodata, self.data)
            else:
                return None # Otherwise, return nothing

        # Initialize an empty array (fill with NoData)
        shp = EASE2_GRID_PARAMS[self._grid]['shape']
        result = np.ones(shp, self._dtype) * nodata
        for i in range(0, self.__class__.sparse_row_idx.shape[0]):
            row = self.__class__.sparse_row_idx[i]
            col = self.__class__.sparse_col_idx[i]
            result[row, col] = self.data[i]
        self._deflated = False
        self.data = result
        result = None

    def inflate_subset(self, slice_idx, nodata = -9999):
        '''
        Returns a subset view of the inflated array, for convenience. This is
        not faster than subsetting the inflated array.

        Parameters
        ----------
        slice_idx : tuple or list
            `((xmin, xmax), (ymin, ymax))` in pixel space
        nodata : int or float
            The NoData value to output in resulting array

        Returns
        -------
        numpy.ndarray
        '''
        x_idx, y_idx = slice_idx
        if not self._deflated:
            return self.data[y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]]

        selector = self.__class__.sparse_selector(slice_idx, self._grid)
        xs = x_idx[1] - x_idx[0] # Get the output array size
        ys = y_idx[1] - y_idx[0]
        shp = (ys, xs)
        data = np.multiply(self.data, selector)
        # Initialize an empty array (fill with NoData)
        result = np.ones(shp) * nodata
        for i in range(0, self.__class__.sparse_row_idx.shape[0]):
            row = self.__class__.sparse_row_idx[i]
            col = self.__class__.sparse_col_idx[i]
            if row < y_idx[1] and col < x_idx[1]:
                row -= y_idx[0]
                col -= x_idx[0]
                if row >= 0 and col >= 0:
                    result[row, col] = data[i]
        return result

    def replace(self, old, new):
        '''
        Replace "old" values with "new" values, e.g., replace old NoData
        values with a new value.

        Parameters
        ----------
        old : int or float
            Value to be replaced
        new : int or float
            Value to replace the old value
        '''
        self.data[self.data == old] = new

    def summarize(self, selector = None, **kwargs):
        '''
        Summarize the values of an array (inflated or deflated) without
        inflating it, optionally for a spatial subset.

        Parameters
        ----------
        selector : numpy.ndarray
            An array that describes which which values should be summarized;
            all non-NaN entries correspond to entries in the data array that
            will be summarized.
        **kwargs
            Additional arguments to `pyl4c.utils.summarize()` or
            `pyl4c.utils.summarize_by_class()`

        Returns
        -------
        dict
        '''
        shp = EASE2_GRID_PARAMS[self._grid]['shape']
        ymin = xmin = 0 # Default is full extent of grid
        ymax, xmax = shp
        if not self._deflated:
            return summarize(self.data[ymin:ymax, xmin:xmax], **kwargs)
        if selector is None:
            selector = np.ones(self.shape)
        # Selector multiplies NaN against areas outside the subset
        return summarize(np.multiply(self.data, selector), **kwargs)

    def summarize_by_class(
            self, selector = None, class_array = None, **kwargs):
        '''
        Summarize the values of an array (inflated or deflated) without
        inflating it, optionally for a spatial subset.

        Parameters
        ----------
        selector : numpy.ndarray
            An array that describes which which values should be summarized;
            all non-NaN entries correspond to entries in the data array that
            will be summarize
        class_array : numpy.ndarray
            An array that has discrete class labels (e.g., decimal numbers);
            values that share a class will be aggregated in the summary
        **kwargs
            Additional arguments to `pyl4c.utils.summarize()` or
            `pyl4c.utils.summarize_by_class()`

        Returns
        -------
        dict
        '''
        assert class_array.shape == self.shape, 'class_array does not have the same shape!'
        shp = EASE2_GRID_PARAMS[self._grid]['shape']
        ymin = xmin = 0 # Default is full extent of grid
        ymax, xmax = shp
        if not self._deflated:
            return summarize_by_class(
                self.data[ymin:ymax, xmin:xmax], class_array, **kwargs)
        if selector is None:
            selector = np.ones(self.shape)
        # Selector multiplies NaN against areas outside the subset
        return summarize_by_class(np.multiply(self.data, selector),
            np.multiply(class_array, selector), **kwargs)

    def timestamp(self):
        '''
        Returns the ISO 8601 timestamp of the file.

        Returns
        -------
        datetime.date
        '''
        cls = self.__class__
        match = cls.date_regex.match(os.path.dirname(self.filename))
        if match is None:
            return match

        return cls.ordinal_to_date(*map(int, match.groups()))


class TCFArray(SparseArray, metaclass = MetaSparseArray):
    '''
    Represents a "sparse" or "land" array, output from the TCF C codebase.

    Parameters
    ----------
    file_path : str
        File system path to a stored output TCF array in "sparse" form;
        assumes that the filename incorporates the data type, e.g.,
        `foo.flt32` or `foo.uint8`.
    '''
    file_regex = re.compile(r'^tcf_.*_(?P<field>Ctot|gpp|nee|rhtot)_avg_(?P<pft>pft\d{1})?_?.*\.(?P<ftype>%s$)' % '|'.join(TYPE_MAP.keys()))
    date_regex = re.compile(r'.*/Y(?P<year>\d{4})/D(?P<day>\d{3})/?$')

    def __init__(self, file_path, grid = 'M09'):
        super(TCFArray, self).__init__(file_path, grid)
        match = self.file_regex.match(os.path.basename(self.filename))
        self._field, self._pft, self._ftype = match.groups()
        self._field_l4c = FIELD_MAP[self._field] # L4C field name
