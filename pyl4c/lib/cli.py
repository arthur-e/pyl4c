'''
Classes for building command line interface (CLI) tools, such as for bulk
processing data at the command line.
'''

import os
import sys
import h5py
import numpy as np
from osgeo import gdal
from cached_property import cached_property
from pyl4c.data.fixtures import EASE2_GRID_PARAMS
from pyl4c.epsg import EPSG
from pyl4c.spatial import array_to_raster, as_array, ease2_coords_approx
from pyl4c.utils import get_ease2_coords, get_ease2_slice_idx
from pyl4c.lib.tcf import TYPE_MAP, SparseArray, TCFArray

class CommandLineInterface(object):
    '''
    A command line interface (CLI) convenience class used for creating
    Python scripts that can be invoked from the command line.
    '''

    def __init__(self):
        pass

    def __check__(self):
        possible_required_keys = ('_output_path', '_output_tpl')
        for key in possible_required_keys:
            if hasattr(self, key):
                assert getattr(self, key) is not None, 'You must specify %s with: --%s=""' % (key, key)

        if hasattr(self, '_field_tpl'):
            assert self._field_tpl.rfind('%d') > 0, 'The field argument must contain a string formatting character, e.g., "SOC/soc_pft%d_mean"'

        if hasattr(self, '_grid'):
            assert self._grid is not None, 'Must specify the EASE-Grid 2.0 size with --grid argument ("M01" or "M09")'

        if hasattr(self, '_mask'):
            if self._mask is not None:
                assert isinstance(self._mask, str), 'Did not recognize --mask as a file path'
                assert os.path.exists(self._mask), 'File not found: %s (Cannot use shortcuts like ~)' % self._mask

        if hasattr(self, '_output_dir'):
            assert os.path.isdir(self._output_dir), 'Directory does not exist: %s' % self._output_dir

        if hasattr(self, '_output_path'):
            assert os.path.exists(os.path.dirname(self._output_path)), 'Did not recognize output_path (Cannot use shortcuts like ~)'

        if hasattr(self, '_output_tpl'):
            assert self._output_tpl.rfind('%s') >= 0, 'The output_tpl argument must have one string formatting character'

        if hasattr(self, '_reference'):
            if self._reference is not None:
                assert isinstance(self._reference, str), 'Did not recognize --reference as a file path'
                assert os.path.exists(self._reference), 'File not found: %s (Cannot use shortcuts like ~)' % self._reference

        if hasattr(self, '_summaries'):
            assert not isinstance(self._summaries, str), 'Could not interpret --summaries argument as a sequence of NumPy functions'

    @cached_property
    def __coords__(self):
        if self._grid in ('M01', 'M09'):
            return get_ease2_coords(self._grid)

        # But if coordinate arrays are not pre-computed...
        return ease2_coords_approx(self._grid)

    @cached_property
    def __shp__(self):
        if self._subset_id is not None:
            x_idx, y_idx = self.__slice_idx__
            return (y_idx[1] - y_idx[0], x_idx[1] - x_idx[0])

        x_coords, y_coords = self.__coords__
        return (len(y_coords), len(x_coords))

    @cached_property
    def __slice_idx__(self):
        # The ((xmin, xmax), (ymin, ymax)) indices in pixel space
        if self._subset_id is not None:
            return get_ease2_slice_idx(
                grid = self._grid, subset_id = self._subset_id)

        return (None, None)

    def __pieces__(self):
        # Returns the next row chunk
        p = 0 # Initialize at piece 0
        start = 0 # Starting at first row, if no subsetting
        if self._subset_id is not None:
            x_idx, y_idx = self.__slice_idx__
            start = y_idx[0] # Starting at top of subset

        num_pieces = self._pieces
        num_rows = self.__shp__[0]
        step = int(np.ceil(num_rows / num_pieces))
        while p < num_pieces:
            yield (start + (p * step), start + ((p+1) * step))
            p += 1

    def gdt_to_dtype(self, gdt):
        return {
            gdal.GDT_Float32: np.float32,
            gdal.GDT_Float64: np.float64,
            gdal.GDT_Int16: np.int16,
            gdal.GDT_Int32: np.int32
        }[gdt]

    def lookup_dtype(self, type_string):
        '''
        Given, e.g., "float32", returns `numpy.float32`.

        Parameters
        ----------
        type_string : str
            A NumPy named type, e.g., "float32", "int16", "byte"
        '''
        return getattr(np, type_string)

    def lookup_gdt(self, type_string):
        '''
        Given, e.g., "float32", returns `gdal.GDT_Float32`.

        Parameters
        ----------
        type_string : str
            A NumPy named type, e.g., "float32", "int16", "byte"
        '''
        return getattr(
            gdal, 'GDT_%s' % type_string.title() if type_string != 'uint8' else 'GDT_Byte')

    def infer_file_mode(self, file_paths):
        '''
        Determine what class of file we're working with.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        str
            One of: "hdf5", "sparse", "other"
        '''
        # Check if a list/tuple given versus a character sequence
        path = file_paths[0] if len(file_paths[0]) > 1 else file_paths
        if path.split('.')[-1] == 'h5':
            return 'hdf5'
        elif path.split('.')[-1] in TYPE_MAP.keys():
            return 'sparse'
        else:
            return 'other'

    def read_array(self, file_path, **kwargs):
        '''
        Reads in the file at the given file path and returns a
        `numpy.ndarray`.

        Parameters
        ----------
        file_path : str
            The location of a file
        start_row : int
            Index of the first row in a row chunk
        end_row : int
            Index of the final row in a row chunk
        mode : str
            The file mode to use
        shp : tuple
            The shape of the raster arrays

        Returns
        -------
        numpy.ndarray
        '''
        if 'start_row' in kwargs.keys() and 'end_row' in kwargs.keys():
            if kwargs['start_row'] is not None and kwargs['end_row'] is not None:
                return self.read_chunked(file_path, **kwargs)
        if 'shp' in kwargs.keys():
            shp = kwargs['shp']
        else:
            shp = None

        mode = self._mode
        if 'mode' in kwargs.keys():
            mode = kwargs['mode']
        if self._subset_id is not None:
            x_idx, y_idx = self.__slice_idx__
            xmin, xmax = x_idx
            ymin, ymax = y_idx
        # If we're reading HDF5 files...
        if mode == 'hdf5':
            with h5py.File(file_path, 'r') as hdf:
                assert self._field in hdf.keys(), 'Could not find the specified field name: %s' % self._field
                # If we're not subsetting...
                if self._subset_id is None:
                    return hdf[self._field][:]
                return hdf[self._field][ymin:ymax, xmin:xmax]
        # If we're reading sparse TCF output arrays...
        elif mode in ('sparse', 'tcf'):
            # TCFArray has stronger assumptions than SparseArray; try to read
            #   the file either way
            tcf = SparseArray(file_path, self._grid)
            tcf.inflate()
            if self._subset_id is None:
                return tcf.data[:]
            return tcf.data[ymin:ymax, xmin:xmax]
        # If we're reading arbitrary raster arrays...
        elif mode == 'other':
            if self._subset_id is not None:
                raise NotImplementedError('No way of compositing subsets on arbitrary raster arrays with unknown shape')
            return as_array(file_path, False)[0]
        else:
            raise NotImplementedError('File mode "%s" not recognized' % mode)

    def read_chunked(
            self, file_path, start_row, end_row, mode = None, shp = None):
        '''
        Reads in the file at the given file path, row chunk by row chunk,
        and returns a `numpy.ndarray`.

        Parameters
        ----------
        file_path : str
            The location of a file
        start_row : int
            Index of the first row in a row chunk
        end_row : int
            Index of the final row in a row chunk
        mode : str
            The file mode to use
        shp : tuple
            The shape of the raster arrays

        Returns
        -------
        numpy.ndarray
        '''
        if mode is None:
            mode = self._mode

        r0, r1 = (start_row, end_row)
        if self._subset_id is not None:
            x_idx, y_idx = self.__slice_idx__
            xmin, xmax = x_idx

        # If we're reading HDF5 files...
        if mode == 'hdf5':
            with h5py.File(file_path, 'r') as hdf:
                assert self._field in hdf.keys(), 'Could not find the specified field name: %s' % self._field
                # If we're not subsetting...
                if self._subset_id is None:
                    return hdf[self._field][r0:r1,:]

                return hdf[self._field][r0:r1, xmin:xmax]

        # If we're reading sparse TCF output arrays...
        elif mode in ('sparse', 'tcf'):
            # TCFArray has stronger assumptions than SparseArray; try to read
            #   the file either way
            tcf = SparseArray(file_path, self._grid)

            tcf.inflate()
            if self._subset_id is None:
                return tcf.data[r0:r1,:]

            return tcf.data[r0:r1, xmin:xmax]

        # If we're reading arbitrary raster arrays...
        elif mode == 'other':
            if self._subset_id is None:
                shp = self.__shp__ if shp is None else shp
                return as_array(
                    file_path, False, (0, r0, shp[1], r1))[0]

            return as_array(file_path, False, (xmin, r0, xmax, r1))[0]

        else:
            raise NotImplementedError('File mode "%s" not recognized' % mode)

    def read_raster(self, file_path, mode = None):
        '''
        Reads in the file at the given file path and returns a `gdal.Dataset`.

        Parameters
        ----------
        file_path : str
            The location of a file
        mode : str
            The file mode to use

        Returns
        -------
        gdal.Dataset
        '''
        if mode is None:
            mode = self._mode

        # For all other file modes...
        assert getattr(self, '_grid', None) is not None, 'Must define input --grid in order to read input HDF5 or sparse arrays'
        gt = EASE2_GRID_PARAMS[self._grid]['geotransform']
        wkt = EPSG[EASE2_GRID_PARAMS[self._grid]['epsg']]

        # If we're reading arbitrary raster arrays...
        if mode == 'other':
            rast, _, _ = as_raster(file_path, False)
            return rast

        # If we're reading HDF5 files...
        elif mode == 'hdf5':
            with h5py.File(file_path, 'r') as hdf:
                assert self._field in hdf.keys(), 'Could not find the specified field name: %s' % self._field
                arr = hdf[self._field][:]
                return array_to_raster(arr, gt, wkt)

        # If we're reading sparse TCF output arrays...
        elif mode in ('sparse', 'tcf'):
            # TCFArray has stronger assumptions than SparseArray; try to read
            #   the file either way
            tcf = SparseArray(file_path, self._grid)
            tcf.inflate()
            return array_to_raster(tcf.data, gt, wkt)

        else:
            raise NotImplementedError('File mode "%s" not recognized' % mode)


class ProgressBar(object):
    '''
    An animated progress bar for printing progress to the screen, as with a
    command line interface. Used as a context manager around a loop, e.g.:

        with ProgressBar(len(things), "Working...") as progress:
            for i, each in enumerate(things):
                ...
                progress.update(i)

    Parameters
    ----------
    total : int
        The total number of loop iterations
    prefix : str
        (Optional) The text to print before the progress bar (Default: "")
    suffix : str
        (Optional) The text to print at the end of the progress bar
        (Default: "")
    decimals : int
        (Optional) Number of decimal places for the progress bar's percent
        (Default: 0)
    length : int
        (Optional) The length of the progress bar, in characters (Default: 30)
    fill : str
        (Optional) The character to display for the filled portion of the bar
        (Default: "|")
    verbose : bool
        True to display the bar (Default: True); set to False if the
        `ProgressBar` should not be displayed
    '''
    def __init__(
            self, total, prefix = '', suffix = '', decimals = 0, length = 30,
            fill = '|', verbose = True):
        self._decimals = decimals
        self._fill = fill
        self._length = length
        self._prefix = prefix
        self._suffix = suffix
        self._total = total
        # Some CLIs might have a "verbose" mode, in which progress *should* be
        #   printed to the screen; when not in "verbose," we still enter and
        #   exit the ProgressBar context; so, we return a "dummy" bar that
        #   cannot print to screen when not in "verbose" mode
        self._verbose = verbose

    def __enter__(self):
        self.update(0) # Initialize the bar
        if not self._verbose:
            return self.dummy() # Return a dummy bar
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._verbose:
            self.update(self._total) # Finalize the bar

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, new_prefix):
        self._prefix = new_prefix

    def dummy(self):
        '''
        Returns a dummy instance of ProgressBar that prints nothing; used
        when not in "verbose" mode (the default).
        '''
        dummy = ProgressBar(0)
        setattr(dummy, 'update', lambda s, i: None)
        return dummy

    def update(self, iteration):
        '''
        When called in a loop, creates a progress bar in the terminal.
        NOTE: Adapted from this example:

            https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

        Parameters
        ----------
        iteration : int
            The current iteration
        total : int
            The total iterations
        prefix : str
            (Optional) Prefix string
        suffix : str
            (Optional) Suffix string
        decimals : int
            (Optional) Positive number of decimals in percent complete
        length : int
            (Optional) Character length of bar
        fill : str
            (Optional) Bar fill character
        '''
        percent = ("{0:." + str(self._decimals) + "f}")\
            .format(100 * (iteration / float(self._total)))
        d = 1 if self._decimals > 0 else 0 # Space for decimal point
        filled_len = int(self._length * iteration // self._total)
        bar = self._fill * filled_len + '-' * (self._length - filled_len)
        # Print new line on complete
        if iteration == self._total:
            # Clear the bar, leave the prefix
            print('\r%s%s' % (self._prefix, ''.rjust(self._length + 10)))
        else:
            print('\r%s [%s] %s%% %s' % (
                self._prefix, bar, percent.rjust(self._decimals + d + 3),
                self._suffix
            ), end = '\r')
