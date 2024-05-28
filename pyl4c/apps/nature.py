'''
Tools for working with L4SM Nature Run data.

TODO: Refactor to use `numpy.ravel_multi_index`?
'''

import numpy as np
import netCDF4
from cached_property import cached_property
from pyl4c.data.fixtures import EASE2_GRID_PARAMS

class NatureRunNetCDF4(object):
    '''
    Represents an L4SM Nature Run netCDF4 dataset. Contains methods for
    converting from the compact, 1D "tile-space" to a 2D EASE-Grid 2.0.

    Would have liked to subclass netCDF4.Dataset directly, but it is part of
    a Cython library so I can't interrogate its class structure.

    Parameters
    ----------
    file_path : str
        The NetCDF4 file path
    '''
    def __init__(self, file_path: str):
        self._dataset_file_path = file_path
        self._grid = 'M09'
        try:
            self.dataset = netCDF4.Dataset(file_path)
        except OSError:
            raise OSError('Due to a bug in netCDF4, you MUST import netCDF4 BEFORE h5py; import this module (calibration) first if using in a script')

    @property
    def variables(self):
        return self.dataset.variables

    @cached_property
    def _col_idx(self):
        return np.array(self.dataset.variables['IG'][:])

    @cached_property
    def _row_idx(self):
        return np.array(self.dataset.variables['JG'][:])

    @classmethod
    def index_bulk(cls, nc, variable, tile_idx, reducer = 'mean'):
        '''
        A more efficient way of indexing, assuming that tile indices are
        already known. This avoids the expensive mapping of the same set
        of EASE-Grid 2.0 coordinates to tile space, and can be applied
        to multiple tile-space netCDF4 files.

        Parameters
        ----------
        nc : netCDF4.Dataset
            The netCDF4 file to sample from
        variable : str
            Name of the variable
        tile_idx : list or tuple
            Tile-space indices

        Returns
        -------
        numpy.ndarray
        '''
        target_variable = nc.variables[variable]
        # Minimize the extent of the array we need to read from the file;
        #   e.g., tile space indices are [0, 1, ... N] but we may only need
        #   to index between A and B, so instead of reading from 0 all the way
        #   to N, just read between A (tmin) and B (tmax).
        tmin, tmax = (min(tile_idx), max(tile_idx) + 1)
        subarray = target_variable[:,tmin:tmax]
        # Check if we're working with a 3-hourly L4SM variable
        if target_variable.ndim == 2:
            value_store = [
                getattr(subarray[:,idx - tmin], reducer)(axis = 0)
                for idx in tile_idx
            ]
        else:
            value_store = [
                subarray[idx - tmin] for idx in tile_idx
            ]
        return np.array(value_store)

    def as_tile(self, array):
        'Converts a 2D EASE-Grid 2.0 array to tile format'
        new_array = np.ones(self._row_idx.shape) * np.nan
        for i in range(0, self._row_idx.shape[0]):
            row, col = self.tile_to_ease2(i)
            new_array[i] = array[row, col]
        return new_array

    def tile_to_ease2(self, tile_idx):
        'Maps 1D tile coordinates to 2D EASE-Grid 2.0 coordinates'
        return (self._row_idx[tile_idx], self._col_idx[tile_idx])

    def ease2_to_tile(self, row, col):
        'Maps 2D EASE-Grid 2.0 coordinates to 1D tile coordinates'
        length = self._row_idx.size
        match = np.logical_and(self._row_idx == row, self._col_idx == col)
        if not np.any(match):
            raise ValueError("No matching row-column index pair was found; requested cell might be outside of the model domain")
        return int(np.arange(length)[match])

    def index(self, variable, indices):
        '''
        Returns values at the given indices, based on row-column pairs
        from a 2D EASE-Grid 2.0.

        NOTE: It is MUCH faster (by an order of magnitude) to use the class
        method index_bulk() if you need to extract the same indices from
        multiple tile-space netCDF4 files. The recommended pattern is based
        on this function's signature, where the `tile_idx` are computed
        first and then passed to index_bulk().

        Parameters
        ----------
        variable : str
            Name of the netCDF4 variable to inflate
        indices : tuple or list
            Sequence of (row, column) 2-tuples

        Returns
        -------
        numpy.ndarray
        '''
        assert hasattr(indices, 'index'), 'Indices should be a sequence of (row, column) pairs'
        target_variable = self.dataset.variables[variable]
        tile_idx = [ # This is what takes the longest
            self.ease2_to_tile(row, col) if row is not None and col is not None else None
            for row, col in indices
        ]
        return self.__class__.index_bulk(self, variable, tile_idx)

    def inflated(self, variable, nodata = -9999):
        '''
        Returns a "tile space" variable (1D) as a 2D NumPy array, calculating
        a daily average as needed.

        Parameters
        ----------
        variable : str
            Name of the netCDF4 variable to inflate
        nodata : int or float
            The NoData value to use in the output array

        Returns
        -------
        numpy.ndarray
        '''
        target_variable = self.dataset.variables[variable]
        shp = EASE2_GRID_PARAMS[self._grid]['shape']
        result = np.ones(shp) * nodata

        # Check if we're working with a 3-hourly L4SM variable
        if target_variable.ndim == 2 and target_variable.shape[0] == 8:
            avg = np.array(target_variable[:]).mean(axis = 0)
            for i in range(0, target_variable.shape[1]):
                # Calculate daily mean (collapse first axis)
                row, col = self.tile_to_ease2(i)
                result[row, col] = avg[i]
        else:
            for i in range(0, target_variable.shape[0]):
                row, col = self.tile_to_ease2(i)
                result[row, col] = target_variable[i]

        return result
