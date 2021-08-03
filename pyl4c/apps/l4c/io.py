'''
File input/output handlers for running L4C Science.
'''

import numpy as np
import h5py

class L4CStreamingInputDataset(object):
    '''
    A context that provides streaming access to an L4C inputs HDF5 file; i.e.,
    instead of reading in an entire data array at once, allows a data array
    to be indexed at each time step.

    Parameters
    ----------
    hdf_file_path : str
        File path to inputs HDF5 dataset
    constants : dict
        Dictionary of constant values
    bounds : dict
        (Optional) Dictionary of bounds for each parameter
    fill : dict
        (Optional) Dictionary of fill values, as two-element tuples, for each
        parameter; used when that parameter exceeds its (lower, upper) bounds
        and defaults to (NaN, NaN)
    '''
    def __init__(
            self, hdf_file_path: str, constants: dict, bounds: dict = None,
            fill: dict = None):
        self.bounds = bounds
        self.fill = fill
        self.constants = constants
        self.file_path = hdf_file_path
        if bounds is not None:
            assert hasattr(bounds, 'keys'), 'Argument "bounds" should be a dict type'
        if fill is not None:
            for key in fill.keys():
                assert len(fill[key]) == 2,\
                    'Tuples in "fill" dictionary should have 2 elements only'

    def __enter__(self):
        self.data = h5py.File(self.file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.data.close()

    def __getitem__(self, key):
        return self.data[key]

    def derive(self, t, field):
        '''
        Some fields cannot be read directly from the input dataset but must
        be calculated based on other fields.

        Parameters
        ----------
        t : int
            Current time step
        field : str
            Name of the input driver to derive

        Returns
        -------
        numpy.ndarray
        '''
        if field == 'apar':
            par = self.data['drivers/par'][t,...]
            fpar = self.data['drivers/fpar'][t,...]
            return np.multiply(par[:,np.newaxis].repeat(81, axis = 1), fpar)

        if field == 'ft':
            k = self.constants['tsurf_freeze-thaw_threshold_kelvin']
            return np.where(self.data['drivers/tsurf'][t,...] <= k, 0, 1)

    def index(self, t, *fields):
        '''
        Get the driver data at time t for one or more fields.

        Parameters
        ----------
        t : int
            Current time step
        *fields : str
            One or more field names
        '''
        drivers = []
        for f in fields:
            if f not in self.data['drivers'].keys():
                slice = self.derive(t, f) # Some fields need to be calculated
            else:
                slice = self.data['drivers/%s' % f][t,...]

            # Protect against out-of-bounds values
            if self.bounds is not None:
                if f in self.bounds.keys():
                    if not np.isnan(self.bounds[f][0]) and not np.isnan(self.bounds[f][1]):
                        # Get the fill value, defaulting to NaN
                        bounds = self.bounds[f]
                        if self.fill is not None:
                            lfill, ufill = self.fill[f]
                            slice = np.where(
                                np.logical_or(slice < bounds[0], slice > bounds[1]),
                                    np.where(slice < bounds[0], lfill,
                                        np.where(slice > bounds[1], ufill, slice)), slice)
                        else:
                            slice = np.where(
                                np.logical_or(slice < bounds[0], slice > bounds[1]),
                                np.nan, slice)
            # Propagate any single-valued fields at 9-km to the 1-km subgrid
            if slice.ndim == 1:
                slice = slice[:,np.newaxis].repeat(81, axis = 1)
            drivers.append(slice)
        return drivers
