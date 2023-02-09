'''
SMAP Level 4 Carbon (L4C) forward run logic, including data structures for
running an L4C forward simulation. A point (site-level) version of the L4C
algorithm can be examined and run in `pyl4c.apps.l4c.main`.
'''

import csv
import datetime
import warnings
import numpy as np
from functools import partial
from pyl4c import suppress_warnings
from pyl4c.lib.cli import ProgressBar

class AbstractData(object):
    '''
    An abstraction of a data container for running L4C simulations.
    Represents discontinuous locations or "points" on a global EASE-Grid 2.0
    over which to run an L4C simulation. Each "point" has a 1-km subgrid
    associated with the 9-km "point" cell.
    '''
    @property
    def labels(self):
        return self._data_labels

    @labels.setter
    def labels(self, new_labels):
        self._data_labels = tuple(new_labels)

    @property
    def shape(self):
        if self.data is not None:
            return self.data.shape


class L4CConstants(object):
    '''
    A namespace for constant values used in running L4C. These constant
    values are broadcast to an array that is conformable with driver data
    for fast and seamless computation in the forward run.

    Parameters
    ----------
    count : int
        Number of point locations ("sites") used in forward simulation
    pft_array : numpy.ndarray
        Array specifying the PFT at each point location
    valid_pft : Sequence
        The valid PFT codes
    '''
    def __init__(self, count : int, pft_array, valid_pft = range(1, 10)):
        # Set invalid PFT codes to 0
        self._count = count
        self._pft = np.where(np.in1d(pft_array.ravel(), valid_pft),
            pft_array.ravel(), 0)

    def conformable(self, array):
        '''
        Creates a `(1) x N x P` array where `P` is the number of PFT classes.
        Useful for guaranteeing that an array is conformable with the other
        arrays it will be concatenated with.

        Parameters
        ----------
        array : numpy.ndarray
        '''
        assert array.ndim <= 4, 'Expected ndim <= 4'
        if array.ndim == 1:
            # A 1D array indicates one value per PFT
            return np.apply_along_axis(
                lambda p: array[p], 0, self._pft
            ).reshape((self._count, 81))
        if array.ndim == 2:
            if array.shape == (self._count, 81):
                return array # Array already conforms
            # A 2D array indicates multiple ("ndim") values per PFT
            return np.apply_along_axis(
                lambda p: array[:,p], 0, self._pft
            ).reshape((array.shape[0], self._count, 81))
        if array.ndim == 3:
            # i.e., a 365-day climatology
            assert array.shape == (365, self._count, 81),\
                'No support for 3D arrays that do not have 365 elements on axis 0 (i.e., a 365-day climatology)'
            return array
        if array.ndim == 4:
            assert array.shape[-3:] == (365, self._count, 81),\
                'No support for 4D arrays that do not have the following sub-space: (%s)' % ' x '.join(map(str, array.shape[-3:]))
            return array

    def add(self, array, label):
        '''
        Adds a new constant as an attribute on this instance.

        Parameters
        ----------
        array : numpy.ndarray
            The array to add
        label : str
            The name by which this array will be known
        '''
        setattr(self, label, self.conformable(array))


class L4CDrivers(AbstractData):
    '''
    Represents a collection of L4C driver variables. Driver variables *drive*
    the model but are not in themselves updated; they are exogenous variables
    with values fixed at the start of the simulation.

    Parameters
    ----------
    count : int
        Number of point locations ("sites") used in forward simulation
    drivers
    labels
    '''
    def __init__(self, count, drivers = None, labels = None):
        if drivers is not None and labels is not None:
            if hasattr(drivers, 'ndim'):
                matched = drivers.shape[0] == len(labels)
            else:
                matched = len(drivers) == len(labels)
            if not matched:
                raise ValueError('There should be as many labels as there are driver variables')
        self._count = count
        self._data = None
        self._data_labels = labels
        # Use appropriate setter function
        self.data = drivers

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new):
        if new is None:
            self._data = None
        elif hasattr(new, 'ndim'):
            self._data = new
        else:
            # Otherwise, concat the sequence of arrays, once made conformable
            self._data = np.concatenate(list(map(
                partial(self.conformable), new)), axis = 0)

    def conformable(self, array):
        '''
        Given an array, returns representation that conforms to expectations.

        Parameters
        ----------
        array : numpy.ndarray
            Input data array

        Returns
        -------
        numpy.ndarray
            `(1 x T x N x 81)` array
        '''
        shp = array.shape
        assert (self._count in shp),\
            'Array is not conformable with %d measurement points' % self._count
        assert array.ndim in (2, 3, 4),\
            'Driver variables should have 2, 3, or 4 axes only'

        # Driver data should be (1 x T x N x 81)
        i_idx = shp.index(self._count)
        if array.ndim == 2:
            t = shp[0] if i_idx == 1 else shp[1] # Get size of time axis
            return array.swapaxes(1, i_idx).reshape((t, self._count, 1))\
                    .repeat(81, axis = 2).reshape((1, t, self._count, 81))

        # NOTE: The swapaxes() calls essentially assert place each axis into
        #   its desired position, moved from its inferred position
        j_idx = shp.index(81) if 81 in shp else shp.index(1) # Subgrid axis
        t_idx = set((0, 1, 2)).difference((i_idx, j_idx)).pop() # Time axis
        if array.ndim == 3:
            t = shp[t_idx] # Get the size of the time axis
            if shp[j_idx] == 81:
                return array.swapaxes(1, i_idx).swapaxes(2, j_idx)\
                    .reshape((1, t, self._count, 81))
            return array.swapaxes(1, i_idx).swapaxes(2, j_idx)\
                .reshape((1, t, self._count, 1)).repeat(81, axis = 3)

        if array.ndim == 4:
            return array.swapaxes(1, t_idx).swapaxes(2, i_idx)\
                .swapaxes(3, j_idx)


class L4CState(AbstractData):
    '''
    Represents a collection of L4C state variables. State variables influence
    the driving of the model; an initial state (at time t=0) is required to
    start running and the variable state might be updated at each time step
    (t) based on the value at step (t-1). State data are stored as
    `M x T x N x 81` arrays.

    Parameters
    ----------
    count : int
        Number of point locations ("sites") used in forward simulation
    state : numpy.ndarray or None
        (Optional) Initial state
    labels : list or tuple or None
        (Optional) Sequence of labels for each state variable, should be
        the same length as the first axis of the `state` of array
    axis_labels : list or tuple or None
        Names of each axis of the stored state matrix; axes are, in order:
        (state variable, time, site, 81)
    dtype
        The data type to coerce on the state array
    '''
    def __init__(
            self, count : int, state = None, labels = None,
            axis_labels = None, dtype = np.float32):
        if state is not None and labels is not None:
            if hasattr(state, 'ndim'):
                matched = state.shape[0] == len(labels)
            else:
                matched = len(state) == len(labels)
            if not matched:
                raise ValueError('There should be as many labels as there are state variables')
        self._axis_labels = axis_labels
        self._count = int(count)
        self._bounds = (0, np.inf)
        self._data = None
        self._data_labels = labels
        self._dtype = dtype
        # Use appropriate setter function
        if state is not None:
            self.data = state.astype(dtype)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new):
        if new is None:
            self._data = None
        elif hasattr(new, 'ndim'):
            assert new.ndim == 4,\
                'New data array must have 4 axes; should be (M x T x N x 81)'
            self._data = new.astype(self._dtype)
        else:
            # Otherwise, concat the sequence of arrays, once made conformable
            self._data = np.concatenate(
                list(map(self.conformable, new)), axis = 0).astype(self._dtype)

    def _serialize(
            self, state, label, output_path, truncate, time_labels, prec,
            verbose = False):
        'Argument "state" should be length-T sequence of length-N sequences'
        now = datetime.datetime.now().strftime('%Y-%m-%d')
        site_labels = list(range(0, self._count))
        if self._axis_labels is not None:
            site_labels = self._axis_labels[-2]
            # Decode UTF-8 strings
            if hasattr(site_labels[0], 'decode'):
                site_labels = list(map(lambda x: x.decode(), site_labels))
        with open(output_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(
                ('# L4C Model State %s serialized %s' % (label, now),))
            writer.writerow(('site', 'time', 'value'))
            with ProgressBar(len(state), verbose = verbose) as progress:
                if verbose:
                    progress.prefix = 'Serializing...'
                for t in range(0, len(state)):
                    for i, label in enumerate(site_labels):
                        # Working with a length-T sequence of length-N lists
                        writer.writerow([
                            label, time_labels[t],
                            format(state[t,i], '.%df' % prec)
                        ])
                    if verbose:
                        progress.update(t)

    @suppress_warnings
    def _serialize_combined(
            self, label, output_path, truncate, time_labels, prec):
        'Serializes the combined (summed) state, summing across fields'
        # Sum across fields, then take the mean across the 1-km subgrid
        state = np.nanmean(self.data.sum(axis = 0), axis = -1)
        if truncate is not None:
            state = state[0:truncate,...].tolist()
        else:
            state = state.tolist()
        self._serialize(
            state, label, output_path, truncate, time_labels, prec)

    def advance(self, label, t, delta = None, func = np.add, bounds = None):
        '''
        Updates the state data at time (t) based on the prior state (state
        at t-1). NOTE: "delta" is the left argument of the binary function.

        Parameters
        ----------
        label : str
            One of the labels in self.labels
        t : int
            Index of the time step to update
        delta : numpy.ndarray
            LHS of the function (func) to change state (Default: None)
        func : function
            The binary function to use to combine delta and the state
            at time (t-1) (Default: numpy.add).
        bounds : dict
        '''
        i = self.labels.index(label)
        if delta is not None:
            new_state = func(delta, self._data[i, t-1, ...])
            # Check for out-of-bounds
            if bounds is None:
                self._data[i, t, ...] = new_state
                return

            self._data[i, t, ...] = np.where(
                new_state > bounds[1], bounds[1], np.where(
                    new_state < bounds[0], bounds[0], new_state))

    def conformable(self, array, allocation = 1):
        '''
        Given an array, returns representation that conforms to expectations.

        Parameters
        ----------
        array : numpy.ndarray
            Input data array
        allocation : int
            Number of time steps to allocate

        Returns
        -------
        numpy.ndarray
            `(1 x T x N x 81)` array
        '''
        shp = array.shape
        assert (self._count in shp),\
            'Array is not conformable with %d measurement points' % self._count
        assert array.ndim in (2, 3),\
            'State variables should have either 2 or 3 axes only'

        # We want to return a (1 x T x N x 81) array; tests are somewhat
        #   verbose in light of the possibility that self._count = 81
        assert 81 in shp,\
            'State variables must have a 1-km subgrid axis with 81 elements'
        i_idx = shp.index(self._count)
        j_idx = shp.index(81)
        # Add concatenation (axis=0) and time (axis=1) axes, then allocate
        #   T time steps
        if array.ndim == 2:
            return np.concatenate((
                array.swapaxes(i_idx, 0).reshape((1, 1, self._count, 81)),
                np.full((1, allocation - 1, self._count, 81), np.nan)
            ), axis = 1)
        if array.ndim == 3:
            return np.concatenate((
                array.swapaxes(i_idx, 1).swapaxes(j_idx, 2)\
                    .reshape((1, 1, self._count, 81)),
                np.full((1, allocation - 1, self._count, 81), np.nan)
            ), axis = 1)

    def update(self, label, t, array, bounds = None):
        '''
        Updates the state data with the given label by inserting an array
        at the specified time step (t).

        Parameters
        ----------
        label : str
            One of the labels in self.labels
        t : int
            Index of the time step to update
        array : numpy.ndarray
        bounds : dict
        '''
        i = self.labels.index(label)
        if bounds is None:
            self._data[i, t, ...] = array
            return

        self._data[i, t, ...] = np.where(
            array > bounds[1], bounds[1], np.where(
                array < bounds[0], bounds[0], array))

    def serialize(
            self, output_tpl, truncate = None, time_labels = None, prec = 3,
            verbose = False):
        '''
        Dumps the current state to a CSV file. Data on the 1-km subgrid are
        summarized; the mean at each 9-km site is reported.

        Parameters
        ----------
        output_tpl : str
            Template filename for the output CSV file
        truncate : int
            (Optional) Largest time index that will be dumped
        time_labels : list or tuple
            (Optional) A sequence of time labels
        prec : int
            Number of decimal places to include (Default: 1)
        verbose : bool
            True to print progress of serialization
        '''
        assert truncate is None or (hasattr(truncate, 'real') and not hasattr(truncate, 'is_integer')), '"truncate" must be an integer'
        # Insert a string formatting character for the state's label
        output_tpl = '.'.join(output_tpl.split('.')[:-1])
        output_tpl = '%s_%%s.csv' % output_tpl

        if time_labels is None:
            time_labels = range(0, self.shape[1])
        else:
            assert isinstance(time_labels[0], str),\
                'time_labels must be str type'

        # Check that there are at least as many time labels as time indices
        t_max = truncate if truncate is not None else self.shape[1]
        assert len(time_labels) >= t_max, 'Not enough time_labels!'

        # If state represents SOC, combine across the SOC pools
        if all([s.startswith('soc') for s in self.labels]):
            self._serialize_combined(
                'SOC', output_tpl % 'SOC', truncate, time_labels, prec)
            return

        # Take the mean across the 1-km subgrid
        state = self.data
        if self.data.ndim > 3:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                state = np.nanmean(self.data, axis = -1)
        print('Datasets to serialize: %d' % self.shape[0])
        for i in range(0, self.shape[0]):
            self._serialize(
                state[i,...], self.labels[i].upper(),
                output_tpl % self.labels[i].upper(), truncate, time_labels,
                prec, verbose = verbose)


class L4CStratifiedState(L4CState):
    '''
    Represents a collection of L4C state variables that are potentially
    stratified into vertical layers. State data are stored as
    `M x T x Z x N x 81` arrays, where `M` enumerates the different state
    variables, `Z` enumerates the depths, `T` is the time axis, and `N`
    enumerates the number of sites.

    Parameters
    ----------
    layers : int
        Number of vertical layers
    count : int
        Number of point locations ("sites") used in forward simulation
    state : numpy.ndarray or None
        (Optional) Initial state
    labels : list or tuple or None
        (Optional) Sequence of labels for each state variable, should be
        the same length as the first axis of the `state` of array
    axis_labels : list or tuple or None
        Names of each axis of the stored state matrix; axes are, in order:
        (state variable, time, site, 81)
    dtype
        The data type to coerce on the state array
    '''
    def __init__(
            self, layers: int, count: int, state = None, labels = None,
            axis_labels = None, dtype = np.float32):
        self._layers = int(layers)
        if axis_labels is not None:
            assert len(axis_labels) == 5,\
                '"axis_labels" should contain one sequence for each axis (5 axes)'
        super().__init__(
            count = count, state = state, labels = labels,
            axis_labels = axis_labels, dtype = dtype)

    def _serialize(
            self, state, label, output_path, truncate, time_labels, prec,
            verbose = False):
        'Argument "state" should be length-T sequence of length-N sequences'
        now = datetime.datetime.now().strftime('%Y-%m-%d')
        site_labels = list(range(0, self._count))
        if self._axis_labels is not None:
            _, _, depth_labels, site_labels, _ = self._axis_labels
            # Decode UTF-8 strings
            if hasattr(site_labels[0], 'decode'):
                site_labels = list(map(lambda x: x.decode(), site_labels))
        with open(output_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(
                ('# L4C Model State %s serialized %s' % (label, now),))
            writer.writerow(('site', 'depth', 'time', 'value'))
            with ProgressBar(len(state), verbose = verbose) as progress:
                if verbose:
                    progress.prefix = 'Serializing...'
                for t in range(0, len(state)):
                    for z, depth in enumerate(depth_labels):
                        for i, site_name in enumerate(site_labels):
                            # Working with a length-T sequence of length-N lists
                            writer.writerow([
                                site_name, depth, time_labels[t],
                                format(state[t][z][i], '.%df' % prec)
                            ])
                        if verbose:
                            progress.update(t)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new):
        if new is None:
            self._data = None
        elif hasattr(new, 'ndim'):
            assert new.ndim >= 4,\
                'New data array must have at least 4 axes; should be (T x Z x N x 81)'
            self._data = new.astype(self._dtype)
        else:
            # Otherwise, concat the sequence of arrays, once made conformable
            self._data = np.concatenate(
                list(map(self.conformable, new)), axis = 0).astype(self._dtype)

    def conformable(self, array, allocation = 1):
        '''
        Given an array, returns representation that conforms to expectations.

        Parameters
        ----------
        array : numpy.ndarray
            Input data array
        allocation : int
            Number of time steps to allocate

        Returns
        -------
        numpy.ndarray
            `(1 x T x Z x N x 81)` array
        '''
        shp = array.shape
        assert (self._count in shp),\
            'Array is not conformable with %d measurement points' % self._count
        assert array.ndim in (3, 4, 5),\
            'State variables should have between 3 and 5 axes only'

        # We want to return a (1 x Z x T x N x 81) array; tests are somewhat
        #   verbose in light of the possibility that self._count = 81
        assert 81 in shp,\
            'State variables must have a 1-km subgrid axis with 81 elements'
        i_idx = shp.index(self._count)
        j_idx = shp.index(81)
        if array.ndim == 3:
            return np.concatenate((
                array.swapaxes(1, i_idx).swapaxes(2, j_idx)\
                    .reshape((1, 1, self._count, 81)),
                np.full((1, allocation - 1, 1, self._count, 81), np.nan)
            ), axis = 1)
        # Arrays of 4D or higher are assumed to have a Z axis
        z_idx = shp.index(self._layers) if self._layers in shp else shp.index(1)
        if array.ndim == 4:
            return np.concatenate((
                array.swapaxes(0, z_idx).swapaxes(2, i_idx).swapaxes(3, j_idx)\
                    .reshape((1, self._layers, 1, self._count, 81)),
                np.full((1, allocation - 1, self._layers, self._count, 81), np.nan),
            ), axis = 1)
        if array.ndim == 5:
            return np.concatenate((
                array.swapaxes(0, z_idx).swapaxes(3, i_idx).swapaxes(4, j_idx)\
                    .reshape((1, self._layers, 1, self._count, 81)),
                np.full((1, allocation - 1, self._layers, self._count, 81), np.nan),
            ), axis = 1)


def report(hdf):
    '''
    Check that we have everything needed to run L4C, print a report to the
    screen.

    Parameters
    ----------
    hdf : h5py.File
    '''
    KEYS = ('apar', 'vpd', 'ft', 'tmin', 'tsoil', 'smrz', 'smsf')

    def find(hdf, prefix, key, pad = 10):
        'Find a key, print the report'
        try:
            field = '%s/%s' % (prefix, key)
            if len(hdf[field].shape) == 2 or key == 'fpar':
                pretty = ('"%s"' % key).ljust(pad)
                print_stats(hdf[field][:], pad, pretty)
            elif len(hdf[field].shape) == 3:
                # Assuming data are enumerated on the first axis
                for i in range(0, hdf[field].shape[0]):
                    pretty = ('"%s" (%d)' % (key, i)).ljust(pad)
                    print_stats(hdf[field][i,...], pad, pretty)
        except KeyError:
            pretty = ('"%s"' % key).ljust(pad)
            print('-- MISSING %s' % pretty)

    def print_stats(data, pad, pretty):
        shp = ' x '.join(map(str, data.shape))
        shp = ('[%s]' % shp).ljust(pad + 7)
        stats = tuple(summarize(data))
        stats_pretty = ''
        if stats[0] is not None:
            stats_pretty = '[%.2f, %.2f]' % (stats[0], stats[2])
            if len(key) < 10:
                print('-- Found %s %s %s' % (pretty, shp, stats_pretty))
            else:
                print('-- Found %s' % pretty)
                print('%s%s %s' % (''.rjust(pad + 10), shp, stats_pretty))

    def summarize(data, nodata = -9999):
        'Get summary statistics for a field'
        if str(data.dtype).startswith('int'):
            return (None for i in range(0, 3))
        data[data == -9999] = np.nan
        return (
            getattr(np, f)(data) for f in ('nanmin', 'nanmean', 'nanmax')
        )

    print('\nL4C: Validating configuration and input datasets for file:')
    print('  %s' % hdf.filename)
    print('\nL4C: Checking for required driver variables...')
    for key in KEYS:
        if key == 'ft' and key not in hdf['drivers'].keys():
            find(hdf, 'drivers', 'tsurf')
        elif key == 'apar' and key not in hdf['drivers'].keys():
            find(hdf, 'drivers', 'par')
            find(hdf, 'drivers', 'fpar')
        else:
            find(hdf, 'drivers', key)

    print('\nL4C: Checking for required state variables...')
    for key in ('PFT', 'npp_sum', 'soil_organic_carbon',):
        find(hdf, 'state', key)

    print('\nL4C: Summarizing metadata...')
    y1, m1, d1, _ = hdf['time'][0,...]
    y2, m2, d2, _ = hdf['time'][-1,...]
    print('-- First date: %s' % datetime.datetime(y1, m1, d1)\
        .strftime('%Y-%m-%d'))
    print('-- Final date: %s' % datetime.datetime(y2, m2, d2)\
        .strftime('%Y-%m-%d'))
    print('-- Total length: %d' % hdf['time'].shape[0])
    print('')
