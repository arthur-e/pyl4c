'''
Tools for representing flux tower sites and working with flux tower data.

NOTE: Tower subgrid should accurately reflect the canonical EASE-Grid 2.0
nesting, i.e., the 1-km cells returned correspond to those nested under the
respective 9-km cell, which is apparent if we look at the subgrid of the
top-left cell:

    >>> TowerSite('test', ease2_to_wgs84((0, 0)), validate = False).subgrid
'''

import csv
import glob
import h5py
import numpy as np
from cached_property import cached_property
from pyl4c import haversine
from pyl4c.ease2 import ease2_search_radius, ease2_to_wgs84, ease2_from_wgs84
from pyl4c.utils import get_pft_array, index

class MetaTowerSite(type):
    '''
    Prototype for Tower classes.
    '''
    @property
    def pft_1km_map(cls):
        if getattr(cls, '_pft_1km_map', None) is None:
            cls._pft_1km_map = get_pft_array('M01')
        return cls._pft_1km_map

    @property
    def pft_9km_map(cls):
        if getattr(cls, '_pft_9km_map', None) is None:
            cls._pft_9km_map = get_pft_array('M09')
        return cls._pft_9km_map


class TowerSite(object, metaclass = MetaTowerSite):
    '''
    Represents an eddy covariance tower site.

    Parameters
    ----------
    id : str
        Unique name for the TowerSite
    coords : tuple | list
        The site's (longitude, latitude) coordinates
    validate : bool
        True to validate the PFT of any mapped EASE-Grid 2.0 cell
    '''
    valid_pft_range = range(1, 9)

    def __init__(self, id, coords, validate = True):
        assert hasattr(coords[0], 'real') and hasattr(coords[1], 'real'),\
            'Coordinates must be numbers, not strings or other types'
        self.id = id
        self.coords = coords
        self.idx_1km = ease2_from_wgs84(coords, 'M01', exact = True)
        self.idx_9km = ease2_from_wgs84(coords, 'M09', exact = True)
        self._subgrid_idx = None
        self._subgrid_pft = None
        self._validate = validate
        if validate:
            self.locate_ease2() # Fire it up!

    @cached_property
    def pft_9km(self):
        'The PFT class at 9-km scale'
        r, c = self.idx_9km
        return self.__class__.pft_9km_map[int(r), int(c)]

    @property
    def subgrid(self):
        'Returns a list of 1-km subgrid indices'
        if self._subgrid_idx is None:
            # A search radius of 4 pixels yields a width, height of 9 pixels
            self._subgrid_idx = ease2_search_radius(self.coords, 4, 'M01')
        return self._subgrid_idx

    @property
    def subgrid_pft(self):
        'Returns an 81-element array of the 1-km subgrid PFTs'
        if self._subgrid_pft is None:
            self._subgrid_pft = index(
                self.__class__.pft_1km_map,
                np.array(self.subgrid).swapaxes(0, 1).tolist())
        return np.array(self._subgrid_pft)

    def locate_ease2(self):
        '''
        Finds the EASE-Grid 2.0 row, column indices for this site.

        Returns
        -------
        list
            A list of tuples, each a 2-element sequence of (row, column)
            indices
        '''
        row, col = self.idx_9km
        pft = self.__class__.pft_9km_map[int(row), int(col)] # Expensive
        if self._validate and pft not in self.valid_pft_range:
            # Get the closest valid EASE-Grid 2.0 cell instead
            try:
                row, col = self.locate_ease2_alternatives(self.coords)
                # Chain forward and reverse transforms to get the grid
                #   cell indices in floating point
                row, col = ease2_from_wgs84(
                    ease2_to_wgs84((row, col)), 'M09', exact = True)
            except ValueError:
                raise ValueError('Could not locate suitable EASE-Grid 2.0 cell')

        # Update the 9-km grid coordinates, as needed
        self.idx_9km = row, col
        return self.idx_9km

    def locate_ease2_alternatives(self, coords, grid = 'M09'):
        '''
        Finds nearest EASE-Grid 2.0 cell w/ valid PFT up to 1 grid cell away.

        Parameters
        ----------
        coords : tuple or list
            2-element sequence of (longitude, latitude) coordinates
        grid : str
            EASE-Grid 2.0 cell size designation, e.g, "M01" or "M09"
            (Default: "M09")

        Returns
        -------
        list
            A list of tuples, each a 2-element sequence of (row, column)
            indices
        '''
        # Round the row, column indices of this point so as to check for
        #   collisions in haversine(), below
        row, col = ease2_from_wgs84(coords, grid, exact = False)
        indices = ease2_search_radius(coords, 1, grid)
        indices_pfts = index(
            self.__class__.pft_9km_map,
            np.array(indices).swapaxes(0, 1).tolist())
        options = [] # Make sure the choices also have valid PFTs
        for i, idx in enumerate(indices):
            if indices_pfts[i] in self.valid_pft_range:
                options.append(idx)
        dists = [ # Calculate great circle distance
            haversine(
                coords, ease2_to_wgs84((r,c), grid)
            ) for r, c in options if not (r == row and c == col)
        ]
        return options[np.argmin(dists)]


def consolidate_tower_data(
        file_glob, output_hdf5_path, n_sites = 356, n_steps = 6575,
        keys = ('nee', 'gpp', 'reco')):
    '''
    Combines the CSVs from multiple calibration tower sites into a single HDF5
    dataset, e.g.:

        consolidate_tower_data(
            "/anx_lagr2/laj/smap/fluxnet2015/merged/*_smapMergedCalVal.csv",
            "/anx_lagr3/arthur.endsley/SMAP_L4C/calibration/Fluxnet2015_LaThuile_tower_data_for_356_sites.h5")

    Parameters
    ----------
    file_glob : str
        GLOB expression targeting all input CSV files
    output_hdf5_path : str
        File path to output HDF5 file
    n_sites : int
        Number of tower sites
    n_steps : int
        Number of time steps
    keys : tuple
        Sequence of keys to extract from each CSV
    '''
    file_paths = glob.glob(file_glob)
    assert len(file_paths) == n_sites, 'Should be one file per tower site'
    all_records = []
    for path in file_paths:
        site_records = []
        with open(path, 'r') as stream:
            reader = csv.DictReader(stream)
            for each in reader:
                site_records.append([
                    np.float(each[k]) if each[k] != 'NaN' else np.nan
                    for k in keys
                ])

            all_records.append(site_records)
            site_records = None

    shp = (n_steps, n_sites) # Output a T x N array
    arr = np.array(all_records) # Becomes an (N x T x k) array
    all_records = None
    with h5py.File(output_hdf5_path, 'a') as hdf:
        for i, field in enumerate(keys):
            hdf.create_dataset(field, shp, dtype = np.float32,
                data = arr[...,i].swapaxes(0, 1))
