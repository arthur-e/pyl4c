'''
Tools for down-scaling L4C 9-km data to 1-km scale, using the PFT means.

    nested = NestedGrid(subset_id = "CONUS")
    arr = nested.pft_mask(8) # 1-km map of PFT 8
    with h5py.File("something.h5", "r") as hdf:
        nested.downscale_hdf5_by_pft(hdf, "GPP/gpp_pft%d_mean")

All of the SMAP Level 4 Carbon (L4C) flux and soil organic carbon (SOC) state
variables are computed on a global, 1-km equal-area grid. We spatially average
the data to 9-km to meet operational constraints on data storage. However,
within each 9-km cell, we save the mean values for each Plant Functional Type
(PFT); e.g., the mean SOC density at all the "Evergreen Needleleaf" pixels in
a 9-km cell (between 0 and 81 pixels) is recorded, along with the means of up
to eight other PFTs. This allows a reconstruction of finer-scale detail, using
the 1-km map of PFT and assigning the mean values within each 9-km cell. The
9-km and 1-km grids are designed to nest perfectly [1]. **Therefore, it is
possible to generate a down-scaled 1-km map of SOC density (and of RH, NEE,
and GPP).**

1. https://nsidc.org/ease/ease-grid-projection-gt
'''

import os
import tempfile
import h5py
import numpy as np
from osgeo import gdal
from scipy.ndimage import zoom
from cached_property import cached_property
from pyl4c.data.fixtures import EASE2_GRID_PARAMS
from pyl4c.utils import get_ease2_slice_idx, get_pft_array
from pyl4c.spatial import ease2_to_geotiff
from pyl4c.lib.cli import CommandLineInterface

class NestedGrid(object):
    '''
    Represents a nested L4C data structure, where each grid cell has a mean
    value for each sub-grid PFT class. This allows the reconstruction of a
    finer scale grid by applying the PFT mean values to the subgrid.

    Parameters
    ----------
    pft : tuple
        PFT values to use in downscaling
    subset_id : str or None
        Name of a well-known spatial subset
    '''
    def __init__(self, pft = range(1, 9), subset_id = None):
        self._offsets = (0, 0)
        self._pft_codes = pft
        self._shp_1km = EASE2_GRID_PARAMS['M01']['shape']
        self._shp_9km = EASE2_GRID_PARAMS['M09']['shape']
        self._slice_idx_1km = None
        self._slice_idx_9km = None
        self._subset_id = subset_id
        if subset_id is not None:
            # Get the slicing indices, shape of the downscaled subset
            self._slice_idx_1km = get_ease2_slice_idx('M01', subset_id)
            self._shp_1km = (
                self._slice_idx_1km[1][1] - self._slice_idx_1km[1][0],
                self._slice_idx_1km[0][1] - self._slice_idx_1km[0][0])
            # Get the slicing indices, shape of the 9-km grid
            self._slice_idx_9km = get_ease2_slice_idx('M09', subset_id)
            self._shp_9km = (
                self._slice_idx_9km[1][1] - self._slice_idx_9km[1][0],
                self._slice_idx_9km[0][1] - self._slice_idx_9km[0][0])
            # Set the 1-km slicing indices
            xoff = self._slice_idx_1km[0][0]
            yoff = self._slice_idx_1km[1][0]
            self._offsets = (xoff, yoff)

    @cached_property
    def pft(self):
        'The 1-km PFT map'
        return get_pft_array(
            'M01', self._subset_id).astype(np.uint8)[np.newaxis,...]

    @property
    def offsets(self):
        '''
        Returns the xoff, yoff offsets for a subset array.
        Returns:
        tuple   (int, int)
        '''
        return self._offsets

    @property
    def shape(self):
        'Returns the 1-km array shape'
        return self._shp_1km

    def pft_mask(self, p):
        '''
        An (M x N) mask for selecting pixels matching specified PFT class.

        Parameters
        ----------
        p : int
            The numeric code for the PFT of interest

        Returns
        -------
        numpy.ndarray
        '''
        return np.where(self.pft == p, 1, 0)

    def downscale_hdf5_by_pft(
            self, hdf, field, scale = 1, dtype = np.float32, nodata = -9999):
        '''
        Will subset the *arrays if they are not the expected shape of the
        subset at 9-km scale. NOTE: Use scale = 1e6 if a total flux
        (e.g., total GPP) is required; 1e6 is the number of square meters in
        a 1-km L4C pixel.

        Parameters
        ----------
        hdf : h5py.File
        field : str
            Template for PFT-mean field in the HDF5 granule, e.g.,
            "GPP/gpp_pft%d_mean"
        scale : int or float
        dtype : numpy.dtype
        nodata : int or float

        Returns
        -------
        numpy.ndarray
        '''
        if nodata < 0:
            assert dtype not in (np.uint0, np.uint8, np.uint16, np.uint32, np.uint64),\
                "Can't use an unsigned data type with a negative NoData value"
        opts = { # Options to zoom()
            'order': 0, 'zoom': 9, 'mode': 'grid-constant', 'grid_mode': True
        }
        if self._subset_id is not None:
            x_idx, y_idx = self._slice_idx_9km
            xmin, xmax = x_idx
            ymin, ymax = y_idx

        downscaled = np.zeros(self._shp_1km)
        for p in self._pft_codes:
            # If needed, subset the 9-km arrays, then down-scale to 1-km
            if self._subset_id is None:
                arr = hdf[field % p][:]
            else:
                arr = hdf[field % p][ymin:ymax, xmin:xmax]
            # Resize, multiply by PFT mask, then add to output where != NoData
            downscaled = np.add(
                downscaled, np.multiply(self.pft_mask(p), zoom(arr, **opts)))

        # Where the PFT map is in the valid range, return data, else NoData
        return np.where(
            np.logical_and(
                np.in1d(self.pft.ravel(), self._pft_codes),
                downscaled.ravel() != nodata),
            np.multiply(downscaled.ravel(), scale), nodata)\
            .reshape(self._shp_1km)\
            .astype(dtype)


class CLI(CommandLineInterface):
    '''
    Command-line interface for running the downscaling procedure.

        python resample.py run <hdf5_granule> --field="GPP/gpp_pft%d_mean"
            --subset-id="CONUS"
    '''

    def __init__(
            self, output_path, pft = range(1, 9),
            field = 'SOC/soc_pft%d_mean', subset_id = None, scale = 1,
            nodata = -9999, dtype = 'float32', verbose = True):
        self._dtype = self.lookup_dtype(dtype)
        self._field_tpl = field
        self._nodata = nodata
        self._output_path = output_path
        self._pft = pft
        self._scale = scale
        self._subset_id = subset_id
        self._verbose = verbose

    def run(self, hdf5_path, compress = True):
        '''
        Downscales an L4C variable from the given HDF5 granule.

        Parameters
        ----------
        hdf5_path : str
            File path to an HDF5 granule
        compress : bool
            True to compress the output file (Default: True)
        '''
        nested = NestedGrid(self._pft, self._subset_id)
        with h5py.File(hdf5_path, 'r') as hdf:
            arr = nested.downscale_hdf5_by_pft(
                hdf, self._field_tpl, self._scale)
            if self._verbose:
                print('Downscaling...')
        xoff, yoff = nested.offsets
        output_path = self._output_path
        if compress:
            tmp = tempfile.NamedTemporaryFile()
            output_path = tmp.name
        # Write initial file
        ease2_to_geotiff(arr, output_path, 'M01', xoff = xoff, yoff = yoff)
        if not compress:
            return # Done
        # Optionally, compress the output file
        opts = gdal.TranslateOptions(
            format = 'GTiff', creationOptions = ['COMPRESS=LZW'])
        # Note that "output_path" is the temporary file
        gdal.Translate(self._output_path, output_path, options = opts)


if __name__ == '__main__':
    import fire
    fire.Fire(CLI)
