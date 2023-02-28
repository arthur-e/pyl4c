'''
Tools for down-scaling L4C 9-km data to 1-km scale, using the PFT means.

    nested = NestedGrid(subset_id = "CONUS")
    arr = nested.pft_mask(8) # 1-km map of PFT 8
    with h5py.File("something.h5", "r") as hdf:
        nested.downscale_hdf5_by_pft(hdf, "GPP/gpp_pft%d_mean")

The above example assumes that you have official SPL4CMDL HDF5 granules from
NSIDC or EarthData Search. It's also possible to downscale netCDF4 granules
from NASA AppEEARS, including granules that represent spatial subsets:

    nc = netCDF4.Dataset('AppEEARS_granule.nc')

    # Read the EASE-Grid 2.0 coordinate arrays, get bounds
    xmin, xmax = min(nc['fakedim1'][:]), max(nc['fakedim1'][:])
    ymin, ymax = min(nc['fakedim0'][:]), max(nc['fakedim0'][:])

    AOI = (xmin, ymin, xmax, ymax)
    # It may be necessary to provide a "shape" argument to ensure that
    #   the land-cover/ PFT data are forced to the shape of the netCDF4 data
    nested = NestedGrid(
        subset_bbox = AOI, shape = nc.variables['SOC_soc_mean'][0].shape)
    result = nested.downscale_netcdf_by_pft(
        nc, field = 'SOC_soc_pft%d_mean', dtype = np.float32)

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
from affine import Affine
from osgeo import gdal
from scipy.ndimage import zoom
from cached_property import cached_property
from pyl4c.data.fixtures import EASE2_GRID_PARAMS
from pyl4c.ease2 import ease2_coords
from pyl4c.utils import get_pft_array
from pyl4c.spatial import ease2_to_geotiff
from pyl4c.lib.cli import CommandLineInterface, ProgressBar

class NestedGrid(object):
    '''
    Represents a nested L4C data structure, where each grid cell has a mean
    value for each sub-grid PFT class. This allows the reconstruction of a
    finer scale grid by applying the PFT mean values to the subgrid. The
    arguments `subset_id` and `subset_bbox` are mutually exclusive.

    Parameters
    ----------
    pft : tuple
        PFT values to use in downscaling
    shape : Sequence
        (Optional) The intended shape (at 9-km resolution) of the L4C data
        that will be downscaled. If not provided, should the shape of the
        data not match the PFT map, an error will be raised.
    subset_bbox : Sequence
        (Optional) An optional bounding box, `(xmin, ymin, xmax, ymax)`,
        specifying a spatial subset that will be downscaled
    '''
    def __init__(self, pft = range(1, 9), shape = None, subset_bbox = None):
        self._offsets = (0, 0)
        self._pft_codes = pft
        self._shp_1km = EASE2_GRID_PARAMS['M01']['shape']
        self._shp_9km = EASE2_GRID_PARAMS['M09']['shape']
        self._slice_idx_1km = None
        self._slice_idx_9km = None
        self._subset_bbox = subset_bbox
        self._ul_coords = (subset_bbox[0], subset_bbox[-1]) # Upper-left coordinates
        self._lr_coords = (subset_bbox[-2], subset_bbox[1]) # Lower-right coordinates
        # This is the "global" affine transformation
        self._transform_1km = Affine.from_gdal(*EASE2_GRID_PARAMS['M01']['geotransform'])
        self._transform_9km = Affine.from_gdal(*EASE2_GRID_PARAMS['M09']['geotransform'])
        # This is the "output" (1-km) affine transformation
        self._transform = self._transform_1km * self._transform_1km.scale(1)
        if subset_bbox is not None:
            # Need to calculate "local" transformation
            gt = list(self._transform_9km.to_gdal())
            gt[0] = self._ul_coords[0]
            gt[3] = self._ul_coords[1]
            transform_9km = Affine.from_gdal(*gt)
            transform_1km = transform_9km * transform_9km.scale(1/9)
            # Figure out row-column coordinates of upper-left corner and
            #   the 9-km extent
            x0, y0 = list(map(int, ~self._transform_9km * self._ul_coords))
            x1, y1 = list(map(int, ~transform_9km * self._lr_coords))
            self._shp_9km = (y1, x1)
            self._slice_idx_9km = [(y0, y0 + y1), (x0, x0 + x1)]
            # Same for the 1-km extent
            x2, y2 = list(map(int, ~self._transform_1km * self._ul_coords))
            x3, y3 = list(map(int, ~transform_1km * self._lr_coords))
            self._shp_1km = (y3, x3)
            # Check that this is the expected shape; because of coordinate
            #   transformations with different libraries, we may need to
            #   fudge things a big
            if shape is not None:
                if self._shp_9km != shape:
                    print(f'WARNING: Expected shape {str(shape)} did not match actual shape {str(self._shp_9km)}; using expected shape')
                    # Adjust the width and height at the bottom-right corner
                    deltas = np.array(shape) - np.array(self._shp_9km)
                    self._shp_9km = shape
                    self._shp_1km = tuple(np.array(self._shp_9km) * 9)
                    self._slice_idx_9km = [
                        (y0, y0 + y1 + deltas[0]),
                        (x0, x0 + x1 + deltas[1])
                    ]
            self._slice_idx_1km = (np.array(self._slice_idx_9km) * 9).tolist()
            # Update the output 1-km transformation
            self._transform = transform_1km

    @cached_property
    def pft(self):
        'The 1-km PFT map'
        return get_pft_array(
            'M01', slice_idx = self._slice_idx_1km).astype(np.uint8)[np.newaxis,...]

    @property
    def shape(self):
        'Returns the 1-km array shape'
        return self._shp_1km

    def _downscale(self, downscaled, scale = 1, dtype = np.float32, nodata = -9999):
        # Where the PFT map is in the valid range, return data, else NoData
        return np.where(
            np.logical_and(
                np.in1d(self.pft.ravel(), self._pft_codes),
                downscaled.ravel() != nodata),
            np.multiply(downscaled.ravel(), scale), nodata)\
            .reshape(self._shp_1km)\
            .astype(dtype)

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

    def downscale_hdf5_by_pft(*args, **kwargs):
        '''
        DEPRECATED. Use `NestedGrid.downscale_hdf_by_pft()` instead.
        '''
        self.downscale_hdf_by_pft(*args, **kwargs)

    def downscale_hdf_by_pft(
            self, hdf, field, scale = 1, dtype = np.float32, nodata = -9999):
        '''
        Resamples 9-km L4C data to 1-km scale by repeating the spatial mean
        values for each PFT on the 1-km land-cover grid.

        Will subset the *arrays if they are not the expected shape of the
        subset at 9-km scale. This function assumes that the grouped dataset,
        `hdf`, is either an official SPL4CMDL HDF5 granule or a netCDF4 file
        generated by NASA AppEEARS.

        NOTE: Use scale = 1e6 if a total flux (e.g., total GPP) is required;
        `1e6` is the number of square meters in a 1-km L4C pixel.

        Parameters
        ----------
        hdf : h5py.File or netCDF4.Dataset
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
        if self._subset_bbox is not None:
            y_idx, x_idx = self._slice_idx_9km
            ymin, ymax = y_idx
            xmin, xmax = x_idx

        downscaled = np.zeros(self._shp_1km, dtype = dtype)
        for p in self._pft_codes:
            # Allow for the possibility of either netCDF4 or h5py datasets
            if hasattr(hdf, 'variables'):
                source = hdf.variables[field % p]
            else:
                source = hdf[field % p]
            # If needed, subset the 9-km arrays, then down-scale to 1-km
            if self._subset_bbox is None:
                arr = source[:]
            else:
                arr = source[ymin:ymax, xmin:xmax]
            # Resize, multiply by PFT mask, then add to output where != NoData
            downscaled = np.add(
                downscaled, np.multiply(self.pft_mask(p), zoom(arr, **opts)))
        return self._downscale(downscaled, scale, dtype, nodata)

    def downscale_netcdf_by_pft(
            self, hdf, field, scale = 1, dtype = np.float32, nodata = -9999):
        '''
        Resamples 9-km L4C data to 1-km scale by repeating the spatial mean
        values for each PFT on the 1-km land-cover grid.

        This function assumes that the grouped dataset, `hdf`, is a netCDF4
        dataset from a file granule generated by NASA AppEEARS. If the granule
        is a spatial subset, that subset matches the bounds defined by the
        `subset_bbox` to `NestedGrid`. It allows for the netCDF4 granule to
        represent multiple time steps; i.e., arrays can be of the shape
        `(T, M, N)` where `T` is one or more time steps.

        NOTE: Use scale = 1e6 if a total flux (e.g., total GPP) is required;
        `1e6` is the number of square meters in a 1-km L4C pixel.

        Parameters
        ----------
        hdf : netCDF4.Dataset
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
        # Get the shape of the input 9-km arrays
        shp = hdf.variables[field % self._pft_codes[0]].shape
        # In NASA AppEEARS, it is possible to request multiple dates of data,
        #   in which case the resulting netCDF4 granule has 3D arrays where
        #   the first is the time axis
        if len(shp) > 2:
            if len(shp) > 3:
                raise ValueError(f'Too many dimensions in "{field % self._pft_codes[0]}" array; expected 2 or 3')
            # Assumes the first axis is a time axis
            dates = shp[0]
        else:
            dates = 1

        result = np.zeros((dates, *self._shp_1km), dtype = dtype)
        with ProgressBar(dates, 'Downscaling...') as progress:
            for t in range(0, dates):
                downscaled = np.zeros(self._shp_1km, dtype = dtype)
                for p in self._pft_codes:
                    # Allow for the possibility of either netCDF4 or h5py datasets
                    if hasattr(hdf, 'variables'):
                        source = hdf.variables[field % p]
                    else:
                        source = hdf[field % p]
                    if dates > 1:
                        arr = source[t,:]
                    else:
                        arr = source[:]
                    # Resize, multiply by PFT mask, then add to output where != NoData
                    downscaled = np.add(
                        downscaled, np.multiply(self.pft_mask(p), zoom(arr, **opts)))
                result[t] = self._downscale(downscaled, scale, dtype, nodata)
                progress.update(t)
        return result


class CLI(CommandLineInterface):
    '''
    Command-line interface for running the downscaling procedure.

        python resample.py run <hdf5_granule> --field="GPP/gpp_pft%d_mean"
            --subset-id="CONUS"

    Parameters
    ----------
    output_path : str
        The output file path
    pft : list or tuple
        The PFT codes to use in down-scaling; should probably be `range(1, 9)`
        (Default)
    field : str
        A Python formatting string representing the SPL4CMDL HDF5 data field
        name to be down-scaled, e.g., `"SOC/soc_pft%d_mean"` (Default) where
        `"%d"` will be filled-in with the numeric PFT code
    subset_id : str
        The name of a well-known geographic subset, see:
        `pyl4c.data.fixtures.SUBSETS_BBOX`
    scale : int or float
        A number to multiply pixels values against, e.g., `1e6` (1,000 square
        kilometers) to convert (g C m-2 day-1) to (g C day-1)
    nodata : int or float
        The NoData value (Default: -9999)
    dtype : str
        The NumPy data type (Default: `"float32"`)
    verbose : bool
        True to print information about the progress
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
