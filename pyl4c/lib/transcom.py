'''
For all things related to TransCom regions but, in particular, for
the statistical summary of SMAP L4C (or other raster array data) by TransCom
region. The TransCom project seems to be poorly documented, but Carbon
Tracker [1] uses it and provides the data used here. Intended use (Example):

    >>> f = h5py.File(file_path)
    >>> soc = f['SOC/soc_mean']
    >>> tc = TransCom()
    >>> tc.summarize_by_transcom(np.where(soc == -9999, 0, soc), 'M09')
    {'Australia': [736.73413, 914.3717],
     'Eurasia Boreal': [3593.8977, 1461.0577],
      ...
     'Tropical Asia': [1294.8158, 1264.5378]}

Command line use:

    $ python transcom.py summarize ./files/*.h5
        --field="SOC/soc_mean"
        --output-path="./summaries.csv"

    $ python transcom.py summarize ./files/*.h5
        --field="SOC/soc_mean"
        --output-path="./summaries.csv" --summaries="('nanmean',)"

1. https://www.esrl.noaa.gov/gmd/ccgg/carbontracker/download.php
'''

import csv
import os
import re
import h5py
import numpy as np
from osgeo import gdal
from osgeo import gdalconst
from osgeo import osr
from cached_property import cached_property
from scipy.io import netcdf_file
from scipy.ndimage import zoom
from pyl4c.data.fixtures import ANCILLARY_DATA_PATHS
from pyl4c.epsg import EPSG
from pyl4c.spatial import EASE2_GRID_PARAMS, array_to_raster, as_array
from pyl4c.lib.cli import CommandLineInterface, ProgressBar
from pyl4c.lib.tcf import TYPE_MAP, TCFArray

TRANSCOM_DATA = ANCILLARY_DATA_PATHS['transcom_netcdf_path']

class TransCom(object):
    '''
    Convenience class for doing zonal statistics over TransCom regions.
    '''
    # Region codes for non-terrestrial/ off-shore regions
    offshore_region_codes = range(12, 22)
    onshore_region_codes = range(1, 12)
    onshore_region_labels = ( # i.e., regions 1 through 11, inclusive
        'North American Boreal',
        'North American Temperate',
        'South American Tropical',
        'South American Temperate',
        'Northern Africa',
        'Southern Africa',
        'Eurasia Boreal',
        'Eurasia Temperate',
        'Tropical Asia',
        'Australia',
        'Europe')
    onshore_regions = dict(zip(onshore_region_codes, onshore_region_labels))

    def __init__(self):
        f = netcdf_file(TRANSCOM_DATA, 'r')
        self.areas = f.variables['transcom_regions_area'].data.copy()
        regions = f.variables['transcom_regions'].data.copy()
        f.close()

        # TransCom array is oriented south-up, so flip it
        self.data = np.flip(regions.astype(np.int8), 0)

    def __filter__(self, arr):
        'Filters the TransCom regions array to terrestrial codes only'
        return np.where(
            ~np.isin(self.data, self.offshore_region_codes),
            self.data, 0)

    def __resample__(self, arr, shp):
        'Nearest neighbor interpolation to a new pixel size'
        # Nearest neighbor interpolation; even though ndimage's zoom()
        #   gets edge pixels wrong, it is probably the best choice here
        return resize(arr, shp, order = 0, preserve_range = True)

    @cached_property
    def __transcom_1km__(self):
        rast = self.transcom_on_ease2_grid(
            grid = 'M01', terrestrial = True)
        return self.__resample__(
            rast.ReadAsArray(), EASE2_GRID_PARAMS['M01']['shape'])

    @cached_property
    def __transcom_9km__(self):
        rast = self.transcom_on_ease2_grid(
            grid = 'M09', terrestrial = True)
        return self.__resample__(
            rast.ReadAsArray(), EASE2_GRID_PARAMS['M09']['shape'])

    @cached_property
    def __area_by_region__(self):
        'Based on a 9-km EASE-Grid (2.0), calculates area in square meters'
        # Reverse the dictionary so that labels identify region codes
        areas = dict((v, k) for k, v in self.onshore_regions.items())
        transcom = self.__transcom_9km__
        for i, label in self.onshore_regions.items():
            areas[label] = transcom[np.where(transcom == i)].shape[0] * 81e6
        return areas

    @property
    def __reported_area_by_region__(self):
        'Reports area in square meters for each TransCom region'
        # Reverse the dictionary so that labels identify region codes
        areas = dict((v, k) for k, v in self.onshore_regions.items())
        for i, label in self.onshore_regions.items():
            areas[label] = self.areas[i]
        return areas

    def as_ease2_array(self, grid = 'M01'):
        '''
        Returns a numpy.ndarray that has the specified EASE-Grid 2.0 shape,
        with the TransCom classes as values.
        '''
        if grid == 'M01':
            return self.__transcom_1km__
        elif grid == 'M09':
            return self.__transcom_9km__
        else:
            raise ValueError('Requested grid size not available')

    def as_raster(self, terrestrial = True):
        '''
        Returns the TransCom data as a `gdal.Dataset`.

        Parameters
        ----------
        terrestrial : bool
            True to filter to only terrestrial TransCom regions
            (Default: True)

        Returns
        -------
        gdal.Dataset
        '''
        # TransCom (source) spatial reference system
        wkt0 = osr.SpatialReference()
        wkt0.ImportFromEPSG(4326)
        gt0 = (-180, 1, 0, 90, 0, -1)
        return array_to_raster(
            self.__filter__(self.data) if terrestrial else self.data,
            gt0, str(wkt0))

    def count_ease2_by_transcom(
            self, array, grid = 'M01', scale = 1, text_labels = False,
            nodata = -9999):
        '''
        Calculates the number of non-NaN/ non-missing values in each TransCom
        region within the provided array.  Calls
        `TransCom.summarize_ease2_by_transcom()` as a subroutine.

        Parameters
        ----------
        array : numpy.ndarray
            Data array with the same SRS, grid size as desired for the
            TransCom data
        grid : str
            The EASE-Grid 2.0 name for which the TransCom data should be
            aligned and resampled on (e.g., "M01" or "M09")
        scale : int or float
            Optional scaling parameter to apply to the input array values,
            e.g., if the array values are (spatial) rates and should be
            scaled by the (equal) area of the grid cell
        text_labels : bool
            True to use the names of the TransCom regions instead of their
            region codes, in the output
        nodata : int or float
            NoData or Fill value in the array data to ignore

        Returns
        -------
        dict
        '''
        count = lambda arr: np.sum(np.where(np.isnan(arr), 0, 1))
        return self.summarize_ease2_by_transcom(
            array, count, grid, text_labels, nodata)

    def rescaled(self, size_degrees = 1):
        '''
        Rescales the TransCom grid (nominally 1-degree by 1-degree) to a
        scalar multiple equirectangular grid size; e.g., size_degrees = 0.5
        would enlarge the array from (180, 360) to (360, 720).

        Parameters
        ----------
        size_degrees : int or float
            The output (equirectangular) grid resolution in degrees

        Returns
        -------
        numpy.ndarray
        '''
        opts = { # Options to zoom()
            'order': 0,
            'zoom': (1/size_degrees),
            'mode': 'grid-constant',
            'grid_mode': True
        }
        return zoom(self.data, **opts)

    def summarize_ease2_by_transcom(
            self, array, summaries = dict(mean = np.nanmean, std = np.nanstd),
            grid = 'M01', scale = 1, text_labels = False, nodata = -9999):
        '''
        Calculates the mean and standard deviation of the input array values
        within each TransCom class.

        NOTE: The statistical summaries are accumulated as 32-bit floating
        point values, which is fastest, but will not be accurate for certain
        data types. If the input data array are large integers (e.g., soil
        organic carbon/ SOC values), then this should not be a problem.

        Parameters
        ----------
        array : numpy.ndarray
            Data array with the same SRS, grid size as desired for the
            TransCom data
        summaries : dict
            Dictionary of {label: function} for every summary statistic
            desired; function should be NumPy summary function, e.g.,
            nanmean, nansum, ...
        grid : str
            The EASE-Grid 2.0 name for which the TransCom data should be
            aligned and resampled on (e.g., "M01" or "M09")
        scale : int or float
            Optional scaling parameter to apply to the input array values,
            e.g., if the array values are (spatial) rates and should be
            scaled by the (equal) area of the grid cell
        text_labels : bool
            True to use the names of the TransCom regions instead of their
            region codes, in the output
        nodata : int or float
            NoData or Fill value in the array data to ignore

        Returns
        -------
        dict
            Dictionary with TransCom regions as labels and a nested Dictionary
            with a  key-value pair for each desired summary statistic.
        '''
        assert array.ndim == 2 or (array.ndim == 3 and array.shape[0] == 1), 'Can only work with 1-band raster arrays'
        if array.ndim == 3:
            array = array[0,...] # Unwrap 1-band raster arrays

        # Extract grid size in km; get the resampled TransCom array
        g = int(re.compile(r'.*(?P<km>\d{2})').match(grid).groups()[0])
        transcom = getattr(self, '__transcom_%dkm__' % g)
        assert transcom.shape == array.shape, 'Input array does not match the TransCom regions grid at the specified grid size'

        # Fill in NaN where there is NoData
        if nodata is not None:
            array = np.where(array == -9999, np.nan, array)
        # Determine how we will organize statistics by class label
        if text_labels:
            # Create, e.g., {'Australia': {}, ...}
            stats = dict([(v, dict()) for v in self.onshore_regions.values()])
        else:
            # Create, e.g., {1: {}, 2: {}, ...}
            stats = dict([(k, dict()) for k in self.onshore_regions.keys()])

        for code, label in self.onshore_regions.items():
            i = label if text_labels else code
            query = np.multiply( # Scale cell values (Default = 1.0)
                np.where(np.isin(transcom, code), array, np.nan), scale)
            for stat_name, func in summaries.items():
                # NOTE: Runs faster if dtype of accumulator is *not* set
                stats[i][stat_name] = func(query)

        return stats

    def transcom_on_ease2_grid(self, grid = 'M01', terrestrial = True):
        '''
        Projects the equirectangular TransCom regions data onto the EASE-Grid
        2.0 spatial reference system.

        NOTE: EASE-Grid 2.0 only extends to 84 degrees latitude North or
        South, so the TransCom regions ought to be restricted to this same
        extent; but it doesn't actually matter and the projection is still
        accurate. I'm making a note here in case it becomes important later:

            self.__y_coords__ = f.variables['lat'].data.copy()
            idx = np.where(np.abs(self.__y_coords__) < 84)[0]
            transcom = self.data[idx,:]

        Parameters
        ----------
        grid : str
            The EASE-Grid 2.0 name for which the TransCom data should be
            aligned and resampled on (e.g., "M01" or "M09")
        terrestrial : bool
            True to filter to only terrestrial TransCom regions

        Returns
        -------
        gdal.Dataset
        '''
        # EASE-Grid 2.0 (target) spatial reference system
        wkt = osr.SpatialReference()
        wkt.ImportFromWkt(EPSG[EASE2_GRID_PARAMS[grid]['epsg']])

        # Create a gdal.Dataset from TransCom data
        rast0 = self.as_raster(terrestrial)
        gt0 = rast0.GetGeoTransform()
        wkt0 = str(rast0.GetProjection())
        py, px = self.data.shape
        px += 12 # HACK: Output is clipped for some reason (=/)

        # The output (projected) raster's GeoTransform is difficult to
        #   determine, but this should do it automatically
        gt = gdal.AutoCreateWarpedVRT(
            rast0, str(wkt0), str(wkt), gdal.GRA_NearestNeighbour).GetGeoTransform()
        # rast0 is input raster, rast is output raster
        rast = gdal.GetDriverByName('MEM').Create('', px, py, 1, gdalconst.GDT_Int16)
        rast.SetGeoTransform(gt)
        rast.SetProjection(str(wkt))
        gdal.ReprojectImage(
            rast0, rast, str(wkt0), str(wkt), gdalconst.GRA_NearestNeighbour)
        return rast


class TransComCLI(CommandLineInterface):
    '''
    A command-line interface (CLI) for convenience; used with Google fire.
    '''
    def __init__(
            self, output_path = None, field = 'SOC/soc_mean',
            summaries = ('nanmean', 'nanstd'), grid = 'M09', **kwargs):
        self._output_path = output_path
        self._field = field
        self._grid = grid
        self._summaries = summaries
        self._kwargs = kwargs
        self._kwargs['grid'] = grid
        self.tc = TransCom()

    def __check__(self):
        assert self._output_path is not None,\
            'You must specify an output_path with: --output-path=""'
        assert os.path.exists(os.path.dirname(self._output_path)),\
            'Did not recognize output_path (Cannot use shortcuts like ~)'

    def __dump_csv__(self, output_path, items):
        with open(output_path, 'w') as stream:
            writer = csv.writer(stream, delimiter = ',', quotechar = '"')
            writer.writerow(('filename', 'transcom_region', 'statistic', 'value'))
            for each in items:
                writer.writerow(each)

    def __expand_summary__(self, items):
        'Expands ["A", {"a": 1, "b": 2}] into [("A", "a", 1), ("A", "b", 2)]'
        new_items = []
        # TODO Would be nice to generalize this using a recursive pattern
        for filename, d in items:
            for label, summary in d.items():
                for key, value in summary.items():
                    new_items.append((filename, label, key, value))
        return new_items

    def report_areas(self):
        '''
        Prints the reported area of each TransCom region.
        '''
        # NOTE: The underlying data are 32-bit floating point, so this
        #   string formatting will maximize the precision
        for k, v, in self.tc.__reported_area_by_region__.items():
            print('%s: %.0f' % (k, v))

    def summarize(self, *file_paths):
        '''
        Creates a statistical summary by class label (TransCom region) for
        each of multiple input HDF5 files. Use:

            $ python transcom.py summarize ./files/*.h5 --field="SOC/soc_mean"
                --output-path="./summaries.csv"

            $ python transcom.py summarize ./files/*.h5 --field="SOC/soc_mean"
                --output-path="./summaries.csv" --summaries="('nanmean',)"

        Parameters
        ----------
        *file_paths : str
        field : str
            Hierarchical path name of field to summarize: `--field="<field>"`
        summaries : str
            For compatibility at the CLI, can describe the statistical summary
            functions desired as a comma-delimited string, e.g.,
            `"('nanmean','nanstd')"` where names refer to NumPy functions
        output_path : str
            File path where the output CSV file should be written
        **kwargs : str
            Other keyword arguments to be passed on to
            `TransCom.summarize_ease2_by_transcom()`
        '''
        self.__check__()
        results = []
        n = len(file_paths)
        assert n > 0, 'No file paths given (Did you forget to specify "field" argument?)'
        # NOTE: fire.Fire() implicitly transforms comma-delimited string into
        #   tuple; here, e.g., "('nanmean', 'nanstd')" becomes:
        #   {'nanmean': np.nanmean, 'nanstd': np.nanstd}
        self._kwargs['summaries'] = dict([
            (name, getattr(np, name)) for name in self._summaries
        ])

        # Determine what kind of files we're working with
        mode = 'other'
        if file_paths[0].split('.')[-1] == 'h5':
            mode = 'hdf5'
        elif file_paths[0].split('.')[-1] in TYPE_MAP.keys():
            mode = 'sparse'

        with ProgressBar(len(file_paths), 'Summarizing files...') as progress:
            for i, filename in enumerate(file_paths):
                if mode == 'hdf5':
                    with h5py.File(filename, 'r') as hdf:
                        arr = hdf[self._field][:]

                elif mode == 'sparse':
                    tcf = TCFArray(filename, self._grid)
                    tcf.inflate()
                    arr = tcf.data

                elif mode == 'other':
                    arr, _, _ = as_array(filename, band_axis = False)

                stats = self.tc.summarize_ease2_by_transcom(arr, **self._kwargs)
                results.append((filename, stats))
                progress.update(i + 1)

        self.__dump_csv__(self._output_path, self.__expand_summary__(results))


if __name__ == '__main__':
    # Performance notes:
    #   - Does run faster if the dtype of the accumulator function
    #       (e.g., np.nanmean) is NOT set
    #   - Does NOT run faster if arrays are raveled prior to summary
    import fire
    fire.Fire(TransComCLI)
