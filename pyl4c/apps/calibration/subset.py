'''
DEPRECATED in favor of updating the Version 4 calibration dataset with the
new Tsoil, SMRZ, and SMSF fields of Nature Run v8.1.

Compiles the data needed for calibrating L4C, everything except the initial
state of the SOC pools, into a single HDF5 file.

    python pyl4c/apps/calibration/main.py run --ts_start="2015-03-01"
        -ts_end="2019-12-31"

Compiles:

1. MODIS fPAR data;
2. L4SM Nature Run (v8.1) soil moisture and temperature;
3. MERRA-2 surface meteorological driver data;
4. Annual NPP sum (for litterfall);

Sources (https://disc.gsfc.nasa.gov/):

    MERRA-2 (Tmin)
        M2SDNXSLV: MERRA-2 statD_2d_slv_Nx: 2d,Daily,Aggregated Statistics,Single-Level,Assimilation,Single-Level Diagnostics V5.12.4
    MERRA-2 (PS, QV2M, T2M)
        M2T1NXSLV: MERRA-2 tavg1_2d_slv_Nx: 2d,Hourly,Single-level
    MERRA-2 (SWGDN)
        M2T1NXLFO: MERRA-2 tavg1_2d_lfo_Nx: 2d,Hourly,Single-level

MERRA-2 Documentation:

    https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXINT.5.12.4/doc/MERRA2.README.pdf

Conversion of down-welling short-wave radiation: As 1 Watt m-2 is equal to
1 Joule second-1), 11.5741 is derived as:

    1 / ((60 secs * 60 mins * 24 hours) / 1e6)

i.e., 1.0 J s-1 multiplied by the number of seconds in a day, converted to
Megajoules (MJ) through dividing by 1e6, then taking the inverse, as someone
preferred division over multiplication. Then, multiply by 0.45 on the
assumption that PAR is 45% of down-welling short-wave radiation. Turns out
this multiplier, used with a 24-hour mean, produces a PAR distribution that
is extremely close to that produced by using the 24-hour sum, which makes
more sense.
'''

import csv
import datetime
import glob
import os
import fire
import numpy as np
import netCDF4
import h5py # Due to bug in netCDF4, cannot import this module before netCDF4
from osgeo import gdal
from osgeo import gdalconst
from cached_property import cached_property
from pyl4c.epsg import EPSG
from pyl4c.utils import get_pft_array, index
from pyl4c.spatial import array_to_raster, intersect_rasters
from pyl4c.towers import TowerSite
from pyl4c.lib.cli import CommandLineInterface, ProgressBar
from pyl4c.lib.netcdf import netcdf_file, netcdf_raster
from pyl4c.apps.calibration.legacy import LegacyTowerDataset
from pyl4c.apps.calibration.nature import NatureRunNetCDF4
from pyl4c.data.fixtures import EASE2_GRID_PARAMS

# Path to CSV file of tower sites and 3-tuple of X-, Y-coordinate, and
#   identifier, corresponding to names in the CSV header file
TOWER_SITE_LIST = '/anx_lagr3/arthur.endsley/SMAP_L4C/calibration/FluxTower_sites_Vv4040_locations.csv'
TOWER_SITE_LIST_FIELDS = ('longitude', 'latitude', 'assumed_name')
TOWER_SITE_FILE_GLOB = '/anx_lagr2/laj/smap/fluxnet2015/merged/*_smapMergedCalVal.csv'

# Template for all L4SM file paths; should have a single %s formatting
#   character where the YYYYMMDD timestamp will go
L4SM_FILE_TPL = '/anx_lagr3/arthur.endsley/SMAP_L4SM/NatureRun/NRv8.1/tiled/SMAP_Nature_v8.1.tavg3_1d_lnr_Nt.%s.nc4'

# Original: /anx_lagr3/laj/smap/natv72/prelaunch/land/tcf_natv72_npp_sum_M01.flt32
# Inflated using: /anx_v2/laj/smap/code/mkgrid/src/mkgrid
NPP_SUM_FILE = '/home/arthur.endsley/L4C_experiments/tcf_natv72_npp_sum_M01.flt32'

METADATA = {
    'fpar': {
        'file_path': '/anx_lagr_pub/data/SKYentists/L4C_calibration_inputs/202002/L4_C_input_NRv72_for_356_tower_sites_smrz100.h5',
        'src_name': 'MOD/fpar',
    },
    'ft': {
        'file_glob': '/anx_lagr4/SMAP/L4C_drivers/MERRA-2/tavg1_2d_lnd_Nx/MERRA2_*.tavg1_2d_lnd_Nx.%s.nc4.nc4',
        'src_keys': ['TSURF', 'lon', 'lat'],
        'transform': lambda x: x <= 273.15
    },
    'par': {
        'file_tpl': '/anx_lagr4/SMAP/L4C_drivers/MERRA-2/tavg1_2d_lfo_Nx/MERRA2_*.tavg1_2d_lfo_Nx.%s.nc4.nc4',
        'src_keys' : ['SWGDN', 'lon', 'lat'],
        'units': 'MJ m-2 day-1',
        'transform': lambda x: np.divide(np.multiply(x, 0.45), 11.5741),
    },
    'vpd': {

    },
    'tmin': { # M2SDNXSLV
        'file_glob': '/anx_lagr4/SMAP/L4C_drivers/MERRA-2/statD_2d_slv_Nx/MERRA2_*.statD_2d_slv_Nx.%s.nc4',
        'src_keys': ['T2MMIN', 'lon', 'lat'],
    },
    'smrz': {
        'src_name': 'sm_rootzone',
        'long_name': 'Soil moisture at the root zone',
        'units': 'm3 m-3',
    },
    'smsf': {
        'src_name': 'sm_surface',
        'long_name': 'Soil moisture at the surface',
        'units': 'm3 m-3',
    },
    'tsoil': {
        'src_name': 'soil_temp_layer1',
        'long_name': 'Soil temperature in layer 1',
        'units': 'degrees K',
    }
}

class CLI(CommandLineInterface):
    def __init__(
            self, output_hdf5_path, version_id = 'v5', debug = False,
            ts_start = '2000-01-01', ts_end = '2015-12-31'):
        if not debug:
            gdal.PushErrorHandler('CPLQuietErrorHandler')

        self._debug = debug
        self._fields = (
            'freeze_thaw', 'tmin', 'vpd', 'smrz', 'tsoil', 'smsf', 'par',
            'fpar')
        self._output_path = output_hdf5_path
        self._origin = datetime.datetime(1970, 1, 1, 0, 0, 0)
        self._pft_1km = get_pft_array('M01')
        self._pft_9km = get_pft_array('M09')
        self._sites = []
        self._sites_skipped_idx = []
        self._site_count = 0
        self._site_idx_1km = [] # ROUNDED 1-km EASE-Grid 2 row, column indices
        self._site_idx_9km = [] # ROUNDED 9-km EASE-Grid 2 row, column indices
        self._site_idx_9km_duplicates = []
        self._site_idx_1km_subgrid = [] # N x 81 array of the 1-km PFT subgrid
        self._site_file_paths = glob.glob(TOWER_SITE_FILE_GLOB)
        self._time_series_bounds = (
            datetime.datetime.strptime(ts_start, '%Y-%m-%d'),
            datetime.datetime.strptime(ts_end, '%Y-%m-%d'))
        self._time_series = []

        assert len(self._site_file_paths) > 0,\
            'File not found: No route to the tower site files'
        assert os.path.exists(os.path.dirname(L4SM_FILE_TPL)),\
            'File not found: No route to the L4SM data directory'

        # The HDF5 file to write out
        self._hdf = h5py.File(output_hdf5_path, 'w')
        self.prep_sites() # Load the tower site data
        self.prep_time_series() # Count the number of time steps

    @cached_property
    def reference_ease2(self):
        'A token EASE-Grid 2.0 global 9-km raster'
        gt, epsg, shp = [
            EASE2_GRID_PARAMS['M09'][k] for k in ('geotransform', 'epsg', 'shape')
        ]
        wkt = EPSG[epsg]
        return array_to_raster(np.zeros(shp), gt, wkt)

    def assemble_fpar(self):
        'Cheat: Uses fPAR from the previous calibration dataset'
        print('Assembling fPAR...')
        n = len(self._time_series)
        with LegacyTowerDataset(
                self._sites, None, skip = self._sites_skipped_idx) as stream:
            fpar = stream.fpar_daily()
            return (None, fpar[0:n,...])

    def assemble_l4sm_fields(self):
        'Subsets L4SM fields (SMRZ, SMSF, and Tsoil) each tower site each day'
        results = [[] for i in range(0, 4)]
        indices = [
            (int(r), int(c)) if r is not None else (None, None)
            for r, c in self._site_idx_9km
        ]
        time_axis = []
        with ProgressBar(
                len(self._time_series), '[L4SM] Processing...') as progress:

            for i, t in enumerate(self._time_series):
                filename = L4SM_FILE_TPL % t.strftime('%Y%m%d')
                if not os.path.exists(filename):
                    time_axis.append(np.nan)
                    continue # Skip

                # Pre-compute the tile-space indices, so we're not doing this
                #   over and over again for each netCDF4 file
                if i == 0:
                    nc0 = NatureRunNetCDF4(
                        L4SM_FILE_TPL % t.strftime('%Y%m%d'))
                    tile_idx = [ # This is what takes the longest
                        nc0.ease2_to_tile(row, col)
                        if row is not None and col is not None else None
                        for row, col in indices
                    ]

                # Record the number of days since the epoch
                time_axis.append((t - self._origin).days)
                nc = NatureRunNetCDF4(filename)
                for j, field in enumerate(('smsf', 'smrz', 'tsoil')):
                    variable = METADATA[field]['src_name']
                    # 4 x T x N
                    results[j].append(nc0.index_bulk(nc, variable, tile_idx))

                # And add daily minimum SMRZ
                results[3].append(nc0.index_bulk(nc, 'smrz', tile_idx, 'min'))
                progress.update(i)

        return (np.array(time_axis), np.array(results))

    def assemble_merra2_fields(self):
        'Subsets MERRA2 PAR, VPD, FT, TMIN fields each tower site each day'
        def is_nc4(filename):
            return os.path.basename(filename).split('.')[-1] == 'nc4'

        def resample_and_index(raster):
            arr = intersect_rasters( # Resample onto 9-km global EASE-Grid 2.0
                self.reference_ease2, raster,
                method = gdalconst.GRA_NearestNeighbour).ReadAsArray()
            return [ # Extract data at each tower site
                arr[int(r), int(c)] if r is not None else np.nan
                for r, c, in self._site_idx_9km
            ]

        time_axis = []
        all_years = np.unique(
            np.array([int(t.strftime('%Y')) for t in self._time_series]))
        all_years.sort()
        days_by_year = {} # Create a lookup of the days in each year
        for t in self._time_series:
            # e.g., {2015: [1, 2, ..., 365], 2016: [1, 2, ..., 366]}
            days_by_year.setdefault(t.year, [])
            days_by_year[t.year].append(int(t.strftime('%j')))

        # PAR and VPD
        results = [[] for i in range(0, 2)]
        for i, field in enumerate(('par', 'vpd')):
            with ProgressBar(len(all_years),
                    '[MERRA-2] Processing "%s"...' % field) as progress:
                for j, year in enumerate(all_years):
                    # PAR and VPD there is a single file for each year
                    nc = netcdf_file(METADATA[field]['file_tpl'] % year)
                    # For 1, 2, ...365 or 366 in this year...
                    for day in days_by_year[year]:
                        now = datetime.datetime.strptime(
                            '%d%s' % (year, str(day).zfill(3)), '%Y%j')
                        # Record the number of days since the epoch
                        time_axis.append((now - self._origin).days)
                        # Get the integer index of the day of interest
                        t = np.argwhere(nc.variables['day'][:] == day)[0][0]
                        # Extract as a raster array the time step for this day
                        rast0 = netcdf_raster(
                            nc, METADATA[field]['src_keys'], time_idx = t)

                        # Resample each var on a 9-km global EASE-Grid 2.0
                        if 'transform' not in METADATA[field].keys():
                            results[i].append(resample_and_index(rast0))
                            continue

                        # PAR must be taken as a fraction of down-welling SW rad.
                        results[i].append(
                            np.apply_along_axis(
                                METADATA[field]['transform'], 0,
                                resample_and_index(rast0)))

                    progress.update(j)

        # FT and Tmin
        results.extend([[], []])
        for i, field in enumerate(('ft', 'tmin')):
            with ProgressBar(len(all_years),
                    '[MERRA-2] Processing "%s"...' % field) as progress:
                for j, year in enumerate(all_years):
                    for day in days_by_year[year]:
                        now = datetime.datetime.strptime(
                            '%d%s' % (year, str(day).zfill(3)), '%Y%j')
                        # Get the filename with this time step's YYYYMMDD string
                        filename = glob.glob(
                            METADATA[field]['file_glob'] % now.strftime('%Y%m%d')
                        ).pop()
                        nc = netCDF4.Dataset(filename)
                        rast0 = netcdf_raster( # Daily files have only 1 time
                            nc, METADATA[field]['src_keys'], time_idx = 0)

                        # Resample each var on a 9-km global EASE-Grid 2.0
                        if 'transform' not in METADATA[field].keys():
                            results[i + 2].append(resample_and_index(rast0))
                            continue

                        # FT must be transformed from Tsurf
                        results[i + 2].append(
                            np.apply_along_axis(
                                METADATA[field]['transform'], 0,
                                resample_and_index(rast0)))

                    progress.update(j)

        return (np.array(time_axis), np.array(results))

    def assemble_npp_sum(self):
        'Assembles the annual NPP sum for each 1-km subgrid'
        print('Getting annual NPP sum...')
        npp_sum_array = np.fromfile(NPP_SUM_FILE, dtype = np.float32)\
            .reshape(EASE2_GRID_PARAMS['M01']['shape'])
        result = []
        for site in self._sites:
            result.append(index(npp_sum_array, site.subgrid))

        return np.concatenate([ # Want an N x 81
            np.array(s).reshape((1, 81)) for s in result
        ], axis = 0)

    def prep_time_series(self):
        'Creates a time series for the calibration dataset'
        self._time_series = []
        # Use only the first file as a reference
        with open(self._site_file_paths[0], 'r') as stream:
            reader = csv.DictReader(stream)
            for record in reader:
                contents = map(int,
                    (record['year'], record['month'], record['day']))

                t = datetime.datetime(*contents)
                if t < self._time_series_bounds[0] or t > self._time_series_bounds[1]:
                    continue # Skip to next
                self._time_series.append(t)

    def prep_sites(self):
        'Load the tower site data'
        x_field, y_field, id_field = TOWER_SITE_LIST_FIELDS
        print('Loading tower sites...')
        with open(TOWER_SITE_LIST, 'r') as stream:
            reader = csv.DictReader(stream)
            self._sites_skipped_idx = []
            for i, record in enumerate(reader):
                name = record[id_field]
                lng, lat = (float(record[x_field]), float(record[y_field]))
                try:
                    tower = TowerSite(name, (lng, lat))
                except ValueError:
                    print('Skipping site %s...' % name)
                    self._sites_skipped_idx.append(i)
                    continue

                # Store this TowerSite
                self._sites.append(tower)

                # Check for duplicates (see if this row, column pair has been
                #   indexed already
                row, col = tower.idx_9km
                try:
                    # Coerce row, column indices to integer to find collisions
                    idx = self._site_idx_9km.index((int(row), int(col))) if row is not None else None
                except ValueError:
                    idx = None

                # Record the first index of a matching grid cell or None
                self._site_idx_9km_duplicates.append(idx)
                self._site_idx_9km.append((int(row), int(col)))
                self._site_count += 1

    def run(self):
        '''
        Assembles the state and driver data for L4SM; writes the data to an
        output HDF5 file.
        '''
        # Read the current contents of the file
        present_drivers = dict()
        present_state = dict()
        with h5py.File(self._output_path, 'r') as hdf:
            drivers = state = list()
            if 'drivers' in hdf.keys():
                present_drivers = hdf['drivers'].keys()
            if 'state' in hdf.keys():
                present_state = hdf['state'].keys()

        # STEP 1: Assemble the fPAR data
        if 'fpar' not in present_drivers:
            _, fpar_data = self.assemble_fpar()
            with h5py.File(self._output_path, 'a') as hdf:
                shp = (len(self._time_series), self._site_count, 81)
                hdf.create_dataset('drivers/fpar', shp, dtype = 'float32',
                    data = fpar_data)

        # STEP 2: Assemble the L4SM Nature Run data fields
        if not all(k in present_drivers for k in ('smsf', 'smrz', 'tsoil')):
            # Check that file data is available for every time step
            exists_l4sm = [
                os.path.exists(L4SM_FILE_TPL % t.strftime('%Y%m%d'))
                for t in self._time_series
            ]
            assert all(exists_l4sm), 'One or more time steps has no L4SM data file'
            # Gets (smsf, smrz, and tsoil) x T x N array
            _, l4sm_data = self.assemble_l4sm_fields()
            with h5py.File(self._output_path, 'a') as hdf:
                shp = (len(self._time_series), self._site_count)
                for i, key in enumerate(('smsf', 'smrz', 'tsoil', 'srmz_min')):
                    if key not in hdf['drivers']:
                        hdf.create_dataset(
                            'drivers/%s' % key, shp, dtype = 'float32',
                            data = l4sm_data[i,...])

                # Get the site's long-term minimum smrz
                smrz_min = np.nanmin(l4sm_data[1,...], axis = 0)
                hdf.create_dataset('legacy/smrz_min', (self._site_count,),
                    dtype = 'float32', data = smrz_min)

        # STEP 3: Assemble the remaining met. drivers from MERRA-2
        if not all (k in present_drivers for k in ('par', 'vpd', 'ft', 'tmin')):
            for field in ('par', 'vpd'):
                exists_merra2 = [
                    os.path.exists(METADATA[field]['file_tpl'] % year)
                    for year in range(
                        self._time_series_bounds[0].year,
                        self._time_series_bounds[1].year + 1)
                ]
                assert all(exists_merra2), 'One or more time steps has no MERRA-2 PAR or VPD data file'
            for field in ('ft', 'tmin'):
                exists_merra2 = [
                    glob.glob(
                        METADATA[field]['file_glob'] % t.strftime('%Y%m%d')
                    ).pop()
                    for t in self._time_series
                ]
                assert all(exists_merra2), 'One or more time steps has no MERRA-2 FT or Tmin data file'
            # Gets (par, vpd, ft, tmin) x T x N array
            time_axis, merra2_data = self.assemble_merra2_fields()
            with h5py.File(self._output_path, 'a') as hdf:
                shp = (len(self._time_series), self._site_count)
                for i, key in enumerate(('par', 'vpd', 'ft', 'tmin')):
                    if key not in hdf['drivers'].keys():
                        hdf.create_dataset(
                            'drivers/%s' % key, shp, dtype = 'float32',
                            data = merra2_data[i,...])

        # STEP 4: Add the annual NPP sum
        if 'npp_sum' not in present_state.keys():
            shp = (self._site_count, 81)
            with h5py.File(self._output_path, 'a') as hdf:
                hdf.create_dataset('state/npp_sum', shp, dtype = 'float32',
                    data = self.assemble_npp_sum())

        # STEP 5: Record (year, month, day, hour)
        print('Writing output HDF5 file...')
        with h5py.File(self._output_path, 'a') as hdf:
            if 'time' in hdf.keys():
                del hdf['time']
            hdf.create_dataset('time', (4, len(self._time_series)),
                dtype = 'int16', data = np.array([
                    (d.year, d.month, d.day, 0) for d in self._time_series
                ]))

            # Write the site IDs...this, unfortunately, has to be verbose
            if 'site_id' in hdf.keys():
                del hdf['site_id']
            dt = h5py.string_dtype(encoding = 'utf-8')
            dataset = hdf.create_dataset('site_id', (len(self._sites),), dtype = dt)
            for i, tower in enumerate(self._sites):
                dataset[i] = tower.id

            # Store subgrid PFTs
            if 'state/pft' in hdf.keys():
                del hdf['state/pft']
            hdf.create_dataset('state/pft', (len(self._sites), 81), dtype = 'int16',
                data = np.concatenate([
                    tower.subgrid_pft.reshape((1, 81))
                    for tower in self._sites
                ], axis = 0))

            # Store the geographic coordinates
            if 'coords/lng_lat' in hdf.keys():
                del hdf['coords/lng_lat']
            hdf.create_dataset('coords/lng_lat', (len(self._sites), 2),
                dtype = 'float32',
                data = np.vstack([tower.coords for tower in self._sites]))

            # Store the 9-km row-column indices
            if 'coords/grid_9km_idx' in hdf.keys():
                del hdf['coords/grid_9km_idx']
            hdf.create_dataset('coords/grid_9km_idx', (len(self._sites), 2),
                dtype = 'float32',
                data = np.vstack([tower.idx_9km for tower in self._sites]))

            # Store the 1-km subgrids
            if 'coords/grid_1km_subgrid_row_idx' in hdf.keys():
                del hdf['coords/grid_1km_subgrid_row_idx']
            if 'coords/grid_1km_subgrid_col_idx' in hdf.keys():
                del hdf['coords/grid_1km_subgrid_col_idx']
            hdf.create_dataset(
                'coords/grid_1km_subgrid_row_idx', (len(self._sites), 81),
                dtype = 'int16', data = np.array([
                    [int(r) for r, c in tower.subgrid]
                    for tower in self._sites
                ]).reshape((len(self._sites), 81)))
            hdf.create_dataset(
                'coords/grid_1km_subgrid_col_idx', (len(self._sites), 81),
                dtype = 'int16', data = np.array([
                    [int(c) for r, c in tower.subgrid]
                    for tower in self._sites
                ]).reshape((len(self._sites), 81)))


if __name__ == '__main__':
    fire.Fire(CLI)
