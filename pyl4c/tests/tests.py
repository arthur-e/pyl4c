import datetime
import hashlib
import os
import unittest
import h5py
import numpy as np
import pyl4c
from scipy.linalg import solve_banded
from scipy.sparse import dia_matrix
from pyl4c import haversine, pft_dominant, equal_or_nan
from pyl4c.ease2 import ease2_from_wgs84, ease2_to_wgs84, ease2_coords, ease2_search_radius, translate_row_col_to_ease2
from pyl4c.science import arrhenius, bias_correction_parameters, rescale_smrz, vpd, ordinals365, climatology365
from pyl4c.spatial import ease2_coords_approx
from pyl4c.utils import MockL4CGranule, composite, get_ease2_coords, get_ease2_slice_idx, get_ease2_slice_offsets, get_pft_array, get_slice_idx_by_bbox, index, partition, partition_generator, subset, summarize, summarize_by_class
from pyl4c.data.fixtures import ANCILLARY_DATA_PATHS as PATHS, restore_bplut
from l4c import L4CForwardProcessPointTestSuite
from transpiration import TranspirationTestSuite

ANCILLARY_DATA_FOUND = os.path.exists(
    PATHS['smap_l4c_ancillary_data_file_path'])
BPLUT_PATH = os.path.join(
    os.path.dirname(pyl4c.__file__),
    'data/files/SMAP_BPLUT_2020-07-31.csv')
BPLUT_CSV_FOUND = os.path.exists(BPLUT_PATH)
BBOX_CONUS = [-124.5, 24.4, -66.7, 50.0]

class CoordinateTransformations(unittest.TestCase):
    def test_ease2_coords(self):
        'Integration: Should generate EASE-Grid 2.0 arrays of WGS84 coords.'
        x, y = ease2_coords('M09')
        self.assertEqual(x.size, 3856)
        self.assertEqual(y.size, 1624)
        self.assertEqual(x.min().round(2), -179.95)
        self.assertEqual(x.max().round(2), 179.95)
        self.assertEqual(y.min().round(2), -84.66)
        self.assertEqual(y.max().round(2), 84.66)
        x, y = ease2_coords('M01')
        self.assertEqual(x.size, 34704)
        self.assertEqual(y.size, 14616)
        self.assertEqual(x.min().round(2), -179.99)
        self.assertEqual(x.max().round(2), 179.99)
        self.assertEqual(y.min().round(2), -85.0)
        self.assertEqual(y.max().round(2), 85.0)

    def test_ease2_search_radius(self):
        'Should accurately identify pixel coordinates in a search radius'
        coords = ease2_search_radius((-83, 42), 1)
        self.assertTrue(np.equal(coords,
            np.array([
                [267, 1038], [267, 1039], [268, 1038], [268, 1039],
                [267, 1037], [268, 1037], [266, 1038], [266, 1039],
                [266, 1037]
            ])).all())

    def test_translate_row_col_to_ease2(self):
        'Should correctly translate row-column coords to EASE-Grid 2.0 coords'
        ul = translate_row_col_to_ease2((0,0))
        lr = translate_row_col_to_ease2((1624,3856))
        self.assertTrue(np.equal(
            np.array(ul).round(1),
            np.array([-17363026.4,   7310036.8])).all())
        self.assertTrue(np.equal(
            np.array(lr).round(1),
            np.array([17372034.5, -7319044.9])).all())

    def test_ease2_grid_centering(self):
        'Integration: EASE-Grid 2.0 transforms should target grid cell center'
        r, c = ease2_from_wgs84(ease2_to_wgs84((0, 0), 'M09'), 'M01')
        self.assertTrue(r == 4 and c == 4)

    def test_ease2_grid_location_from_wgs84(self):
        'Should accurately locate row-column pair from WGS84 coordinate pair'
        r, c = ease2_from_wgs84((0, 0))
        self.assertTrue(r == 812 and c == 1928)
        r, c = ease2_from_wgs84((0, 0), exact = True)
        self.assertTrue(r == 811.5 and c == 1927.5)


class Data(unittest.TestCase):
    @unittest.skipIf(not BPLUT_CSV_FOUND, 'No path to recent BPLUT file')
    def test_restore_bplut(self):
        'Should read in a BPLUT stored as a CSV file'
        bp = restore_bplut(BPLUT_PATH)
        self.assertTrue(equal_or_nan(
            bp['LUE'].round(2),
            np.array([
                [ np.nan, 1.17, 1.4 , 2.03, 1.33, 0.95, 1.64, 2.49, 2.49, np.nan]
            ], dtype = np.float32)).all())


class Default(unittest.TestCase):
    def test_haversine(self):
        'Should accurately calculate the great circle distance'
        p1, p2 = ((-83.748333, 42.281389), (-83.045833, 42.331389))
        dist = haversine(p1, p2, radius = 6371e3)
        self.assertEqual(round(dist, 2), 58036.75)

    def test_pft_dominant(self):
        'Should accurately report dominant PFT'
        np.random.seed(9)
        pft = np.random.randint(0, 10, 50).reshape((5,10))
        self.assertTrue(
            np.equal(pft_dominant(pft), np.array([6, 8, 8, 8, 8])).all())


class Science(unittest.TestCase):
    def test_ordinals365(self):
        'Should correctly calculate a 365-day day-of-year index'
        dates = [
            datetime.date(2000, 1, 1) + datetime.timedelta(days = d)
            for d in range(0, 366 + 365)
        ]
        ordinals = ordinals365(dates)
        self.assertEqual(len(ordinals), 366 + 365)
        self.assertEqual(max(ordinals), 365)
        self.assertEqual(min(ordinals), 1)

    def test_climatology365(self):
        'Should correctly calculate a 365-day climatology'
        np.random.seed(9)
        arr = 10 * np.random.sample(366 + (365 * 2)) # e.g., 3 years
        dates = [
            datetime.date(2000, 1, 1) + datetime.timedelta(days = d)
            for d in range(0, 366 + (365 * 2))
        ]
        clim = climatology365(arr, dates)
        self.assertEqual(clim.mean().round(3), 5.032) # Same as arr.mean()
        idx = np.random.randint(0, arr.size, 90)
        arr[idx] = np.nan
        clim2 = climatology365(arr, dates)
        self.assertEqual(clim2.mean().round(3), 5.008)

    def test_arrhenius(self):
        'Should accurately calculate Arrhenius response to temperature'
        arr = arrhenius(np.arange(0, 25, 5) + 273.15, beta0 = 273.15).round(3)
        self.assertTrue(np.equal(
            np.array([0.166, 0.296, 0.478, 0.712, 1]), arr).all())
        arr = arrhenius(np.arange(0, 25, 5) + 273.15, beta0 = 288).round(3)
        self.assertTrue(np.equal(
            np.array([0.15 , 0.277, 0.459, 0.699, 1]), arr).all())

    def test_bias_correction_parameters(self):
        'Should calculate correct bias correction parameters'
        np.random.seed(9)
        arr = np.random.randint(0, 30, 30).reshape((15,2))
        self.assertTrue(np.equal(
            bias_correction_parameters(arr).round(3),
            np.array([7.353, 0.781])).all())
        np.random.seed(42)
        arr = np.random.randint(0, 30, 30).reshape((15,2))
        self.assertTrue(np.equal(
            bias_correction_parameters(arr, npoly = 2).round(3),
            np.array([-1.834,  1.036,  0.])).all())

    def test_rescale_smrz(self):
        'Should correctly re-scale root-zone soil moisture'
        np.random.seed(7)
        A = np.random.randint(0, 100, 9).reshape((3,3))
        self.assertTrue(np.equal(
            rescale_smrz(A, smrz_min = np.ones((1,3)) * 25).round(1),
            np.array([
                [75.2, 88.7,  5. ],
                [88.2, 94.8,  5. ],
                [97.7, 82.7,  5. ]])).all())
        self.assertTrue(np.equal(
            rescale_smrz(A, smrz_min = np.ones((1,3)) * 23).round(1),
            np.array([
                [76.4, 89.1, 31.4],
                [88.6, 94.9,  5. ],
                [97.8, 83.4,  5. ]])).all())
        self.assertTrue(np.equal(
            rescale_smrz(A, smrz_min = np.ones((1,3)) * 14).round(1),
            np.array([
                [80.6, 90.5, 59. ],
                [90.2, 95.5, 55.2],
                [98. , 85.9,  5. ]])).all())

    def test_tridiagonal_solver(self):
        'NOT a test of pyl4c; a test that pyl4c dependencies work as expected'
        A = np.array([ # Example from GitHub user cbellei
            [10, 2, 0, 0], [3, 10, 4, 0], [0, 1, 7 ,5], [0, 0, 3, 4]
        ], dtype = np.float32)
        B = np.array([ # Example from scipy.linalg.solve_banded()
            [5, 2, -1, 0, 0], [1, 4, 2, -1, 0], [0, 1, 3, 2, -1],
            [0, 0, 1, 2, 2], [0, 0, 0, 1, 1]
        ], dtype = np.float32)
        a0 = np.array([3, 4, 5, 6])
        b0 = np.array([0, 1, 2, 2, 3])
        sol1 = np.linalg.solve(A, a0)
        sol2 = solve_banded((1, 1), np.flipud(dia_matrix(A).data), a0)
        self.assertTrue(np.equal(sol1.round(4), sol2.round(4)).all())
        sol1 = np.linalg.solve(B, b0)
        sol2 = solve_banded((1, 2), np.flipud(dia_matrix(B).data), b0)
        self.assertTrue(np.equal(sol1.round(4), sol2.round(4)).all())

    def test_vpd(self):
        'Sould correctly calculate VPD'
        self.assertEqual(round(vpd(0.1, 1000, 273.15), 2), 459.14)
        self.assertEqual(round(vpd(0.1, 1000, 278.15), 2), 720.41)
        self.assertEqual(round(vpd(0.2, 1000, 278.15), 2), 585.27)
        self.assertEqual(round(vpd(0.2, 2000, 278.15), 2), 298.57)

class Subsetting(unittest.TestCase):
    bbox = BBOX_CONUS
    x_coords = np.arange(-179.5, 180, 1)
    y_coords = np.arange(-89.5, 90, 1)

    @classmethod
    def setUpClass(cls):
        cls.approx_ease2_coords = ease2_coords_approx('M09', in_1d = True)
        # NOTE: There is potential circularity here; if get_ease2*() functions
        #   aren't working then we can't create a mock
        cls.hdf_9km = MockL4CGranule()
        # TECHNICALLY sets up an integration test, as we're using a feature
        #   in spatial module
        cls.hdf_9km_sorta = MockL4CGranule(
            coords = ease2_coords_approx('M09', in_1d = False))

    @classmethod
    def tearDownClass(cls):
        cls.hdf_9km.close()
        cls.hdf_9km_sorta.close()

    def test_index(self):
        'Should index an array given arbitrary row, column indices'
        arr = np.arange(0, 36).reshape((6, 6))
        idx = [(4, 2, 3), (4, 3, 2)]
        self.assertTrue(np.equal([28, 15, 20], index(arr, idx)).all())

    @unittest.skipIf(not ANCILLARY_DATA_FOUND, 'No path to ancillary data')
    def test_get_ease2_coords(self):
        'Should retrieve EASE-Grid 2.0 coordinate arrays'
        kwargs = {'in_1d': True}
        x1, y1 = get_ease2_coords(grid = 'M01', **kwargs)
        x9, y9 = get_ease2_coords(grid = 'M09', **kwargs)
        self.assertAlmostEqual(round(float(np.min(x1)), 5), -179.99481)
        self.assertAlmostEqual(round(float(np.max(x1)), 5), 179.99481)
        self.assertAlmostEqual(round(float(np.min(x9)), 5), -179.95332)
        self.assertAlmostEqual(round(float(np.max(x9)), 5), 179.95332)

    @unittest.skipIf(not ANCILLARY_DATA_FOUND, 'No path to ancillary data')
    def test_get_ease2_slice_etc(self):
        'Should produce EASE-Grid 2.0 slice indices or offsets'
        kwargs = {'subset_id': 'CONUS'}
        self.assertEqual(
            get_ease2_slice_idx(grid = 'M01', **kwargs),
            ((5350, 10922), (1699, 4289)))
        self.assertEqual(
            get_ease2_slice_idx(grid = 'M09', **kwargs),
            ((594, 1214), (189, 477)))
        self.assertEqual(
            get_ease2_slice_offsets(grid = 'M01', **kwargs), (5350, 1699))
        self.assertEqual(
            get_ease2_slice_offsets(grid = 'M09', **kwargs), (594, 189))

    @unittest.skipIf(not ANCILLARY_DATA_FOUND, 'No path to ancillary data')
    def test_get_pft_array(self):
        'Should produce the desired PFT array, subset or not'
        kwargs = {'subset_id': 'CONUS'}
        self.assertTrue(
            np.equal(get_pft_array(grid = 'M01').shape, (14616, 34704)).all())
        self.assertTrue(
            np.equal(get_pft_array(grid = 'M09').shape, (1624, 3856)).all())
        self.assertTrue(
            np.equal(
                get_pft_array(
                    grid = 'M01', **kwargs).shape, (2590, 5572)).all())
        self.assertTrue(
            np.equal(
                get_pft_array(
                    grid = 'M09', **kwargs).shape, (288, 620)).all())

    def test_get_slice_idx_by_bbox(self):
        'Should correctly determine slicing indices based on a bounding box'
        nw_bbox = [-20, 10, -10, 20]   # Northern and Western hemispheres
        sw_bbox = [-20, -20, -10, -10] # Southern and Western hemispheres
        ne_bbox = [10, 10, 20, 20]     # ...
        se_bbox = [10, -20, 20, -10]

        nw_x_idx, nw_y_idx = get_slice_idx_by_bbox(
            self.x_coords, self.y_coords, subset_bbox = nw_bbox)
        nwx0, nwx1 = nw_x_idx
        nwy0, nwy1 = nw_y_idx
        sw_x_idx, sw_y_idx = get_slice_idx_by_bbox(
            self.x_coords, self.y_coords, subset_bbox = sw_bbox)
        swx0, swx1 = sw_x_idx
        swy0, swy1 = sw_y_idx
        ne_x_idx, ne_y_idx = get_slice_idx_by_bbox(
            self.x_coords, self.y_coords, subset_bbox = ne_bbox)
        nex0, nex1 = ne_x_idx
        ney0, ney1 = ne_y_idx
        se_x_idx, se_y_idx = get_slice_idx_by_bbox(
            self.x_coords, self.y_coords, subset_bbox = se_bbox)
        sex0, sex1 = se_x_idx
        sey0, sey1 = se_y_idx

        self.assertTrue(np.all(np.equal(self.x_coords[nwx0:nwx1], np.arange(-19.5, -10, 1))))
        self.assertTrue(np.all(np.equal(self.x_coords[swx0:swx1], np.arange(-19.5, -10, 1))))
        self.assertTrue(np.all(np.equal(self.x_coords[nex0:nex1], np.arange(10.5, 20, 1))))
        self.assertTrue(np.all(np.equal(self.x_coords[sex0:sex1], np.arange(10.5, 20, 1))))
        self.assertTrue(np.all(np.equal(self.y_coords[nwy0:nwy1], np.arange(10.5, 20, 1))))
        self.assertTrue(np.all(np.equal(self.y_coords[swy0:swy1], np.arange(-19.5, -10, 1))))
        self.assertTrue(np.all(np.equal(self.y_coords[ney0:ney1], np.arange(10.5, 20, 1))))
        self.assertTrue(np.all(np.equal(self.y_coords[sey0:sey1], np.arange(-19.5, -10, 1))))

        # Now, in reverse!
        nw_x_idx, nw_y_idx = get_slice_idx_by_bbox(
            self.x_coords[::-1], self.y_coords[::-1], subset_bbox = nw_bbox)
        nwx0, nwx1 = nw_x_idx
        nwy0, nwy1 = nw_y_idx
        sw_x_idx, sw_y_idx = get_slice_idx_by_bbox(
            self.x_coords[::-1], self.y_coords[::-1], subset_bbox = sw_bbox)
        swx0, swx1 = sw_x_idx
        swy0, swy1 = sw_y_idx
        ne_x_idx, ne_y_idx = get_slice_idx_by_bbox(
            self.x_coords[::-1], self.y_coords[::-1], subset_bbox = ne_bbox)
        nex0, nex1 = ne_x_idx
        ney0, ney1 = ne_y_idx
        se_x_idx, se_y_idx = get_slice_idx_by_bbox(
            self.x_coords[::-1], self.y_coords[::-1], subset_bbox = se_bbox)
        sex0, sex1 = se_x_idx
        sey0, sey1 = se_y_idx

        self.assertTrue(np.all(np.equal(
            self.x_coords[::-1][nwx0:nwx1], np.arange(-10.5, -20, -1))))
        self.assertTrue(np.all(np.equal(
            self.x_coords[::-1][swx0:swx1], np.arange(-10.5, -20, -1))))
        self.assertTrue(np.all(np.equal(
            self.x_coords[::-1][nex0:nex1], np.arange(19.5, 10, -1))))
        self.assertTrue(np.all(np.equal(
            self.x_coords[::-1][sex0:sex1], np.arange(19.5, 10, -1))))
        self.assertTrue(np.all(np.equal(
            self.y_coords[::-1][nwy0:nwy1], np.arange(19.5, 10, -1))))
        self.assertTrue(np.all(np.equal(
            self.y_coords[::-1][swy0:swy1], np.arange(-10.5, -20, -1))))
        self.assertTrue(np.all(np.equal(
            self.y_coords[::-1][ney0:ney1], np.arange(19.5, 10, -1))))
        self.assertTrue(np.all(np.equal(
            self.y_coords[::-1][sey0:sey1], np.arange(-10.5, -20, -1))))

    def test_partition_etc(self):
        'Should partition an array of positive indices into n pieces'
        self.assertEqual(
        partition(np.arange(1e3), 1), [(0, 1e3 + 1)])
        self.assertEqual(
        partition(np.arange(1e3), 2), [(0, 500), (500, 1001)])
        self.assertEqual(
        partition(np.arange(1e3), 3), [(0, 333), (333, 666), (666, 1001)])
        parts = [(0, 333), (333, 666), (666, 1001)]
        for i, part in enumerate(partition_generator(1e3, 3)):
            with self.subTest(i = i):
                self.assertEqual(part, parts[i])

    @unittest.skipIf(not ANCILLARY_DATA_FOUND, 'No path to ancillary data')
    def test_subset(self):
        'Should subset an input HDF5 file on EASE-Grid 2.0 grid'
        arr, xoff, yoff = subset(
            self.__class__.hdf_9km, 'NEE/nee_mean', subset_id = 'CONUS')
        self.assertTrue(np.equal(arr.shape, (288, 620)).all())
        self.assertTrue(xoff, 594)
        self.assertTrue(yoff, 189)

        arr, xoff, yoff = subset(
            self.__class__.hdf_9km, 'NEE/nee_mean',
            subset_bbox = [-124.5, 24.4, -66.7, 50.0])
        self.assertTrue(np.equal(arr.shape, (288, 620)).all())
        self.assertTrue(xoff, 594)
        self.assertTrue(yoff, 189)

    def test_subset2(self):
        'Should subset an input HDF5 file on (approximate) EASE-Grid 2.0 grid'
        # NOTE: This test uses affine transforms to obtain EASE-Grid 2.0
        #   coordinates, rather than the ancillary dataset's coordinates;
        #   this allows the test to be run *without* access to ancillary data
        arr, xoff, yoff = subset(
            self.__class__.hdf_9km_sorta, 'NEE/nee_mean', subset_id = 'CONUS')
        self.assertTrue(np.equal(arr.shape, (288, 620)).all())
        self.assertTrue(xoff, 594)
        self.assertTrue(yoff, 189)

        arr, xoff, yoff = subset(
            self.__class__.hdf_9km_sorta, 'NEE/nee_mean',
            subset_bbox = [-124.5, 24.4, -66.7, 50.0])
        self.assertTrue(np.equal(arr.shape, (288, 620)).all())
        self.assertTrue(xoff, 594)
        self.assertTrue(yoff, 189)

        # Also test the passing of x_coords and y_coords explicitly
        x_coords, y_coords = self.__class__.approx_ease2_coords
        arr, xoff, yoff = subset(
            self.__class__.hdf_9km_sorta, 'NEE/nee_mean', subset_id = 'CONUS',
            x_coords = x_coords, y_coords = y_coords)
        self.assertTrue(np.equal(arr.shape, (288, 620)).all())
        self.assertTrue(xoff, 594)
        self.assertTrue(yoff, 189)


class Compositing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(9)

    def test_composite(self):
        'Should composite 2 or more arrays together, in parallel or serial'
        checksum = hashlib.sha256()
        arrays = [
            np.random.normal(255, 50, 100).astype(np.int16).reshape((10, 10))
            for i in range(0, 10)
        ]
        result = composite(*arrays, reducer = 'mean')
        checksum.update(result)
        self.assertEqual(checksum.digest(), b"\xc9M\x18.\x16\xf0%7W\xb5\xb3\x1aH\x0bn\x87z'xM\x9b\xcf\xb5\x9a\xc0\xaf\xae\xc8\n\xeaf\x1a")

        checksum = hashlib.sha256()
        result = composite(*arrays, reducer = 'mean', processes = 2)
        checksum.update(result)
        self.assertEqual(checksum.digest(), b"\xc9M\x18.\x16\xf0%7W\xb5\xb3\x1aH\x0bn\x87z'xM\x9b\xcf\xb5\x9a\xc0\xaf\xae\xc8\n\xeaf\x1a")


class Summarization(unittest.TestCase):
    def test_summarize(self):
        'Should correctly summarize the values in a numeric array'
        a = np.arange(0, 25).reshape((5, 5))
        b = a.copy() # Copy and place NoData into the array
        b.ravel()[np.in1d(b, b[1,])] = -9999
        c = a.copy()
        c.ravel()[np.in1d(c, c[1,])] = 1234.0
        self.assertEqual(summarize(a, {
            'mean': lambda x: x.sum() / x.size
        })['mean'], a.mean())
        self.assertEqual(summarize(a, {'mean': np.mean})['mean'], a.mean())
        self.assertEqual(
            summarize(a, {'mean': np.mean}, scale = 3)['mean'],
            np.multiply(a, 3).mean())

        # NoData
        self.assertEqual(
            summarize(b, {'mean': np.nanmean})['mean'], 13.25)
        self.assertEqual(
            summarize(
                c, {'mean': np.nanmean}, nodata = 1234.0)['mean'], 13.25)

        # Multiple summaries
        self.assertEqual(len(summarize(a, {
            'median': np.median,
            'mean': np.mean,
            'std': np.std
        })), 3)

    def test_summarize_by_class(self):
        'Should correctl summarize'
        a = np.arange(0, 25).reshape((5, 5))
        labels = np.ones((5, 5))
        self.assertEqual(
            summarize_by_class(
                a, labels, {'mean': np.nanmean})[1]['mean'], a.mean())

        labels2 = labels.copy() # Copy and place 0.0 into the array
        labels2.ravel()[np.in1d(a, a[1,])] = 0
        self.assertEqual(
            summarize_by_class(
                a, labels2, {'mean': np.nanmean})[1]['mean'], 13.25)

        self.assertEqual(
            len(summarize_by_class(
                a, labels2, {'mean': np.nanmean}).keys()), 1)
        self.assertEqual(
            len(summarize_by_class(
                a, labels2, {'mean': np.nanmean}, ignore = None).keys()), 2)



if __name__ == '__main__':
    unittest.main()
