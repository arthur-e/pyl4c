'''
Test cases related to the pyl4c.apps.calibration module.
'''

import os
import unittest
import numpy as np
import pyl4c
from pyl4c.apps.calibration import BPLUT, cbar
from pyl4c.data.fixtures import restore_bplut, BPLUT as V4_BPLUT

BPLUT_PATH = os.path.join(os.path.dirname(pyl4c.__file__), 'data/files/SMAP_BPLUT_2020-07-31.csv')
BPLUT_CSV_FOUND = os.path.exists(BPLUT_PATH)
TEMP_HDF5_PATH = os.path.join(os.path.dirname(pyl4c.__file__), 'temp.h5')

class CalibrationTestSuite(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEMP_HDF5_PATH):
            os.remove(TEMP_HDF5_PATH)

    @unittest.skipIf(not BPLUT_CSV_FOUND, 'No path to recent BPLUT file')
    def test_bplut_with_network_access(self):
        'Should create in-memory representation of BPLUT with BPLUT() class'
        bp0 = BPLUT(restore_bplut(BPLUT_PATH))
        bp1 = BPLUT(bp0.data, hdf5_path = TEMP_HDF5_PATH)
        bp2 = BPLUT(hdf5_path = TEMP_HDF5_PATH)
        self.assertEqual(bp0.labels, bp1.labels)
        self.assertEqual(bp0.labels, bp2.labels)
        bp0_vals = bp0['LUE'].round(2)
        bp1_vals = bp1['LUE'].round(2)
        bp2_vals = bp2['LUE'].round(2)
        self.assertTrue(np.equal(
            bp0_vals[~np.isnan(bp0_vals)].round(2),
            bp1_vals[~np.isnan(bp1_vals)].round(2)).all())
        self.assertTrue(np.equal(
            bp1_vals[~np.isnan(bp1_vals)].round(2),
            bp2_vals[~np.isnan(bp2_vals)].round(2)).all())

    def test_bplut(self):
        'Should create in-memory representation of BPLUT with BPLUT() class'
        bp = BPLUT(V4_BPLUT)
        self.assertEqual(len(bp.labels), 18)
        self.assertTrue(np.equal(bp['LUE'][:,1], 1.71))

    def test_bplut_flat(self):
        'Should retrieve parameters in order for a given PFT'
        bp = BPLUT(V4_BPLUT)
        self.assertEqual(bp.flat(1).size, 18)
        self.assertEqual(bp.flat(1)[0], 1.71)

    def test_bplut_update(self):
        'Should update the correct entries in the BPLUT'
        # Test that entries are equal or are NaN (which is not equal to NaN)
        equal_or_nan = lambda x, y: np.equal(x[~np.isnan(x)], y[~np.isnan(y)])
        bp = BPLUT(V4_BPLUT)
        bp.update(0, (-9999, -9999), ('LUE', 'smsf0'), flush = False)
        bp.update(7, (-9999, -9999), ('CUE', 'smrz1'), flush = False)
        self.assertTrue(np.equal(
            bp.data['LUE'][0,np.array((0,7))],
            np.array((-9999, 2.53))).all())
        self.assertTrue(equal_or_nan(
            bp.data['smsf'][:,np.array((0,7))],
            np.array([[-9999, -42], [np.nan, 41]], dtype = np.float32)).all())
        self.assertTrue(equal_or_nan(
            bp.data['smrz'][:,np.array((0,7))],
            np.array([[np.nan, -15], [np.nan, -9999]], dtype = np.float32)).all())

    def test_cbar(self):
        'Should calculate cbar correctly'
        np.random.seed(9)
        rh = np.random.sample(10).reshape((10, 1))
        kmult = np.arange(0, 1, 0.1).reshape((10, 1))
        self.assertEqual(
            cbar(rh, kmult, q_rh = 75, q_k = 50).round(3)[0], 0.437)
        self.assertEqual(
            cbar(rh, kmult, q_rh = 85, q_k = 50).round(3)[0], 0.541)
        self.assertEqual(
            cbar(rh, kmult, q_rh = 85, q_k = 10).round(3)[0], 2.123)


if __name__ == '__main__':
    unittest.main()
