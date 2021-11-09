'''
Test cases related to the pyl4c.apps.l4c module.
'''

import datetime
import os
import unittest
import numpy as np
import warnings
import pyl4c
from pyl4c.data.fixtures import BPLUT, restore_bplut
from pyl4c.apps.l4c.main import L4CForwardProcessPoint

INPUTS_FILE_PATH = '/anx_lagr3/arthur.endsley/SMAP_L4C/calibration/v5_Y2020/L4_C_tower_site_drivers_NRv8-3_for_356_sites.h5'
INPUTS_FILE_FOUND = os.path.exists(INPUTS_FILE_PATH)

class L4CForwardProcessPointTestSuite(unittest.TestCase):
    '''
    Integration tests for the L4CForwardProcessPoint model.
    '''
    @classmethod
    def setUpClass(cls):
        cls.default_config = {
            # NOTE: This is the wrong BPLUT version for the data used, but
            #   it doesn't matter because we're just running tests
            'bplut': BPLUT,
            'inputs_file_path': INPUTS_FILE_PATH,
            'site_count': 356,
            'time_steps': 100,
            'start': '2000-01-01',
            'end': (datetime.datetime(2000, 1, 1) + datetime.timedelta(days = 100)).strftime('%Y-%m-%d'),
        }

    @unittest.skipIf(not INPUTS_FILE_FOUND, 'No path to recent input drivers data file')
    def test_model_instantiation(self):
        'Model should be instantiated correctly in debug mode or not'
        config = self.__class__.default_config.copy()
        m0 = L4CForwardProcessPoint(config, verbose = False, debug = True)
        m1 = L4CForwardProcessPoint(config, verbose = False, debug = False)
        for model in (m0, m1):
            self.assertEqual(model.fluxes.shape, (3, 100, 356, 81))
            self.assertEqual(model.fluxes.labels, ('gpp', 'rh', 'nee'))
            self.assertEqual(model.state_initial.shape, (6, 1, 356, 81))
        # If debug = True, should have many more state variables to track
        self.assertEqual(m0.state.shape, (13, 100, 356, 81))
        self.assertEqual(m0.state.labels, (
            'soc1', 'soc2', 'soc3', 'e_mult', 't_mult', 'w_mult', 'f_tmin',
            'f_vpd', 'f_ft', 'f_smrz', 'f_tsoil', 'f_smsf', 'apar'
        ))
        # If debug = False...
        self.assertEqual(m1.state.shape, (6, 100, 356, 81))
        self.assertEqual(m1.state.labels, (
            'soc1', 'soc2', 'soc3', 'e_mult', 't_mult', 'w_mult'
        ))
        self.assertTrue(np.isnan(model.fluxes.data).all())
        self.assertTrue(np.isnan(model.state.data).all())
        self.assertTrue(not np.isnan(model.state_initial.data).all())

    @unittest.skipIf(not INPUTS_FILE_FOUND, 'No path to recent input drivers data file')
    def test_model_instantiation_with_longer_record(self):
        'Model should be instantiated correctly with a longer record'
        config = self.__class__.default_config.copy()
        config['time_steps'] = 150
        config['end'] = (datetime.datetime(2000, 1, 1) +\
            datetime.timedelta(days = 150)).strftime('%Y-%m-%d')
        m1 = L4CForwardProcessPoint(config, verbose = False, debug = False)
        self.assertEqual(m1.state.shape, (6, 150, 356, 81))
        self.assertEqual(m1.fluxes.shape, (3, 150, 356, 81))

    @unittest.skipIf(not INPUTS_FILE_FOUND, 'No path to recent input drivers data file')
    def test_model_constants_setup(self):
        'Model should contain arrays of the parameter values'
        config = self.__class__.default_config.copy()
        m1 = L4CForwardProcessPoint(config, verbose = False, debug = False)
        self.assertEqual(m1.constants.litterfall.shape, (356, 81))
        self.assertEqual(m1.constants.decay_rates.shape, (3, 356, 81))

    @unittest.skipIf(not INPUTS_FILE_FOUND, 'No path to recent input drivers data file')
    def test_model_forward_run(self):
        'Model should have predictable behavior after one forward run step'
        config = self.__class__.default_config.copy()
        m1 = L4CForwardProcessPoint(config, verbose = False, debug = False)
        m1.run(1)
        self.assertTrue(not np.isnan(m1.fluxes.data).all())
        self.assertTrue(np.equal(
            m1.fluxes.data[:,0,0,0].round(3),
            np.array([5.014, 0.939, -2.32]).astype(np.float32)
        ).all())
        self.assertTrue(np.equal(
            m1.fluxes.data[:,0,42,0].round(3),
            np.array([5.765, 2.339, -0.947]).astype(np.float32)
        ).all())
        self.assertTrue(np.equal(
            m1.state.data[:,0,0,0].round(2).astype(np.float64),
            np.array([44.09, 66.11, 2395.2, 0.44, 1, 0.75]).astype(np.float32)
        ).all())
        self.assertTrue(np.equal(
            m1.state.data[:,0,99,0].round(2).astype(np.float64),
            np.array([37.51, 30.29, 2628.11, 0.07, 0.04, 0.46]).astype(np.float32)
        ).all())


if __name__ == '__main__':
    unittest.main()
