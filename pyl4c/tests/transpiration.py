'''
Test cases related to the pyl4c.lib.transpiration module.
'''

import os
import unittest
import numpy as np
import pyl4c
from pyl4c.lib.transpiration import canopy_evaporation, psychrometric_constant, radiation_net, svp_slope

class TranspirationTestSuite(unittest.TestCase):
    _depths = -np.array((0.05, 0.1, 0.2, 0.4, 0.75, 1.5)).reshape((6,1))

    def test_psychrometric_constant(self):
        'Should accurately calculate the psychrometric constant'
        pressure = np.array((100e3, 80e3, 100e3, 80e3))
        temp_k = 273.15 + np.array((10, 10, 25, 25))
        answer = [65.74, 52.59, 66.69, 53.35]
        for i in range(0, 4):
            self.assertEqual(
                answer[i],
                psychrometric_constant(pressure[i], temp_k[i]).round(2))
        # Example from FAO:
        #   http://www.fao.org/3/X0490E/x0490e07.htm#psychrometric%20constant%20(g)
        self.assertEqual(
            54.55, np.round(psychrometric_constant(81.8e3, 25 + 273.15), 2))

    def test_radiation_net(self):
        'Should accurately calculate net radiation to the land surface'
        swrad = np.array((500, 5000, 500, 5000, 500, 5000, 500, 5000))
        albedo = np.array((0.4, 0.4, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8))
        temp_k = 273.15 + np.array((10, 10, 10, 10, 25, 25, 25, 25))
        answer = [223.3, 2923.3, 23.3, 923.3, 241.8, 2941.8, 41.8, 941.8]
        for i in range(0, 8):
            self.assertEqual(
                answer[i],
                radiation_net(swrad[i], albedo[i], temp_k[i]).round(1))

    def test_svp_slope(self):
        'Should accurately calculate slope of SVP curve'
        self.assertEqual( 82.3, svp_slope(273.15 + 10).round(1))
        self.assertEqual(144.7, svp_slope(273.15 + 20).round(1))
        self.assertEqual(188.7, svp_slope(273.15 + 25).round(1))

    def test_wet_canopy_evaporation(self):
        'Should accurately calculate wet canopy evaporation'
        evap = canopy_evaporation(
            100e3, 25 + 273.15, 0.5, 1000, 1, 0.5, 5000, 0.1).round(6)
        self.assertEqual(evap, 0.000151)


if __name__ == '__main__':
    unittest.main()
