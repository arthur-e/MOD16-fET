'''
Unit tests for the `mod16fet` Python utilities library.
'''

import os
import unittest
import numpy as np
from mod16fet import MOD16, PFT_VALID

# MOD17_BPLUT = os.path.join(
#     os.path.dirname(mod17.__file__), 'data/MOD17_BPLUT_C5.1_MERRA_NASA.csv')

class ETComponents(unittest.TestCase):
    '''
    Suite of test cases related to components of ET.
    '''

    @classmethod
    def setUp(cls):
        cls.params = dict().fromkeys(MOD16.required_parameters, None)
        cls.params.update({
            'gl_sh': 0.01,
            'gl_wv': 0.01,
            'g_cuticular': 1e-5,
            'tmin_close': -8, # deg C
            'tmin_open': 8, # deg C
            'vpd_open': 650,
            'vpd_close': 3000,
            'rbl_min': 60,
            'rbl_max': 90,
            'csl': 2.4e-3,
            'beta': 250 # Original (hard-coded constant) value
        })
        cls.params_list = [cls.params[k] for k in MOD16.required_parameters]
        cls.pressure = 100e3
        cls.tmean = 273.15 + 25
        cls.tmin = 273.15 + 20
        cls.tmax = 273.15 + 30
        cls.vpd = 1000
        cls.lai = 1.5
        cls.fpar = 0.5
        cls.f_wet = 0.5
        cls.rhumidity = 0.3
        cls.r_corr = (101300 / cls.pressure) * (cls.tmean / 293.15)**1.75
        cls.lw_net = -10
        cls.lw_net_day = -20
        cls.lw_net_night = -10
        cls.sw_rad = 150
        cls.sw_rad_day = 175
        cls.sw_albedo = 0.3
        cls.mat = 285
        # 5 levels of each
        cls._pressure = np.arange(98e3, 103e3, 1e3)
        cls._tmean = 273.15 + np.array([0, 10, 20, 30, 40])
        cls._vpd = np.arange(0, 5000, 1000)
        cls._lai = np.arange(0.5, 3, 0.5)
        cls._fpar = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        cls._f_wet = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def test_et_cal(self):
        np.random.seed(42)
        params_list = [self.params_list] * len(PFT_VALID)
        # Generate some PFT fractions
        pft_frac = np.random.rand(len(PFT_VALID))
        pft_frac /= pft_frac.sum()
        result = MOD16._evapotranspiration(
            params_list, self.lw_net, self.lw_net_day, self.lw_net_night,
            self.sw_rad, self.sw_rad_day, self.sw_albedo, self.tmean,
            self.tmin, self.tmax, self.mat, self.vpd, self.rhumidity,
            self.pressure, self.fpar, self.lai, self.f_wet)
        self.assertTrue(result.shape[0] == len(PFT_VALID))


if __name__ == '__main__':
    unittest.main()
