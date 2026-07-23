'''
If a parameter sensitivity analysis is requested, `analysis="parameters"`,
then the "model output" that is analyzed is the Nash-Sutcliffe efficiency
of the current model (with the parameter sweep), based on the observed
tower data.

If a sensitivity analysis of the *driver data* is requested,
`analysis="drivers"`, instead, then the model output is the predicted value
using an average value for the parameters.
'''

import json
import os
import yaml
import warnings
import numpy as np
import h5py
from collections import OrderedDict
from tqdm import tqdm
from mod16fet import MOD16_FET, PFT_VALID
from mod16fet.utils import restore_bplut
from mod16fet.calibration import DRIVER_NAMES, CalibrationAPI
from mod17.science import nash_sutcliffe
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol

OUTPUT_TPL = '/home/arthur.endsley/Workspace/NTSG/projects/Y2026_CONUS_30m_ET/data/MOD16-fET_sensitivity_%s_analysis.json'
CONFIG_FILE = '/home/arthur.endsley/Workspace/NTSG/projects/Y2026_CONUS_30m_ET/workflows/202606_calibration_workflow/20260705_MOD16-fET_calibration_config.yaml'
BPLUT = '/Stout1/MODIS_VIIRS/CONUS_30m_ET/calibration/MOD16-fET_BPLUT_20260704.csv'
# with open(os.path.join(MOD16_DIR, 'data/MOD16-fET_calibration_config.yaml'), 'r') as file:
with open(CONFIG_FILE, 'r') as file:
    CONFIG = yaml.safe_load(file)
BOUNDS = OrderedDict({ # Based on [2, 98] percentiles of Cal-Val data
    'lw_net': [-145.8,  -10.9],
    'sw_rad': [ 28.5, 357.5],
    'sw_albedo': [0.1, 0.5],
    'tmean': [259.8, 302. ],
    'tmin': [253.9, 296. ],
    'tmax': [265.1, 309.9],
    'vpd': [  31.4, 2951.9],
    'rhumidity': [0.,  0.9],
    'pressure': [ 68546.7, 101353. ],
    'fpar': [0.1, 0.9],
    'lai': [0.,  4.3],
})
PARAM_BOUNDS = {
    'tmin_close': (-8.0, -6.0),
    'tmin_open': (3.99, 22.04),
    'vpd_open': (650.0, 1000.0),
    'vpd_close': (4100.0, 8000.0),
    'gl_sh': (0.01429, 0.08638),
    'gl_wv': (0.01429, 0.08638),
    'g_cuticular': (3e-05, 0.00015),
    'csl': (0.00747, 0.0291),
    'rbl_min': (100.0, 250.0),
    'rbl_max': (970.0, 1000.0),
    'beta': (340.0, 1000.0),
    'fpar_scale': (0.5, 1.0),
}


def main(pft = None, analysis = 'parameters'):
    assert analysis == 'parameters' or pft is None,\
        'Cannot do a PFT-level analysis of the sensitivity when --analysis="drivers"'
    # Stratify the data using the validation mask so that an equal number of
    #   samples from each PFT are used
    api = CalibrationAPI(config = CONFIG_FILE)
    drivers_list = []
    obs_list = []
    if pft is None:
        for p in PFT_VALID:
            obs, drivers, _ = api._load_data(pft = p, verbose = False)
            obs_list.append(obs)
            drivers_list.append(drivers)

    drivers = dict()
    for key in DRIVER_NAMES:
        drivers[key] = []
        for drivers_set in drivers_list:
            drivers[key].append(drivers_set[key])
    tower_obs = np.concatenate(obs_list)

    drivers_vector = []
    for key in DRIVER_NAMES:
        drivers_vector.append(np.concatenate(drivers[key]))

    # Generate a vectorized set of (default) parameters
    if analysis == 'drivers':
        bplut = restore_bplut(BPLUT)
        params_vector = []
        # NOTE: Calculate the average of the parameters
        for key in MOD16_FET.required_parameters:
            params_vector.append(np.nanmean(bplut[key]))

    # For a sensitivity analysis of the parameters
    if analysis == 'parameters':
        filename = OUTPUT_TPL % 'ET'
        if pft is not None:
            filename = OUTPUT_TPL % f'ET-PFT{pft}'
        params = MOD16_FET.required_parameters
        problem = {
            'num_vars': len(params),
            'names': params,
            'bounds': [
                PARAM_BOUNDS[p] for p in params
            ]
        }
        # NOTE: Number of samples must be a power of 2
        param_sweep = sobol_sample(problem, 512 if pft is None else 128)
        Y = np.zeros([param_sweep.shape[0]])
        for i, X in enumerate(tqdm(param_sweep)):
            yhat = MOD16_FET._et(X, *drivers_vector)
            Y[i] = nash_sutcliffe(yhat, tower_obs, norm = True)

    elif analysis == 'drivers':
        filename = OUTPUT_TPL % 'ET-drivers'
        problem = {
            'num_vars': len(DRIVER_NAMES),
            'names': DRIVER_NAMES,
            'bounds': list(BOUNDS.values())
        }
        # NOTE: Number of samples must be a power of 2
        param_sweep = sobol_sample(problem, 2048)
        Y = np.zeros([param_sweep.shape[0]])
        # Exclude warnings, because some driver data combinations will lead
        #   to physically implausible situations
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i, X in enumerate(tqdm(param_sweep)):
                Y[i] = MOD16_FET._et(params_vector, *X)

    metrics = sobol.analyze(problem, Y)
    with open(filename, 'w') as file:
        json.dump(dict([(k, v.tolist()) for k, v in metrics.items()]), file)


def load_data(pft, validation_mask_only = False):
    print('Loading driver datasets...')
    lookup = CONFIG['data']['datasets']
    with h5py.File(CONFIG['data']['file'], 'r') as hdf:
        nsteps = hdf['time'].shape[0]
        if pft is not None:
            site_list = hdf['FLUXNET/site_id'][:].tolist()
            if hasattr(site_list[0], 'decode'):
                site_list = [s.decode('utf-8') for s in site_list]
            sites = pft_dominant(hdf['state/PFT'][:], site_list = site_list)
            sites = sites == pft
        else:
            shp = hdf[lookup['Tmin']].shape
            sites = np.ones(shp[1]).astype(bool)
        lw_net_day = hdf[lookup['LWGNT'][0]][:][:,sites]
        lw_net_night = hdf[lookup['LWGNT'][1]][:][:,sites]
        sw_albedo = np.nanmean(
            hdf[lookup['albedo']][:][:,sites], axis = -1)
        sw_rad_day = hdf[lookup['SWGDN'][0]][:][:,sites]
        sw_rad_night = np.zeros(sw_rad_day.shape)
        temp_day = hdf[lookup['T10M'][0]][:][:,sites]
        temp_night = hdf[lookup['T10M'][1]][:][:,sites]
        tmin = hdf[lookup['Tmin']][:][:,sites]
        temp_annual = hdf[lookup['MAT']][:][:,sites]
        if 'VPD' in lookup.keys():
            vpd_day = hdf[lookup['VPD'][0]][:][:,sites]
            vpd_night = hdf[lookup['VPD'][1]][:][:,sites]
        else:
            vpd_day = MOD16.vpd(
                hdf[lookup['QV10M_daytime']][:][:,sites],
                hdf[lookup['PS_daytime']][:][:,sites],
                temp_day)
            vpd_night = MOD16.vpd(
                hdf[lookup['QV10M_nighttime']][:][:,sites],
                hdf[lookup['PS_nighttime']][:][:,sites],
                temp_night)
        # After VPD is calculated, air pressure is based solely
        #   on elevation
        elevation = hdf[lookup['elevation']][:]
        elevation = elevation[np.newaxis,:]\
            .repeat(nsteps, axis = 0)[:,sites]
        pressure = MOD16.air_pressure(elevation.mean(axis = -1))
        # Read in fPAR, LAI, and convert from (%) to [0,1]
        fpar = np.nanmean(hdf[lookup['fPAR']][:][:,sites], axis = -1)
        lai = np.nanmean(hdf[lookup['LAI']][:][:,sites], axis = -1)
        # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR and LAI
        fpar /= 100
        lai /= 10
        tower_obs = hdf['FLUXNET/latent_heat'][:][:,sites]
        if pft is None:
            is_test = hdf['FLUXNET/validation_mask'][:].sum(axis = 0).astype(bool)
    # Compile driver datasets
    drivers = [
        lw_net_day, lw_net_night, sw_rad_day, sw_rad_night, sw_albedo,
        temp_day, temp_night, temp_annual, tmin, vpd_day, vpd_night,
        pressure, fpar, lai
    ]
    # Speed things up by focusing only on data points where valid data exist
    mask = ~np.isnan(tower_obs)
    if pft is None and validation_mask_only:
        # Stratify the data using the validation mask so that an equal number
        #   of samples from each PFT are used
        mask = np.logical_and(is_test, mask)
    drivers = [d[mask] for d in drivers]
    return (drivers, tower_obs[mask])


if __name__ == '__main__':
    import fire
    fire.Fire(main)
