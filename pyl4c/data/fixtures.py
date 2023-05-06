'''
Commonly used global constants. **New users should change
`ANCILLARY_DATA_PATHS` if it doesn't match their file system.**

Source for EASE-Grid 2.0 parameters:

    https://nsidc.org/ease/ease-grid-projection-gt

NOTE: EASE-Grid 2.0 parameters do not match exactly those specified at NSIDC
because, in fact, they must be 1000 or 9000, depending on the grid size, in
order to get the right number of rows and columns in the output image.
Grid resolution is only a whole number for the polar (north or south
hemisphere) grids.
'''

import csv
import numpy as np
from collections import OrderedDict

ANCILLARY_DATA_PATHS = {
    'smap_l4c_ancillary_data_file_path': 'SPL4C_Vv4040_SMAP_L4_C.Ancillary.h5',
    'smap_l4c_1km_ancillary_data_lc_path': 'MCD12Q1_M01_lc_dom_uint8',
    'smap_l4c_9km_ancillary_data_lc_path': 'MOD12Q1_M09_lc_dom_uint8',
    'smap_l4c_1km_ancillary_data_x_coord_path': 'SMAP_L4_C_LON_14616_x_34704_M01_flt32',
    'smap_l4c_1km_ancillary_data_y_coord_path': 'SMAP_L4_C_LAT_14616_x_34704_M01_flt32',
    'smap_l4c_9km_ancillary_data_x_coord_path': 'SMAP_L4_C_LON_1624_x_3856_M09_flt32',
    'smap_l4c_9km_ancillary_data_y_coord_path': 'SMAP_L4_C_LAT_1624_x_3856_M09_flt32',
    'smap_l4c_9km_pft_subgrid_counts_CONUS': 'SMAP_L4C_Vv4040_1km_subgrid_PFT_counts_CONUS.h5',
    'smap_l4c_9km_sparse_col_index': 'MCD12Q1_M09land_col.uint16',
    'smap_l4c_9km_sparse_row_index': 'MCD12Q1_M09land_row.uint16',
    'transcom_netcdf_path': 'CarbonTracker_TransCom_and_other_regions.nc'
}


# Version 4020 BPLUT parameters in order of PFT numeric code (PFT 0 through 9)
BPLUT = OrderedDict({
    # Only reason for OrderedDict (instead of dict) is for consistency with
    #   pyl4c.data.fixtures.BPLUT()
    '_version': '4',
    'LUE': np.array([[ # gC per MJ
        np.nan, 1.71, 1.38, 1.71, 1.19, 1.95, 1.68, 2.53, 3.6, np.nan
    ]]),
    'CUE': np.array([[
        np.nan, 0.7, 0.57, 0.79, 0.78, 0.65, 0.6, 0.72, 0.71, np.nan
    ]]),
    'tmin': np.array([ # degrees K
        [np.nan, 235, 230, 259, 260, 259, 251, 246, 266, np.nan],
        [np.nan, 309, 303, 301, 284, 304, 302, 314, 319, np.nan]
    ]),
    'vpd': np.array([ # Pascals
        [np.nan, 0, 15, 869, 1500, 4, 0, 228, 1500, np.nan],
        [np.nan, 4169, 7000, 3452, 5401, 4282, 4229, 4516, 7000, np.nan]
    ]),
    'smrz': np.array([ # Percent saturation
        [np.nan, -30, 18, -30, -26, 22, -30, -15, 10, np.nan],
        [np.nan, 74, 26, 49, 87, 72, 76, 30, 68, np.nan]
    ]),
    'smsf': np.array([ # Percent saturation
        [np.nan, -50, -46, -9, -50, -15, -3, -42, -49, np.nan],
        [np.nan, 46, 59, 43, 53, 48, 51, 41, 30, np.nan]
    ]),
    'ft': np.array([ # Frozen = 0, Thawed = 1
        [np.nan, 0.58, 0.36, 0.55, 0.9, 0.91, 0.85, 0.77, 1, np.nan],
        [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, np.nan]
    ]),
    'tsoil': np.array([[ # degrees K
        np.nan, 265.04, 477.83, 238.81, 267.26, 292.82, 232.24, 263.97, 329.63, np.nan
    ]]),
    # NOTE: The medium/structural and slow/recalcitrant decay constants
    #   (2nd and 3rd rows) are true decay constants, unlike in Ops
    #   BPLUT which presents only dimensionless scalars.
    'decay_rates': np.array([ # days^-1
        [np.nan, 0.027, 0.028, 0.028, 0.03, 0.015, 0.025, 0.019, 0.035, np.nan],
        [np.nan, 1.08e-2, 1.12e-2, 1.12e-2, 1.2e-2, 0.6e-2, 1e-2, 0.76e-2, 1.4e-2, np.nan],
        [np.nan, 2.51e-4, 2.6e-4, 2.6e-4, 2.79e-4, 1.4e-4, 2.33e-4, 1.77e-4, 3.26e-4, np.nan]
    ]),
    'f_metabolic': np.array([[ # Fraction of daily litterfall entering "fast" or metabolic pool
        np.nan, 0.49, 0.71, 0.67, 0.67, 0.62, 0.76, 0.78, 0.78, np.nan
    ]]), # Also known as "f_met"
    'f_structural': np.array([[ # Fraction of structural (str) pool transferred in "humification"
        np.nan, 0.3, 0.3, 0.7, 0.3, 0.35, 0.55, 0.5, 0.8, np.nan
    ]]) # Also known as "f_str" -- See Jones et al. (2017, IEEE TGARS, p.5)
})


EASE2_GRID_PARAMS = {
    # A GeoTransform for a north-up raster is:
    #   (x_min, pixel_width, 0, y_max, 0, -pixel_height)
    'M01': {
        'epsg': 6933,
        'geotransform': (-17367530.45, 1000, 0, 7314540.83, 0, -1000),
        'resolution': 1000.89502334956, # From Brodzik et al.
        'shape': (14616, 34704),
        'size': 14616*34704
    },
    'M09': {
        'epsg': 6933,
        'geotransform': (-17367530.45, 9000, 0, 7314540.83, 0, -9000),
        'resolution': 9008.055210146, # From Brodzik et al.
        'shape': (1624, 3856),
        'size': 1624*3856
    },
    'N09': {
        'epsg': 6931,
        'geotransform': (-9000000.0, 9000, 0, 9000000.0, 0, -9000),
        'resolution': 9000,
        'shape': (2000, 2000),
        'size': 2000*2000
    },
    'M25': {
        'epsg': 6933,
        'geotransform': (-17367530.45, 25000, 0, 7307375.92, 0, -25000),
        'resolution': 25000,
        'shape': (584, 1388),
        'size': 584*1388
    },
    'M36': {
        'epsg': 6933,
        'geotransform': (-17367530.45, 36000, 0, 7314540.83, 0, -36000),
        'resolution': 36032.22,
        'shape': (406, 964),
        'size': 406*964
    }
}


HDF_PATHS = { # Where in the HDF hierarchy certain variables live
    'SPL4CMDL': { # By Earthdata Dataset ID
        '4': { # By Version number
            'longitude': 'GEO/longitude',
            'latitude': 'GEO/latitude',
            'SOC': 'SOC/soc_mean',
            'SOC*': 'SOC/soc_pft%d_mean',
            'GPP': 'GPP/gpp_mean',
            'GPP*': 'GPP/gpp_pft%d_mean',
            'NEE': 'NEE/nee_mean',
            'NEE*': 'NEE/nee_pft%d_mean',
            'RH': 'RH/rh_mean',
            'RH*': 'RH/rh_pft%d_mean',
        }
    },
    'SPL4SMGP': {
        '4': {'longitude': 'cell_lon', 'latitude': 'cell_lat',}
    }
}


PFT = OrderedDict({
    0: ('Water', 'WET'),
    1: ('Evergreen Needleleaf', 'ENF'),
    2: ('Evergreen Broadleaf', 'EBF'),
    3: ('Deciduous Needleleaf', 'DNF'),
    4: ('Deciduous Broadleaf', 'DBF'),
    5: ('Shrub', 'SHB'),
    6: ('Grass', 'GRS'),
    7: ('Cereal Crop', 'CCR'),
    8: ('Broadleaf Crop', 'BCR'),
    9: ('Urban and Built-Up', 'URB')
})


# BBOX extents, in decimal degrees, based on gdal_rasterize -te option:
#   https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal-translate-te
SUBSETS_BBOX = { # <xmin>, <ymin>, <xmax>, <ymax>
    'CONUS': [-124.5, 24.4, -66.7, 50.0],
    'WesternHemisphere': [-180, -90, 0, 90],
    'WesternHemisphere2': [-180, -90, -25, 90],
    'NorthernHemisphere': [-180, 0, 180, 90],
    'NorthernHemisphere45': [-180, 45, 180, 90],
    'NorthernHemisphere40': [-180, 40, 180, 90],
    'Nigeria': [2.3, 4, 14.8, 14],
    # AOI for: Chiodi and Harrison (2013) Journal of Climate 26 (3):822–837
    'ChiodiHarrison2013': [-160, -5, -110, 5],
    'Kang2014EMS': [-125, 47, -113, 49],
    'NorthernPlains': [-116.1, 40.9, -96.4, 49],
    'Montana': [-116.1, 44.4, -104.01, 49],
    'Iowa': [-96.7, 40.58, -90.1, 43.5],
    'Bangladesh1kmSubset': [78.9, 23.2, 96.1, 29.3]
}


def parameter_mapped(name, pft_array, bplut = BPLUT):
    '''
    Given a BPLUT parameter and a PFT array, returns an array with the
    corresponding parameter values for each PFT code.

    Parameters
    ----------
    name : str
        The name of the BPLUT parameter
    pft_array : numpy.ndarray
        Array of any size or shape with numeric elements corresponding to
        PFT codes
    bplut : dict
        (Optional) A collection of BPLUT parameters

    Returns
    -------
    numpy.ndarray
        Array of same size, shape as pft_array but with parameter values
        in place of PFT codes
    '''
    param = bplut[name]
    # Basically, index the <name> array, in PFT order, by PFT numeric codes
    return np.asarray(param)[:,np.ravel(pft_array)].reshape(pft_array.shape)


def restore_bplut(csv_file_path, version_id = None):
    '''
    Translates a BPLUT CSV file to a Python internal representation
    (OrderedDict instance).

    Parameters
    ----------
    csv_file_path : str
        File path to the CSV representation of the BPLUT
    version_id : str
        (Optional) Version identifier for the BPLUT

    Returns
    -------
    OrderedDict
    '''
    header = ('LC_index', 'LC_Label', 'model_code', 'NDVItoFPAR_scale',
        'NDVItoFPAR_offset', 'LUEmax', 'Tmin_min_K', 'Tmin_max_K',
        'VPD_min_Pa', 'VPD_max_Pa', 'SMrz_min', 'SMrz_max', 'FT_min',
        'FT_max', 'SMtop_min', 'SMtop_max', 'Tsoil_beta0', 'Tsoil_beta1',
        'Tsoil_beta2', 'fraut', 'fmet', 'fstr', 'kopt', 'kstr', 'kslw',
        'Nee_QA_Rank_min', 'Nee_QA_Rank_max', 'Nee_QA_Error_min',
        'Nee_QA_Error_max', 'Fpar_QA_Rank_min', 'Fpar_QA_Rank_max',
        'Fpar_QA_Error_min', 'Fpar_QA_Error_max', 'FtMethod_QA_mult',
        'FtAge_QA_Rank_min', 'FtAge_QA_Rank_max', 'FtAge_QA_Error_min',
        'FtAge_QA_Error_max', 'Par_QA_Error', 'Tmin_QA_Error',
        'Vpd_QA_Error', 'Smrz_QA_Error', 'Tsoil_QA_Error', 'Smtop_QA_Error')
    contents = []
    with open(csv_file_path, 'r') as stream:
        reader = csv.DictReader(
            filter(lambda row: row[0] != '#', stream), fieldnames = header)
        for row in reader:
            contents.append(row)

    result = BPLUT.copy()
    result['_version'] = 'UNKNOWN' if version_id is None else version_id
    result['LUE'] = np.array([[
        contents[p-1]['LUEmax'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(4)
    result['CUE'] = np.array([[
        (1 - float(contents[p-1]['fraut'])) if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(4)
    result['tmin'] = np.array([
        [contents[p-1]['Tmin_min_K'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['Tmin_max_K'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(1)
    result['vpd'] = np.array([
        [contents[p-1]['VPD_min_Pa'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['VPD_max_Pa'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(2)
    result['smrz'] = np.array([
        [contents[p-1]['SMrz_min'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['SMrz_max'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(1)
    result['smsf'] = np.array([
        [contents[p-1]['SMtop_min'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['SMtop_max'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(1)
    result['ft'] = np.array([
        [contents[p-1]['FT_min'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['FT_max'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(3)
    result['tsoil'] = np.array([[
        contents[p-1]['Tsoil_beta0'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(1)
    result['f_metabolic'] = np.array([[
        contents[p-1]['fmet'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(3)
    result['f_structural'] = np.array([[
        contents[p-1]['fstr'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(1)
    result['decay_rates'] = np.array([
        [contents[p-1]['kopt'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['kstr'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['kslw'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(3)
    # The "kstr" and "kslw" values are really the fraction of kopt
    #   assigned to the second and third pools
    result['decay_rates'][1,:] = np.multiply(
        result['decay_rates'][0,:], result['decay_rates'][1,:])
    result['decay_rates'][2,:] = np.multiply(
        result['decay_rates'][0,:], result['decay_rates'][2,:])
    return result


def restore_bplut_flat(csv_file_path, version_id = None):
    '''
    Translates a BPLUT CSV file to a Python internal representation
    (OrderedDict instance). Compare to `restore_bplut()`, this version
    sets parameter values as flat lists instead of n-dimensional NumPy
    arrays.

    Parameters
    ----------
    csv_file_path : str
        File path to the CSV representation of the BPLUT
    version_id : str
        (Optional) Version identifier for the BPLUT

    Returns
    -------
    OrderedDict
    '''
    params = restore_bplut(csv_file_path, version_id)
    result = dict()
    for key, value in params.items():
        if key not in ('tmin', 'vpd', 'smsf', 'smrz', 'ft'):
            result[key] = value
            continue
        for i, array in enumerate(value.tolist()):
            result[f'{key}{i}'] = np.array(array).reshape((1,len(array)))
    return OrderedDict(result)
