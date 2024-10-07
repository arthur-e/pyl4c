'''
Given a MODIS granule, generates a GeoTIFF version; optionally, with the
QA/QC and cloud masks applied.
'''

import numpy as np
from typing import Sequence
from osgeo import gdal
from pyl4c.epsg import SR_ORG
from pyl4c.spatial import array_to_raster, dump_raster
from pyl4c.lib.modis import dec2bin_unpack, mod15a2h_qc_fail

def main(
        filename: str, output_path: str, valid_min: int = 0,
        valid_max: int = 100, nodata: int = 255,
        grid: str = 'MOD_Grid_MOD15A2H', field: str = 'Fpar_500m',
        field_qc: str = 'FparLai_QC', dtype: str = 'uint8',
        reclass: Sequence = None, apply_mask: bool = False):
    '''
    Extract a VIIRS/MODIS gridded dataset as a GeoTIFF. Dataset should have
    integer digital numbers (i.e., integer type). NOTE: Does not apply the
    appropriate scale factor so as to keep output file size small (by
    preserving integer data type).

    The "reclass" argument allows a QC band to reclassified into a discrete
    raster; e.g., "reclass=[7,]" would emit a binary raster with "1" at every
    pixel where the least-significant bit (bit 7) in the byte representation
    of the input decimal is "1" and "0" otherwise. Combinations of bit flags,
    e.g., "reclass=[5,7]" may result in addition, e.g., "1" + "2" = "3" where
    the byte representation has "1" in both the 5th and 7th bits.

    Parameters
    ----------
    filename : str
    output_path : str
    valid_min : int
    valid_max : int
    nodata : int
    field : str
    field_qc : str
    dtype : str
        Name of a NumPy data type, e.g., "float32" (numpy.float32)
        (Default: "uint8")
    reclass : tuple or list
        Sequence of bit positions 0 through 7, inclusive; in the binary
        unpacked representation of an 8-bit decimal number on [0,255], a "1"
        in each bit position will be flagged as 1, 2, 3, ... in the output
    apply_mask : bool
        True to open the QC layer and apply QC logic (Default: False)
    '''
    dtype = getattr(np, dtype)
    wkt = SR_ORG[6842]
    # Read in data array
    ds = gdal.Open(
        'HDF4_EOS:EOS_GRID:"%s":%s:%s' % (filename, grid, field))
    if ds is None:
        raise AssertionError(
            'Name error: Either field "%s" or grid "%s" not in MODIS product granule' % (field, grid))
    gt = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    ds = None
    if field == 'Fpar_500m':
        ds = gdal.Open('HDF4_EOS:EOS_GRID:"%s":%s:%s' % (filename, grid, field_qc))
        qc = mod15a2h_qc_fail(ds.ReadAsArray())
        ds = None
        # Mask values outside valid range; mask bad QC pixels
        masked = np.logical_or(
            np.logical_or(arr < valid_min, arr > valid_max), qc)
        arr = np.where(masked, nodata, arr)
    elif reclass is not None:
        n = len(reclass) + 2
        # Enumerate possible bytes
        choices = dec2bin_unpack(
            np.array(range(0, n), dtype = np.uint8))[:,n:].tolist()
        rc = dec2bin_unpack(arr)
        arr = np.zeros(arr.shape)
        (i, j) = (min(reclass), max(reclass)) # Get bit range of interest
        for b, choice in enumerate(choices):
            arr[np.equal(rc[...,i:(j+1)], choice).all(axis = 2)] += b
    dump_raster(
        array_to_raster(arr, gt, wkt, dtype = dtype, nodata = nodata),
        output_path)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
