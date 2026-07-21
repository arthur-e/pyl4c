#!/bin/bash

# Author: K. Arthur Endsley
# Projects input raster onto an EASE-Grid 2.0. Creates a compressed GeoTIFF
#   for either 1-km and 9-km EASE-Grid 2.0 projections.
# Use:
#   $ sh project_onto_EASE-Grid2.sh <input_GeoTIFF_raster> <cell_size_meters> <nodata>

# NOTE: This is ideal if the global extent does match that of the EASE-Grid 2.0 projection,
#   but of course it won't work well at all if this is not the case.

base=`expr match ${1} '\(.*\)\.tif'`

echo "Using bilinear resampling..."

if [ ${2} -eq 9000 ]; then
  # See: https://nsidc.org/ease/ease-grid-projection-gt
  gdalwarp -r bilinear -t_srs "EPSG:6933" -tr 9000 9000 -te -17367530.450 -7301459.170 17336469.550 7314540.830\
    -srcnodata ${3} -dstnodata -9999 -multi -wo NUM_THREADS=6 -ot Int16 ${1} tmp9km.tif
  gdal_translate -co COMPRESS=LZW tmp9km.tif "${base}_epsg6933_9km.tif"
  rm tmp9km.tif
fi

if [ ${2} -eq 1000 ]; then
  gdalwarp -r bilinear -t_srs "EPSG:6933" -tr 1000 1000 -te -17367530.450 -7301459.170 17336469.550 7314540.830\
    -srcnodata ${3} -dstnodata -9999 -multi -wo NUM_THREADS=6 -ot Int16 ${1} tmp1km.tif
  gdal_translate -co COMPRESS=LZW tmp1km.tif "${base}_epsg6933_1km.tif"
  rm tmp1km.tif
fi
