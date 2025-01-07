pyl4c
========================

[![DOI](https://zenodo.org/badge/392401528.svg)](https://zenodo.org/badge/latestdoi/392401528)

This is a collection of Python tools for managing, analyzing, and visualizing SMAP L4C data; running L4C Science; and working with related models in the Terrestrial Carbon Flux (TCF) framework. In particular:

- Working with data in EASE-Grid 2.0 projection (`ease2.py`;)
- Converting HDF5 geophysical variables to GeoTIFF format (`spatial.py`);
- Creating statistical summaries of SMAP L4C variables or other raster arrays (`utils.py`);
- Reproducing L4C operational model logic (`science.py`);
- Down-scaling 9-km SMAP fields to 1-km resolution (`/apps/resample.py`)
- Calibrating the L4C model (`apps/calibration`);
- Running the L4C model (`apps/l4c`);
- Aligning and summarizing SMAP L4C variables with TransCom regions (`lib/transcom.py`);

The entire project is contained in the `pyl4c` module. Once installed:

```py
import pyl4c
```

Documentation
-------------

[Read the online documentation here.](https://arthur-e.github.io/pyl4c/)


Setup and Installation
----------------------

Because this project is highly modular, it must be installed as a package in order to resolve module references/ paths.
Check out `setup.sh` for an example of setting up the virtual environment prior to installation with `pip`.
Installation with `pip`, inside a virtual environment (`virtualenv`), is the recommendation.
Below, we install the `pyl4c` library in "development mode," which enables you to edit the source code.

```sh
$ pip install -e .
```

**Some tasks require ancillary datasets; be sure to check out "Linking Ancillary Datasets," below.**

**Some extra features must be requested in order to have their dependencies installed.**

```sh
# To install support for calibration of L4C
pip install -e .[calibration]

# To install support for command line interfaces and the "scripts" folder
pip install -e .[cli]

# To install support for reading netCDF4 files
pip install -e .[netcdf]

# To install support for resampling L4C data by TransCom regions
pip install -e .[transcom]
```

This will also install the project's dependencies. **NOTE: Because the GDAL Python bindings can be difficult to install, I recommend installing them as binaries through your system's package manager.** For instance, on Ubuntu GNU/Linux:

```sh
sudo apt install python3-gdal
```

You may encounter an error installing `pyl4c` from `setup.py`, depending on the version of the GDAL library you have installed. See `setup.py` to check which version of GDAL that is expected. You can install a specific version of the GDAL Python bindings that is consistent with your system installation by:

```sh
pip install GDAL==$(gdal-config --version)
```

There can also be issues with installing GDAL in a virtual environment; see [this thread](https://gis.stackexchange.com/questions/153199/import-error-no-module-named-gdal-array) and also try:

```sh
pip install --no-build-isolation --no-cache-dir --force-reinstall gdal==$(gdal-config --version)
```

If there are "undefined symbol" issues, despite the above steps, try installing `numpy` from source, first, before re-installing GDAL as above:

```sh
# Requires gcc version 8.0.0+
pip install --no-binary=numpy numpy
```


### Linking Ancillary Datasets

You should create a file, `pyl4c/data/files/ancillary_data_paths.yaml`, using the following template:

```yaml
smap_l4c_ancillary_data_file_path: "SPL4C_Vv4040_SMAP_L4_C.Ancillary.h5"
smap_l4c_1km_ancillary_data_lc_path: "MCD12Q1_M01_lc_dom_uint8"
smap_l4c_9km_ancillary_data_lc_path: "MOD12Q1_M09_lc_dom_uint8"
smap_l4c_1km_ancillary_data_x_coord_path: "SMAP_L4_C_LON_14616_x_34704_M01_flt32"
smap_l4c_1km_ancillary_data_y_coord_path: "SMAP_L4_C_LAT_14616_x_34704_M01_flt32"
smap_l4c_9km_ancillary_data_x_coord_path: "SMAP_L4_C_LON_1624_x_3856_M09_flt32"
smap_l4c_9km_ancillary_data_y_coord_path: "SMAP_L4_C_LAT_1624_x_3856_M09_flt32"
smap_l4c_9km_pft_subgrid_counts_CONUS: "SMAP_L4C_Vv4040_1km_subgrid_PFT_counts_CONUS.h5"
smap_l4c_9km_sparse_col_index: "MCD12Q1_M09land_col.uint16"
smap_l4c_9km_sparse_row_index: "MCD12Q1_M09land_row.uint16"
transcom_netcdf_path: "CarbonTracker_TransCom_and_other_regions.nc"
```

Each of the filenames corresponds to an ancillary data file that is probably needed. You should update that value with an absolute file path to the corresponding file on your file system.


### Dependencies

This package requires system support for HDF5 and the Geospatial Data Abstraction Library (GDAL).

- Python 3.5+
- GDAL (2.4+)
- HDF5

Development headers for GDAL might also be necessary to get the Python bindings to install correctly. On Ubuntu GNU/Linux:

```sh
# Install support for HDF5 (and the Python 3 bindings)
sudo apt install libhdf5-103 libhdf5-dev python3-h5py

# Install support for GDAL Python bindings (and the Python 3 bindings)
sudo apt install gdal-bin libgdal-dev python3-gdal
```

**NOTE:** For using `calibration` tools, NetCDF (3 and 4) and `nlopt` are required which, in turn, may require additional system libraries. On Ubuntu GNU/Linux:

```sh
sudo apt install libnlopt0
```

**NOTE:** The basemap toolkit for `matplotlib` must be installed separately:

```sh
pip install git+https://github.com/matplotlib/basemap.git
```
