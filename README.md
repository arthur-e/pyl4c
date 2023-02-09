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

**Some extra features must be requested in order to have their dependencies installed.**

```sh
# To install support for calibration of L4C
pip install -e pyl4c[calibration]

# To install support for command line interfaces and the "scripts" folder
pip install -e pyl4c[cli]

# To install support for reading netCDF4 files
pip install -e pyl4c[netcdf]

# To install support for resampling L4C data by TransCom regions
pip install -e pyl4c[transcom]
```

This will also install the project's dependencies. **NOTE: Because the GDAL Python bindings can be difficult to install, I recommend installing them as binaries through your system's package manager.** For instance, on Ubuntu GNU/Linux:

```sh
sudo apt install python3-gdal
```

You may encounter an error installing `pyl4c` from `setup.py`, depending on the version of the GDAL library you have installed. See `setup.py` to check which version of GDAL that is expected. You can install a specific version of the GDAL Python bindings that is consistent with your system installation by:

```sh
pip install GDAL==$(gdal-config --version)
```

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
