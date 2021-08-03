# smap_l4c_pytools

**Author:** K. Arthur Endsley (endsley@umich.edu)  
**Created:** August 2019

This is a collection of Python tools for managing, analyzing, and visualizing SMAP L4C data. In particular:

- Working with data in EASE-Grid 2.0 projection (`ease2.py`;)
- Converting HDF5 geophysical variables to GeoTIFF format (`spatial.py`);
- Creating statistical summaries of SMAP L4C variables or other raster arrays (`utils.py`);
- Reproducing L4C operational model logic (`science.py`);
- Calibrating the L4C model (`apps/calibration`);
- Running the L4C model (`apps/l4c`);
- Aligning and summarizing SMAP L4C variables with TransCom regions (`lib/transcom.py`);

The entire project is contained in the `pyl4c` module. Once installed:

```py
import pyl4c
```

**Needed improvements:**

- [ ] **Commit to either the GDAL 2.x or GDAL 3.x coordinate ordering scheme for EPSG 4326;** see [this Github issue](https://github.com/OSGeo/gdal/issues/1546).
- [ ] **Replace `epsg` module with `pyproj.CRS`** (adding dependency `pyproj`),

---------------

## Setup and Installation

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

---------------

## Documentation

Documentation can be generated with [pdoc3](https://pdoc3.github.io/pdoc/), which can be installed:

```sh
pip install pyl4c[docs]
```

Or simply:

```sh
pip install pdoc3
```

Then, see the `build.sh` script in the `docs/` folder for a hint about how to generate the documentation.

---------------

## Contents

```
__init__.py             --- Essential basics: haversine() function
ease2.py                --- EASE-Grid 2.0 affine transformations
epsg.py                 --- EPSG spatial reference system information
science.py              --- Derived quantities related to scientific objectives
spatial.py              --- Geospatial/ GIS tools
stats.py                --- Various statistical functions
towers.py               --- For working with eddy covariance (EC) flux tower data
utils.py                --- Subsets, compositing, and summary stats of L4C grids

data/
    fixtures.py         --- See NOTE [1]

apps/
    resample.py         --- Downscale L4C variables from 9-km to 1-km EASE-Grid 2.0
    calibration/
        __init__.py     
        main.py         --- CLI for compiling L4C calibration data
        nature.py       --- NatureRunNetCDF4 class, for SMAP L4SM Nature Run data
        legacy.py
    l4c/
        __init__.py     --- L4CForwardProcessPoint class for running L4C simulations
        io.py           --- Data storage abstractions for running the L4C model
        main.py         --- CLI for running L4C point simulations

lib/
    cli.py              --- Classes for building command line interfaces (CLI)
    modis.py            --- For working with MODIS MOD15 and VIIRS VNP15
    netcdf.py           --- Tools for working with NetCDF version 3 or 4 data
    nsidc_download.py   --- Download data from NSIDC
    tcf.py              --- Tools for working with legacy TCF or "land" format data
    transcom.py         --- Work with TransCom regions
    visualize.py        --- Programmatic plotting of L4C data

scripts/                --- See NOTE [2]
```

- NOTE [1]: You can change `ANCILLARY_DATA_PATHS` and `HDF_PATHS` here, as needed. The ancillary products should be stored in `/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/`
- NOTE [2]: Various scripts for convenience. They are intended merely as examples of more comprehensive workflows; no guarantee about their function is made.

### Notes on File Structure

Many of the modules in `apps/` and `lib/` have a `CLI()` class defined. This is an arbitrary Python class intended to be invoked at the module level through Google's `fire` library, i.e., at the bottom of these module scripts there is:

```py
if __name__ == '__main__':
    fire.Fire(CLI)
```

This enables the module to be used at the command line. Typically, the methods implemented on the `CLI()` class have to do with file input-output and batch data processing.

---------------

## EASE-Grid 2.0 Global Grids

Much of the work in this library depends on translating between WGS84 coordinates (longitude-latitude), EASE-Grid 2.0 coordinates (meters easting and northing), and EASE-Grid 2.0 row-column coordinates. Unfortunately, there are multiple ways to do these translations with no clear "best" solution. The historical development of this library traces out the many options:

1. **To convert between WGS84 coordinates and row-column coordinates, we initially relied upon the coordinate arrays stored in the L4C HDF5 granules** (e.g., the `GEO` fields in an NSIDC HDF5 granule). This is the basis of the work in all `pyl4c.utils.get_ease2*` and `pyl4c.utils.subset*` functions.
2. **Later, the `affine` library and `osr.CoordinateTransformation` were tried as these don't require any ancillary HDF5 data granules;** but, unfortunately, they produce *different* results compared to the coordinates stored in the HDF5 granules. This is the basis for `pyl4c.spatial.xy_to_pixel`, a generic function for affine transformations.
3. Finally, the `pyl4c.ease2` module was developed, directly translated from C code (and, in turn, from IDL code) that was written by others.

**To summarize the different approaches, one need only point out the three different ways of getting an array of EASE-Grid 2.0 grid cells as longitude-latitude coordinates:**

- `pyl4c.utils.get_ease2_coords()` used to read the coordinate arrays from an ancillary HDF5 granule (deprecated now), which required network access to the ancillary file; however, this produces the same results as `pyl4c.ease2.ease2_coords()`, so it was deprecated.
- `pyl4c.utils.ease2_coords_approx()` will approximate the coordinates given by the previous function using the affine transformation of the GDAL library.
- `pyl4c.ease2.ease2_coords()` will also use an affine transformation, but uses one with parameters derived from legacy code.

**So, essentially, there's a disagreement between the GDAL-derived affine transformation and the legacy affine transformation.**

The latitude and longitude coordinates for the 9-km EASE-Grid 2.0 global grid is stored in each L4C granule.
However, these coordinates are different than the ones that might be generated from an affine transformation, as in `pyl4c.spatial.xy_to_pixel`.
The function `pyl4c.spatial.ease2_coords_approx` represents my closest match to the coordinates stored in the L4C Vv4040 latitude and longitude fields, **but it is only accurate out to 2-3 decimal places!**
It's not clear what accounts for this discrepancy, i.e., where the L4C Vv4040 granules get their latitude-longitude data from.

---------------

## Design Plans

### Summarizing by Land Cover over Arbitrary Subsets

- Summarization *without* down-scaling... Basically, count how many 1-km subgrid cells of a given PFT are within each 9-km cell; use this to scale the respective PFT mean at 9 km, then summarize.
- Down-scaling with an arbitrary land-cover map (on a similar EASE-Grid 2.0)

## Concepts

### Plant Functional Type (PFT) Means and Cell Size

One important concept to understanding the default behavior of some of these functions is that of the "PFT subgrid." SMAP L4C calculates vegetation and carbon dynamics on a 1-km grid consistent with the MODIS land-cover map (or map of plant functional types, PFTs).
The results are posted to a 9-km grid, however, the mean in each PFT group (mean among all 1-km cells of a given PFT within the larger 9-km cell) is recorded.
So, operations "on a PFT subgrid" use the PFT mean, not the overall mean of the 9-km cell.

### Cell Size and Spatial Rates

Many carbon science variables are expressed as spatial rates, e.g., g C m^-2 ("grams of carbon per square meter").
The mean of a spatial rate is straightforward to calculate but converting the spatial rate to a total stock size (usually g C or "grams of carbon") is slightly tricky.
It requires scaling each spatial rate by the area *prior to* adding up the individual measurements.
So, some tools and scripts in this library have a `scale` argument; it scales individual cell values prior to summarizing them.
It defaults to 1.0, which has no effect on the output.
It should be set to the area of the cell when a total stock size is desired.

For example, 9-km cells contain 81,000,000 square meters (81e6), so this should be the scaling parameter when a total stock size is desired.
Below, we calculate the sum within each PFT *after* scaling the spatial rates; this converts them to a stock size, so the result is total grams of carbon.

```sh
$ python scripts/bulk_summarize_HDF_by_PFT_CONUS.py run ./*.h5 --output_path="~/results.csv" --summaries="('nansum',)" --scale=81e6 --nodata=-9999
```
