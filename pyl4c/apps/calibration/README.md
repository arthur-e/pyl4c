L4C Calibration
===============

**Differences from the Matlab version:**

- In the Matlab version, the autotrophic respiration fraction, `fraut`, is calibrated instead of plant carbon-use efficiency (CUE). Plant CUE is reported in the public BPLUT, which led to this confusion, where it is (1 - CUE) that is actually calibrated in the Matlab code. Calibrating on CUE works just as well, and that is what is done in this Python version. Moreover, the bounds for the `fraut` (or 1 - CUE) parameter in the Matlab code were given as [0, 0.7], which doesn't make sense since it allows for autotrophic respiration to go to zero. These bounds make more sense for CUE, however, since a CUE of 0.7 (i.e., autotrophic respiration can't be less than 30\% of GPP) is more physically realistic.


Calibration Procedure
---------------------

Calibration of SMAP L4C can be done entirely through the Unix shell (command line) by running `pyl4c/apps/calibration/main.py` as a Python script. File paths for required input datasets should be specified in the calibration configuration JSON file, e.g.:

```json
{
  "BPLUT_file": "...",
  "drivers_file": "...",
  "scratch_file": "...",
  "towers_file": "..."
}
```

**See an example (and the default file used in calibration) here:** `pyl4c/data/files/config_calibration.json`

On the NTSG network, the required files are stored at:

  `/anx_lagr3/arthur.endsley/SMAP_L4C/calibration`

**Required data:**

- HDF5 file with driver datasets (see "Harmonized HDF Specification" below); this file should contain all of the input datasets necessary to calibrate GPP and RECO. In a typical re-calibration, the soil moisture fields (`smsf`, `smrz`) need to be updated. The path to this file is specified as the `drivers_file` property in the calibration configuration JSON file.
- HDF5 file with eddy covariance (EC) flux tower observation data; currently, this file is named `Fluxnet2015_LaThuile_tower_data_for_356_sites.h5` and should not need to be changed. The path to this file is specified as the `towers_file` property in the calibration configure JSON file.
- A recent BPLUT CSV file; typically, this is the BPLUT from the previous calibration (i.e., current operational product). The path to this file is specified as the `BPLUT_file` property in the calibration configuration JSON file.

**During calibration, a "scratch" file is created.** This file contains 365-day climatologies for input drivers, tower observations (optionally filtered), and the working BPLUT that is progressively updated. The path to this file is specified as the `scratch_file` property in the calibration configuration JSON file.

**Calibration steps:**

Note that some commands require a specific PFT to be chosen (most notably, when running an optimization, as calibration is performed separately for each PFT). In such cases, the `pft` command is used to select a PFT and `pft <pft>` indicates that a numeric PFT code should be given in place of `<pft>`. Other commands require one of two arguments, "gpp" or "reco", to indicate which is being calibrated; these are indicated as `<gpp|reco>` and only one should be provided. Similarly, `<smrz|tmin>` and `<smsf|tsoil>` indicate that only one of the strings "smrz", "tmin", "smsf", or "tsoil" should be provided.

1. Create the scratch file: `python main.py setup`
2. Optionally, preview the effect of filtering the tower data: `python main.py pft <pft> filter-preview <gpp|reco> <window_size>`
3. Optionally, apply a smoothing filter to the the data for that PFT: `python main.py pft <pft> filter <gpp|reco> <window_size>`
4. Optionally, apply a smoothing filter to the data for ALL PFTs: `python main.py filter-all <gpp|reco> <window_size>`
5. Optionally, plot the current GPP response function: `python main.py pft <pft> plot-gpp <smrz|tmin>`
6. Run the GPP calibration with 20 random trials: `python main.py pft <pft> tune-gpp --trials=20`
7. Optionally, plot the new GPP response function: `python main.py pft <pft> plot-gpp <smrz|tmin>`
8. Optionally, plot the current RECO response function: `python main.py pft <pft> plot-reco <smsf|tsoil>`
9. Run the RECO calibration with 20 random trials: `python main.py pft <pft> tune-reco --trials=20`
10. Optionally, plot the new RECO response function: `python main.py pft <pft> plot-reco <smsf|tsoil>`

The updated BPLUT is stored in the scratch file and can be dumped to a Python pickle file:

```py
from pyl4c.apps.calibration import BPLUT

bp = BPLUT(hdf5_path = 'scratch_file.h5')
bp.pickle('new_bplut.pickle')
```


Harmonized HDF Specification
----------------------------

Going forward, HDF5 files used in both calibration and forward model runs should have the following layouts.

**Driver Data for Forward Run**

```
# Shape is indicated, where N is number of sites and T is number of time steps
coords/
    grid_1km_subgrid_col_idx  (N x 81)
    grid_1km_subgrid_row_idx  (N x 81)
    grid_9km_idx              (N x 2)
    lng_lat                   (N x 2)

drivers/
    fpar      (T x N x 81)  fPAR, as a proportion on [0, 1]
    par       (T x N)       [MJ m-2 day-1]
    smrz      (T x N)       Wetness (%) units, re-scaled using srmz_min and smrz_max = 100
    smrz0     (T x N)       (Optional) Original SMRZ before re-scaling
    smrz_min  (N)           (Optional) Site-level minimum SMRZ
    smrz_max  (N)           (Optional) Site-level maximum SMRZ
    smsf      (T x N)       Wetness (%) units
    tmin      (T x N)       [deg K]
    tsoil     (T x N)       [deg K]
    tsurf     (T x N)       [deg K]
    vpd       (T X N)       [Pascals]

legacy/                 (Optional) Any data from prior versions goes here
    lc_dom

site_id       (N)       Site names/ unique identifiers
site_pft_9km  (N)       Dominant PFT class in 9-km pixel

state/
    PFT       (N x 81)      1-km PFT classes
    porosity  (N)           (Optional) Porosity from L4SM land model
    npp_sum   (N x 81)      Annual sum of net primary production (NPP)
    soil_organic_carbon
              (3 x N x 81)  SOC content (g C m-2) for each of 3 SOC pools

time          (T x 4)   Where T is the number of time steps
```

**Tower Data for Calibration**

```
GPP       (T x N)
NEE       (T x N)
RECO      (T x N)
site_id   (N)
```


Multiple Calibration Datasets
-----------------------------

**Because of the different diagnostic products and different L4C versions over the years, there are different ways of assembling the calibration dataset.**

The annual NPP sum and soil organic carbon (SOC) pool data are not needed for calibration, but are often included in these datasets so they can be used for forward model runs as well.


------------------------------

**L4C Version 5 Ops (Vv5000) Calibration Data:**

**See: `pyl4c.apps.calibration.subset`,** although this is not the script used to create this dataset. It turned out to be much easier to just use the Version 4 dataset, updating the L4SM fields with Nature Run v8.1 data.

This dataset is for 356 flux tower sites:

```
/anx_lagr3/arthur.endsley/SMAP_L4C/calibration/v5_Y2020/L4_C_tower_site_drivers_NRv8-1_for_356_sites.h5
```

**The specific provenance of each field:**

- SMSF, SMRZ, and Tsoil are taken from the L4SM Nature Run v8.1 dataset, downloaded from NASA NCCS Discover;
- PAR, Tmin, FT, VPD are taken from the MERRA-2 re-analysis dataset; I simply used what was already on our network;
  - `/anx_lagr2/laj/smap/Natv72/subset/L4_C_input_NRv72_Vv4xxxx_smapMergedCalVal.h5`
- **fPAR is taken straight from the Version 4 calibration dataset; see `pyl4c.apps.calibration.legacy.LegacyTowerDataset.fpar_daily()`**


------------------------------

**L4C Version 4/ L4C Nature Run v7.2 "Legacy" Calibration Data:**

**See: `pyl4c.apps.calibration.legacy`**

This dataset, for 356 flux tower sites in the combined La Thuile and Fluxnet 2015 network, can be created with the `pyl4c.apps.calibration.legacy` module. The `main()` function in this module will assemble a new calibration dataset. The Version 4 calibration data can be found under the new HDF5 fields layout in this file:

```
/anx_lagr3/arthur.endsley/SMAP_L4C/calibration/v5_Y2020/L4_C_tower_site_drivers_NRv7-2_for_356_sites.h5
```

**Not all of the original inputs could be found.** Most of the data are sourced directly from a single file used in prior calibrations, which has the *old* fields layout:

```
/anx_lagr2/laj/smap/Natv72/subset/L4_C_input_NRv72_Vv4xxxx_smapMergedCalVal.h5
```

SOC and the daily litterfall (annual NPP sum) are taken from a series of sparse, TCF output files; the "C0, C1, C2" files are assumed to be the "Day 0" SOC state in the fast, medium, and slow pools for the Nature Run:

```
/anx_lagr3/laj/smap/natv72/prelaunch/land/tcf_natv72_C0_M01land_0001365.flt32
/anx_lagr3/laj/smap/natv72/prelaunch/land/tcf_natv72_C1_M01land_0001365.flt32
/anx_lagr3/laj/smap/natv72/prelaunch/land/tcf_natv72_C2_M01land_0001365.flt32
/anx_lagr3/laj/smap/natv72/prelaunch/land/tcf_natv72_npp_sum_M01land.flt32
```

These files are in the sparse TCF format and were inflated using the binary:

```
/anx_v2/laj/smap/code/mkgrid/src/mkgrid
```

**The specific provenance of each field:**

- SMSF, SMRZ, and Tsoil are taken from the `MET` group in `L4_C_input_NRv72_Vv4xxxx_smapMergedCalVal.h5`
- PAR, Tmin, Tsurf (used for freeze/thaw calculation), and VPD are also taken from the `MET` group in `L4_C_input_NRv72_Vv4xxxx_smapMergedCalVal.h5`
- SMRZ is rescaled (using a site-maximum SMRZ of 100%)
- PAR truly is PAR, i.e., it is converted from the downwelling short-wave radiation recorded in `L4_C_input_NRv72_Vv4xxxx_smapMergedCalVal.h5`
- PFT is extracted from the 1-km PFT map using an affine transformation of the tower site coordinates
