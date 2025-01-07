L4C Calibration
===============

**Important considerations:**

- In the L4C operational code, the autotrophic respiration ($R_A$) fraction, `fraut`, instead of plant carbon-use efficiency (CUE), is used to partition GPP into NPP and $R_A$. This may be confusing because plant CUE is reported in the public BPLUT, while it is (1 - CUE) that is actually used in the operational code. Here, we calibrate CUE and convert it to (1 - CUE) when updating the BPLUT.


Calibration Procedure
---------------------

Calibration of SMAP L4C can be done entirely through the Unix shell (command line) by running the `optimize.py` (non-linear least-squares optimization) and/or `mcmc.py` (Markov Chain Monte Carlo) scripts. While it's possible to calibrate the GPP and RECO models using either of these two approaches, we recommend:

- `optimize.py` for calibrating the GPP model
- `mcmc.py` for calibrating the RECO `CUE` parameter (with other free parameters)
- `optimize.py` again for calibrating the other RECO free parameters, after fixing `CUE`
- Finally, running `tune_soc()` in `optimize.py` to calibrate SOC parameters


**Required data:**

- A configuration file:
  - For `optimize.py`, see this template: `pyl4c/data/files/config_L4C_calibration_V8.yaml`
  - For `mcmc.py`, see this template: `pyl4c/data/files/config_L4C_MCMC_calibration_V8.yaml`
- HDF5 file with driver datasets (see "Harmonized HDF Specification" below); this file should contain all of the input datasets necessary to calibrate GPP and RECO. In a typical re-calibration, the soil moisture fields (`smsf`, `smrz`) need to be updated. The path to this file is specified as the `drivers_file` property in the calibration configuration JSON file.
- HDF5 file with eddy covariance (EC) flux tower observation data.
- A recent BPLUT CSV file; typically, this is the BPLUT from the previous calibration (i.e., current operational product). The path to this file is specified as the `BPLUT_file` property in the calibration configuration JSON file.


**Calibration steps:**

Note that some commands require a specific PFT to be chosen (most notably, when running an optimization, as calibration is performed separately for each PFT). In such cases, the `pft` command is used to select a PFT and `pft <pft>` indicates that a numeric PFT code should be given in place of `<pft>`. Other commands require one of a fininte number of arguments, e.g., `<smrz|tmin>` and `<smsf|tsoil>` indicate that only one of the strings "smrz", "tmin", "smsf", or "tsoil" should be provided.

1. Update the GPP optimization configuration file, starting with `config_L4C_calibration.yaml` as a template.
2. Run the GPP parameter optimization using `optimization.py` with multiple random trials: `python optimize.py pft <pft> tune-gpp`
3. Update the RECO optimization configuration file, starting with `config_L4C_MCMC_calibration.yaml` as a template.
4. Run the RECO calibration using `mcmc.py`: `python mcmc.py pft <pft> tune-reco`
5. Burn and thin the chains as needed, then select optimal `CUE` parameter, updating the RECO configuration file.
6. Run the RECO optimization for the remaining free parameters: `python optimize.py pft <pft> tune-reco --fixed=[CUE]`


Harmonized HDF Specification
----------------------------

Going forward, HDF5 files used in both calibration and forward model runs should have the following layouts.

**Driver Data for Forward Run**

```
# Shape is indicated, where N is number of sites and T is number of time steps
coords/
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

site_id       (N)       Site names/ unique identifiers
site_pft_9km  (N)       Dominant PFT class in 9-km pixel

state/
    PFT       (N x 81)      1-km PFT classes
    porosity  (N)           (Optional) Porosity from L4SM land model

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
