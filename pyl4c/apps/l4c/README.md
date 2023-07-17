L4C Science
===========

The L4C Science model is a Python version of the operational L4C algorithm. It follows the L4C model specification exactly, but due to minor differences in spatial reference system (SRS) definitions and a persistent difference in soil organic carbon (SOC) initialization [1], it does not produce results exactly identical to that of L4C Ops. However, the results are very similar and the environmental responses are exactly identical [2].

The core model is called `pyl4c.apps.l4c.main.L4CForwardProcessPoint` because it is intended for "point" data only; i.e., it computes fluxes (and SOC state) for specific geographic point locations, usually eddy covariance (EC) flux towers. This is in contrast to the L4C Ops model (written in C) that runs for the entire global grid.

[1] A lot of effort has been put into diagnosing the small differences between L4C Science and L4C Ops, as the algorithms are the same. It has been determined that the differences in fluxes are due to 1) Different fPAR values at certain sites, particularly those sites contaminated by water pixels (an issue likely related to different SRS definitions); 2) Different SOC values in each pool that are used on model Day 1 (March 31, 2015). The SOC values in L4C Science were taken directly from the SOC restart file that is ostensibly used in L4C Ops, so it is not apparent what accounts for the different results.
[2] That is, the Emult, Kmult, Tmult values are exactly the same in L4C Science as in L4C Ops.


Requirements
------------

An input drivers HDF5 file is required; see `pyl4c/apps/calibration/README.md` for a specification of what this file should contain.

Required drivers and units:

```
PAR     MJ m-2 day-1
fPAR    Dimensionless, on range: [0, 1]
SMRZ    Wetness, on range: [0, 100]
SMSF    Wetness, on range: [0, 100]
Tsurf   degrees K
Tmin    degrees K
Tsoil   degrees K
VPD     Pascals
```

**We distinguish between three types of data variables:**

- *State variables* influence the driving of the model; an initial state (at time t=0) is required to start running and the variable state is updated at each time step (t) based on the value at step (t-1). See `pyl4c.apps.l4c.L4CState`.
- *Driver variables* drive the model but are not in themselves updated; they are exogenous variables with values fixed at the start of the simulation. See `pyl4c.apps.l4c.L4CDrivers`.
- *Flux variables* do NOT drive the model except in some intermediate step step, e.g., NEE is a function of GPP and RH. Flux variables are the primary "outputs" of the model. See `pyl4c.apps.l4c.L4CState` (flux variables are stored the same way as state variables).


Running L4C Science
-------------------

The L4C Science model can be run using the following short Python script which, in practice, should be modified to suit the user's needs:

```py
import datetime
import warnings
import h5py
from pyl4c.data.fixtures import BPLUT # To use the Version 4 BPLUT
from pyl4c.apps.l4c.main import L4CForwardProcessPoint

# Configuration for all 356 sites in the post-launch period
config = {
    'bplut': BPLUT,
    'inputs_file_path': 'L4_C_Vv4040_state_and_drivers_2015-2017.h5',
    'site_count': 356, # Number of flux tower sites
    'time_steps': 1007 # Number of time steps (days) to simulate
}

# Optional list of timestamps, to help in interpreting the output files
dt = [
    (datetime.datetime(2015, 3, 31) + datetime.timedelta(days = d))\
        .strftime('%Y-%m-%d')
    for d in range(0, config['time_steps'])
]

# It's a good idea to check the inputs file before running
with h5py.File(config['inputs_file_path'], 'r') as hdf:
    report(hdf, config)

# Optionally filter out NumPy warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    # Create and run the model!
    l4c = L4CForwardProcessPoint(config, debug = False)
    l4c.run()

    # Finally, write out CSV file for each flux/ state variable
    l4c.fluxes.serialize(
        'run_YYYYMMDD_VERSION_ID.csv', time_labels = dt, prec = 2)
    l4c.state.serialize(
        'run_YYYYMMDD_VERSION_ID.csv', time_labels = dt, prec = 2)
```
