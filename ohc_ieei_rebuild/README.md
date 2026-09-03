# OHC / iEEI Rebuild — 1850 PI Control

Independent implementation of the Ocean Heat Content (OHC) and integrated
Earth Energy Imbalance (iEEI) calculations described in `../New_OHC_iEEI.md`,
scoped for now to the CESM2 1850 pre-industrial control run
(`b.e21.BW1850.f09_g17.CMIP6-piControl.001`) on NCAR's Glade filesystem.
This module is self-contained: it does not import or depend on any other
code in this repository.

## Decisions made for this build (see `New_OHC_iEEI.md` sec. 2 for the options)

- **Density / specific heat**: POP2's own internal reference constants
  (`constants.F90`): `rho = 4.1/3.996 * 1000 ≈ 1025.5 kg/m^3`,
  `cp = 3996 J/(kg*K)`.
- **Temperature convention**: absolute `TEMP` (not an anomaly) — treat the
  OHC time series as something to difference/trend, not to read as an
  absolute quantity in isolation.
- **ASR/OLR variables**: CAM `FSNT` (net shortwave, top of model) and
  `FLNT` (net longwave, top of model).
- **Earth radius** (for converting iEEI per-area to total joules): CAM's
  own internal constant, `6.37122e6 m`.

## Files

- `config.py` — Glade paths, case string, physical constants, placeholders
  to fill in.
- `grid_utils.py` — ocean masking/area-weighting and atmosphere
  area-weighting helpers.
- `ohc.py` — `compute_ohc`, plus dataset loading and the POP2 monthly
  time-stamp fix.
- `ieei.py` — `compute_eei`, `compute_ieei`, month-length weighting,
  global-mean and total-joules helpers.
- `run_ohc_piControl.py`, `run_ieei_piControl.py` — CLI entry points.
- `environment.yml` — conda environment spec.
- `job_scripts/*.pbs` — Casper (PBS) job scripts.

## Before running on Glade

1. Edit `config.py`: set `GLADE_USERNAME` and `PBS_PROJECT_ACCOUNT`, and set
   `OCN_GRID_FILE` if the piControl TEMP files don't already carry
   `TAREA`/`KMT`/`dz`/`z_w_bot` (some archive layouts strip static grid
   variables out of single-variable-per-file time series — check with
   `ncdump -h` before assuming).
2. Update the `#PBS -A` account code in both `job_scripts/*.pbs` files to
   match `PBS_PROJECT_ACCOUNT`.
3. Build the conda environment: `conda env create -f environment.yml` (or
   `mamba env create -f environment.yml`), then update the `conda activate`
   line (currently a comment) in the two `.pbs` job scripts to match where
   it was built.
4. Confirm `config.OCN_TSERIES_DIR` / `ATM_TSERIES_DIR` still resolve to
   real files (`ls` them) — GLADE campaign-storage paths can move.

## Running

```
qsub job_scripts/compute_ohc_piControl.pbs
qsub job_scripts/compute_ieei_piControl.pbs
```

or interactively on a Casper session:

```
python run_ohc_piControl.py
python run_ieei_piControl.py
```

Output:
- `{OUTPUT_ROOT}/b.e21.BW1850.f09_g17.CMIP6-piControl.001.OHC.nc` —
  `OHC` (per-area, by depth bin `-1`/300/700/2000 m) and
  `OHC_global_mean`.
- `{OUTPUT_ROOT}/b.e21.BW1850.f09_g17.CMIP6-piControl.001.iEEI.nc` —
  `EEI_global_mean`, `iEEI_global_mean`, `iEEI_total_joules`.

## Sanity checks worth running on the output

- `ocean_area_m2 / global_area_m2` (an `OHC` output attr) should be close
  to Earth's actual ocean fraction (~0.7); a large deviation points to a
  masking or unit bug.
- `OHC_global_mean` should show only slow drift over the piControl run, no
  jumps or discontinuities.
- `EEI_global_mean` should be small and centered near zero for a
  well-equilibrated piControl run; a persistent large offset suggests a
  variable-choice or unit problem rather than real model drift.

## Not yet built

Ocean Heat Flux (OHF) and the OHC-vs-iEEI closure check
(`New_OHC_iEEI.md` sec. 2.2/2.4) are out of scope for this first,
PI-control-only build.
