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

- `config.py` — Glade paths, case string, physical constants.
  `GLADE_USERNAME`/`PBS_PROJECT_ACCOUNT` are filled in for `wkamp`
  (account `UCUB0207`).
- `grid_utils.py` — ocean masking/area-weighting and atmosphere
  area-weighting helpers.
- `ohc.py` — `compute_ohc`, plus dataset loading and the POP2/CAM monthly
  time-stamp fix (`shift_noleap_time_back_one_month`).
- `ieei.py` — `compute_eei`, `compute_ieei`, month-length weighting,
  global-mean and total-joules helpers.
- `run_ohc_piControl.py`, `run_ieei_piControl.py` — CLI entry points, with
  optional inclusive `--start-year`/`--end-year` filtering (applied to the
  shifted, true-calendar-month time axis — both scripts now apply
  `shift_noleap_time_back_one_month` before filtering, so a shared
  `--start-year`/`--end-year` pair selects the same calendar months from
  both pipelines).
- `energy_fluxes.py`, `run_fluxes_piControl.py` — energy-flux pipeline:
  every heat flux into and out of the ocean (POP2 `SHF` and its eight
  coupler components, `QFLUX`, `TFW_T`, `RESID_T`), the sea-ice (CICE),
  land (CLM) and atmosphere (CAM TOA and surface) terms it exchanges energy
  with, and global-mean `TS`, all as global-total time series in W (sea-ice
  heat content in J).
- `energy_budget_summary.ipynb` — the closure analysis: OHC vs iEEI, each
  energy balance checked as verified/questioned, the mismatch in scale, and
  why surface temperature stays flat while the system gains energy.
- `test_pipeline_smoke.py` — minimal smoke test: loads a few months from
  the first two files of each pipeline's input, runs every pipeline
  function, sanity-checks the output, and round-trips a NetCDF write/read,
  then cleans up after itself.
- `environment.yml` — conda environment spec (for building a fresh env);
  not needed if reusing an existing environment like `wk_MHWs` below.
- `job_scripts/*.pbs` — Casper (PBS) job scripts, using the `wk_MHWs`
  conda environment at `/glade/work/wkamp/conda-envs/wk_MHWs`:
  - `test_ohc_ieei_smoke.pbs` — run this first to confirm the pipelines work.
  - `compute_ohc_piControl_100yr.pbs`, `compute_ieei_piControl_100yr.pbs`,
    `compute_fluxes_piControl_100yr.pbs` — year 1 through year 100 of the
    piControl run.
  - `compute_ohc_piControl.pbs`, `compute_ieei_piControl.pbs` — full-record
    runs (no year filtering); still reference `environment.yml` in a
    comment rather than `wk_MHWs` — update before using.

## Before running on Glade

1. `config.OCN_GRID_FILE` can stay `None` for the piControl case: its
   `TEMP`/`FSNT`/`FLNT` time-series files already carry `TAREA`/`KMT`/`dz`/
   `z_w_bot`/`gw`. Only set it if pointing this at a different case whose
   archive layout strips those static grid variables out.
2. Confirm `config.OCN_TSERIES_DIR` / `ATM_TSERIES_DIR` still resolve to
   real files (`ls` them) — GLADE campaign-storage paths can move.
3. Confirm the `wk_MHWs` conda environment still has `xarray`, `numpy`,
   `pandas`, `netCDF4`, `cftime`, `dask` — it did as of the last check.

## Running

```
qsub job_scripts/test_ohc_ieei_smoke.pbs          # run first
qsub job_scripts/compute_ohc_piControl_100yr.pbs
qsub job_scripts/compute_ieei_piControl_100yr.pbs
qsub job_scripts/compute_fluxes_piControl_100yr.pbs
```

or interactively on a Casper session:

```
conda activate /glade/work/wkamp/conda-envs/wk_MHWs
python test_pipeline_smoke.py
python run_ohc_piControl.py --start-year 1 --end-year 100
python run_ieei_piControl.py --start-year 1 --end-year 100
python run_fluxes_piControl.py --start-year 1 --end-year 100
```

Output:
- `{OUTPUT_ROOT}/b.e21.BW1850.f09_g17.CMIP6-piControl.001.OHC.y1-100.nc` —
  `OHC` (per-area, by depth bin `-1`/300/700/2000 m), `OHC_global_mean`
  (per m^2 of *ocean* -- not directly comparable to `iEEI_global_mean`, which
  is per m^2 of *Earth*), and `dOHC_total_joules` (total heat gained since the
  first time step, directly comparable to `iEEI_total_joules`), and
  `ocean_volume_m3` per depth bin.
- `{OUTPUT_ROOT}/b.e21.BW1850.f09_g17.CMIP6-piControl.001.iEEI.y1-100.nc` —
  `EEI_global_mean`, `iEEI_global_mean`, `iEEI_total_joules`.
- `{OUTPUT_ROOT}/b.e21.BW1850.f09_g17.CMIP6-piControl.001.fluxes.y1-100.nc` —
  one variable per budget term (see each variable's `long_name`/`source`
  attrs); divide by the `earth_area_m2` attr for W per m^2 of Earth.
- Running without `--end-year` (the original two `job_scripts/*.pbs`, not
  the `_100yr` ones) drops the `.y<start>-<end>` suffix and writes
  `...OHC.nc` / `...iEEI.nc` instead.

## Sanity checks worth running on the output

- `ocean_area_m2 / global_area_m2` (an `OHC` output attr) should be close
  to Earth's actual ocean fraction (~0.7); a large deviation points to a
  masking or unit bug.
- `OHC_global_mean` should show only slow drift over the piControl run, no
  jumps or discontinuities.
- `EEI_global_mean` should be small and centered near zero for a
  well-equilibrated piControl run; a persistent large offset suggests a
  variable-choice or unit problem rather than real model drift.

## Closure check

The Ocean Heat Flux (OHF) and OHC-vs-iEEI closure check (`New_OHC_iEEI.md`
sec. 2.2/2.4) are built by the energy-flux pipeline and analysed in
`energy_budget_summary.ipynb`.
