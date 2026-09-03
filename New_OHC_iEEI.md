# Plan: Independent Rebuild of OHC and iEEI Calculations

Status: standalone planning document. This describes a from-scratch,
independent reimplementation of the Ocean Heat Content (OHC) and
integrated Earth Energy Imbalance (iEEI) calculations for CESM2 output,
to be used as a cross-check against any prior results. It is written to
be self-contained: it does not assume familiarity with, or reuse of, any
other file in this repository. Everywhere the correct choice isn't
obvious, the document says "determine value/approach" rather than
picking one — that determination should happen by deriving it from first
principles or from the CESM2/POP2 model source and documentation, not by
copying whatever an existing script did.

## 1. Objective

Independently compute, for a set of CESM2 simulations:

1. **Ocean Heat Content (OHC)** — spatial fields and global-mean
   time series, at several depth ranges.
2. **Ocean Heat Flux (OHF)** — global-mean surface heat flux into the
   ocean, used only to cross-check OHC's time evolution.
3. **integrated Earth Energy Imbalance (iEEI)** — the time integral of
   top-of-model absorbed shortwave minus outgoing longwave radiation.
4. A **closure check**: does the time-integrated TOA energy imbalance
   (iEEI) match the change in ocean heat content plus the time-integrated
   ocean surface heat flux, to within expected model drift?

This is a numerical-methods rebuild, not a physics literature review —
the goal is to catch implementation bugs (unit errors, wrong constants,
mis-weighted averages, off-by-one time alignment, etc.) by producing an
independent second implementation and comparing.

## 2. Foundational equations

### 2.1 Ocean Heat Content (OHC)

Per unit ocean-column area, at each horizontal grid point:

```
OHC(x,y,t) = Σ_k  ρ · c_p · T(x,y,k,t) · dz(k)      [J/m^2]
```

- `T` = ocean potential temperature, one value per vertical layer `k`
  (POP2 history variable `TEMP`, native units °C)
- `dz(k)` = layer thickness in the vertical (POP2 variable `dz`, native
  units cm — convert to m)
- `ρ` = seawater density (kg/m^3) — **determine value/approach**: use a
  single reference constant, or the model's own density field (POP2
  variable `RHO`, native units g/cm^3, needs ×1000 conversion to kg/m^3).
  If using a constant, derive it from the CESM2/POP2 source
  (`constants.F90`) rather than an assumed "typical seawater" value —
  candidate constants seen in different contexts are 1025 and
  1026 kg/m^3; confirm which one POP2 itself uses internally for this
  purpose before committing to it.
- `c_p` = specific heat capacity of seawater (J/(kg·K)) — **determine
  value/approach** the same way; candidate constants seen in different
  contexts are 3850 and 3996 J/(kg·K).
- Sum is over vertical levels `k`, optionally restricted to a depth
  range (see depth bins below), using the depth of the bottom of each
  layer (POP2 variable `z_w_bot`, cm) as the boundary test.
- Grid cells with no ocean at that level must be masked out. POP2's
  `KMT` variable gives the number of active vertical levels in each
  water column; a level `k` is valid at a horizontal point only if
  `k <= KMT` there (equivalently, `KMT > 0` to include the column at all
  for surface-level calculations).

Depth bins to compute: full water column, and cumulative bins bounded at
300 m, 700 m, and 2000 m (using `z_w_bot` compared against these
thresholds, converted to the same length units, e.g. cm if `z_w_bot` is
in cm).

Global (or regional) area-weighted mean, using T-grid cell area
(POP2 variable `TAREA`, native units cm^2 — convert to m^2):

```
OHC_global_mean(t) = Σ_(x,y) [ OHC(x,y,t) · TAREA(x,y) ] / Σ_(x,y) TAREA(x,y)     [J/m^2]
                      (sum restricted to valid ocean columns)
```

An unweighted spatial mean is *not* correct here — POP2's displaced-pole
grid has highly non-uniform cell areas, so area weighting is required to
get a physically meaningful global mean.

To get total ocean heat content in Joules rather than a per-area mean,
multiply by total ocean surface area: `OHC_global_mean * ocean_area_m2`,
where `ocean_area_m2 = Σ TAREA` over valid ocean columns only (this is
smaller than the total grid area, since some grid cells are land).

**Determine value/approach — absolute vs. anomaly temperature:**
integrating `ρ·c_p·T` with `T` in Celsius produces a number that depends
on the arbitrary choice of temperature scale (using Kelvin instead would
add a huge constant offset with no physical meaning). Two defensible
approaches:
  - (a) Integrate absolute `TEMP` (as defined above) and only ever
    interpret/plot *differences* of the resulting series (e.g. relative
    to its own first time step, or between two simulations) — never the
    absolute value in isolation.
  - (b) Subtract a reference-period climatology, `T - T_ref`, before
    integrating, where `T_ref` is the time-mean of `TEMP` over a fixed
    reference period (e.g. the first N years of the run, or a
    steady-state control-run period), matching how OHC is usually
    reported in the published literature (e.g. Levitus/NOAA, Cheng et
    al. anomaly-based OHC time series).
  Pick one and apply it consistently; document the choice and the
  reference period used (if any) alongside the results.

### 2.2 Ocean Heat Flux (OHF) — for the closure check only, not part of iEEI

```
OHF(x,y,t) = QFLUX(x,y,t) + SHF(x,y,t)      [W/m^2]
```

`QFLUX` and `SHF` are POP2 surface heat flux history variables.
**Determine value/approach**: confirm the sign convention of each term
(both should represent heat flux *into* the ocean, positive downward,
before being summed — check each variable's `long_name`/sign convention
in its file metadata rather than assuming). `OHF_global_mean` is
computed with the same `TAREA`-weighted average as OHC.

### 2.3 Earth Energy Imbalance (EEI) and integrated EEI (iEEI)

```
EEI(t) = ASR(t) - OLR(t)         [W/m^2]     (absorbed shortwave minus outgoing longwave)
```

**Determine value/approach — which CAM diagnostic variables to use for
ASR/OLR.** CAM history output offers more than one candidate pair for
top-of-atmosphere vs. top-of-model radiative fluxes:
  - net shortwave and net longwave *at the top of the model* (CAM
    variables `FSNT` and `FLNT`)
  - net shortwave *at the top of the atmosphere* and outgoing longwave
    *at the top of the atmosphere* (CAM variables `FSNTOA` and `FLUT`)
  These are not identical (the top of the model is not exactly the top
  of the atmosphere in CAM's finite-volume dynamical core), and CAM also
  writes a directly-diagnosed net TOA imbalance under other variable
  names in some configurations. Check which fields are actually present
  in the available output and which one is the physically appropriate
  "top of atmosphere" for an EEI calculation; do not assume a name
  without checking the file's variable list and `long_name` metadata.

Integrate in time, weighting each month by its length in seconds so
unequal calendar-month lengths don't bias the sum (leap-year handling
should match the calendar attribute of the simulation's time axis,
e.g. `noleap`, `365_day`, or a real Gregorian calendar):

```
seconds_per_month(m) = 86400 * days_in_month(m)
iEEI(t) = Σ_(t'≤t)  EEI(t') · seconds_per_month(t')       [J/m^2]   (cumulative sum over time)
```

Multiply by Earth's surface area to get total Joules:
`earth_SA = 4π·R^2`. **Determine value/approach** for `R`: the literal
mean Earth radius (6,371 km) vs. CAM's own internal planetary radius
constant (which may differ slightly) — check the CAM model source/
documentation for the constant it actually uses internally, since mixing
the two inconsistently across a calculation that also involves the
ocean's grid-derived area (§2.4) will bias the closure check.

`iEEI` should be zeroed at a defined start time (i.e., compute the
cumulative sum only from a chosen start year/month onward) so it
represents "energy imbalance accumulated since a specific reference
point," matching whatever start point is used for the OHC anomaly in the
closure check (§2.4).

### 2.4 OHC ↔ iEEI closure check

The ocean (POP2) grid and the atmosphere (CAM) grid do not cover exactly
the same surface area (different land masks/grid discretizations), so a
correction factor is needed before comparing an ocean-derived quantity
(OHC, OHF) against an atmosphere-derived quantity (iEEI) as if they
applied to the same total surface:

```
ocean_area_m2   = Σ TAREA over valid ocean columns                  (from POP2 grid, §2.1)
global_area_m2  = Σ TAREA over the full POP2 grid (ocean + land)     (from POP2 grid)
surfacearea_factor = ocean_area_m2 / global_area_m2

iOHF     = surfacearea_factor · cumsum(OHF_global_mean · seconds_per_month)
OHC_anom = OHC_global_mean(t) - OHC_global_mean(t0)      (t0 = the same reference start point as iEEI)
```

**Determine value/approach**: whether an additional correction is needed
for the difference between the POP2 ocean-grid surface area and the
CAM atmosphere-grid total surface area specifically (as opposed to using
`global_area_m2` computed directly from the POP2 grid, above) — compute
both grids' total surface areas independently and check whether they
already agree closely enough (both should be close to Earth's true
surface area, ~510×10^6 km^2, and the ocean fraction should be close to
~70%; large deviations indicate a grid-area unit or masking bug worth
finding, which is exactly the kind of bug this rebuild is meant to
surface) before introducing an extra correction factor.

Closure is checked by plotting/comparing `iEEI`, `iOHF`, and `OHC_anom`
(all in J/m^2, all referenced to the same start time `t0`) and by taking
their ratios, which should approach 1.0 if the ocean and atmosphere
energy budgets are consistent. Note: known CESM2(WACCM6) behavior is
that its ocean is not in full radiative-dynamical equilibrium in a long
control run, so some divergence between OHC growth and TOA EEI is
expected in very long integrations (multi-century/millennium runs) even
with a fully correct calculation — don't treat all divergence as a bug.

## 3. CESM2 variable and grid reference

### 3.1 Variable names

**Ocean (POP2 monthly-mean history files, typically named
`*.pop.h.<VAR>.<start>-<end>.nc`):**
`TEMP` (potential temperature, °C), `RHO` (in-situ density, g/cm^3),
`SALT` (salinity), `dz` (layer thickness, cm), `TAREA`/`UAREA` (T-grid/
U-grid cell area, cm^2), `KMT` (number of active vertical levels per
water column), `z_t` (depth of layer center, cm), `z_w_bot` (depth of
layer bottom, cm), `QFLUX`, `SHF` (surface heat fluxes, W/m^2),
`TLONG`/`TLAT` (T-grid longitude/latitude, for plotting/regional
masking).

**Atmosphere (CAM monthly-mean history files, typically named
`*.cam.h0.<start>-<end>.nc` with multiple variables per file):**
candidates for the EEI calculation: `FSNT`, `FLNT`, `FSNTOA`, `FLUT`.
Other commonly available CAM radiation/energy diagnostics that may be
useful for sanity-checking (not required for the core calculation):
`FSNS`/`FSNSC` (surface net SW, all-sky/clear-sky), `FLNS`/`FLNSC`
(surface net LW), `FLNTC`/`FSNTC`/`FSNTOAC` (clear-sky top-of-model/TOA),
`TS` (surface temperature), `LHFLX`/`SHFLX` (surface latent/sensible heat
flux). Always confirm exact variable names and units against the actual
file's metadata (`ncdump -h` or equivalent) rather than assuming a name
is present — CAM output variable lists differ between component-set
configurations.

### 3.2 Grid

Standard CESM2 configuration: POP2 ocean grid `gx1v7` (nominal 1°,
displaced-pole grid), horizontal dimensions `nlat=384, nlon=320`, 60
vertical levels. CAM atmosphere grid `f09` (~1° finite-volume,
`lat`×`lon` roughly 192×288, confirm against the actual files). Grid
metadata needed for the OHC calculation (`TAREA`, `KMT`, `dz`,
`z_w_bot`) can be read directly from any POP2 history file that
contains them (they are typically static/time-invariant fields present
in every monthly file, so any single file suffices), or obtained from an
independent grid-description source for `gx1v7` if a from-scratch,
file-independent grid definition is preferred for the rebuild.

### 3.3 CESM2 case-string convention

CESM2 case names follow the pattern
`b.e21.<component_set>.<grid>.<experiment>.<ensemble_member>`, e.g.
`b.e21.BW1850.f09_g17.CMIP6-piControl.001`. The ensemble member is
usually a 3-digit numeric suffix. When identifying which files belong to
which simulation/ensemble member, parse this string rather than assuming
a fixed directory layout — different experiments may organize their
output directories differently even when case strings follow this
pattern.

## 4. Data locations (input) and target simulations

The rebuild should be run against the same category of CESM2 simulations
as any prior effort, so results are comparable: a pre-industrial control
run, a historical run, a future scenario run (e.g. SSP2-4.5), and one or
more climate-intervention scenario runs (e.g. stratospheric aerosol
injection, marine cloud brightening) branched from that scenario. The
exact case names and ensemble sizes to use should be confirmed against
whatever CESM2 output is actually available on the compute system in
use, rather than assumed — CESM2 output for these experiment classes is
distributed via NCAR's GLADE/campaign storage and NCAR's GDEX data
service; locate the relevant case directories there (each containing
`ocn/proc/tseries/month_1/` and `atm/proc/tseries/month_1/`
subdirectories with the POP2 and CAM monthly time-series files described
in §3.1) and record the exact paths used once found, since GDEX dataset
IDs and GLADE campaign paths can change over time.

## 5. Compute environment

Target NCAR's Casper cluster (PBS job scheduler) for batch processing:
- Confirm/obtain a valid PBS project account code before submitting jobs
  (do not reuse an old account code without confirming it is still
  valid/authorized for this work).
- Use the `casper` queue for data-analysis-scale jobs (not a
  compute-scale HPC queue).
- Build a fresh Python environment with at minimum: `xarray`, `numpy`,
  `pandas`, `netCDF4`, `cftime`, `dask` (for out-of-core/chunked
  processing of large time-series files), and a plotting library
  (`matplotlib`) for the closure-check figures. Do not assume any
  previously-built conda environment still exists or is correctly
  specified — build and pin a fresh environment for this rebuild.

## 6. Steps to build the rebuild, independently

1. Resolve every "determine value/approach" item in §2 (physical
   constants, absolute-vs-anomaly convention, ASR/OLR variable choice,
   Earth radius constant, area-mismatch correction) by deriving the
   answer from the CESM2/POP2/CAM model documentation and source, or
   from the file metadata itself — not by consulting any prior
   implementation. Record each decision and its justification.
2. Write an independent OHC computation: for each input simulation,
   read `TEMP`, `dz`, `TAREA`, `KMT`, `z_w_bot` for the ocean grid,
   compute `OHC(x,y,t)` at the full column and the three depth bins
   (§2.1), mask non-ocean columns, and compute the `TAREA`-weighted
   global mean.
3. Write an independent OHF computation (§2.2) for the same simulations,
   if the closure check will be run.
4. Write an independent iEEI computation: for each input simulation,
   read the chosen ASR/OLR variable pair, compute month-length-weighted
   `EEI`, and cumulative-sum it into `iEEI` from a defined start time
   (§2.3).
5. Run both computations over every target simulation (§4), verifying
   basic physical sanity at each step (e.g. ocean area is a plausible
   fraction of Earth's surface area; OHC values and EEI values fall in
   physically reasonable ranges; time axes align and cover the expected
   period without gaps or duplicated months).
6. Run the closure check (§2.4) for at least the control run and one
   scenario run, and inspect whether the ratio of ocean-derived to
   atmosphere-derived cumulative energy approaches 1.0 as expected,
   flagging any large, unexplained divergence for further investigation
   rather than dismissing it.
7. Compare the rebuilt OHC and iEEI time series (and the closure-check
   result) against any prior results for the same simulations. Any
   discrepancy is a lead: trace it back to a specific step (unit
   conversion, masking, weighting, variable choice, or time alignment)
   in either the rebuild or the prior implementation.
