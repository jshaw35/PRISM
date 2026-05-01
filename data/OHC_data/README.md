# CESM2 Ocean Heat Content (OHC) Calculation Results

## Overview

This directory contains the calculated ocean heat content (OHC) from CESM2-WACCM monthly output using the full equation of state (model-computed density).

## Files

### 1. `cesm2_ohc_global_timeseries.nc`
NetCDF file containing the global monthly OHC time series.

**Dimensions:**
- `time`: 600 months (2035-2084, 50 years)

**Variables:**
- `OHC_global`: Global mean ocean heat content [J/m²]
  - Shape: (600,)
  - Mean: 7.44e+08 J/m²
  - Range: [-8.99e+08, 2.16e+09] J/m²
  - Std Dev: 5.81e+08 J/m²

### 2. `ohc_validation_plots.png`
4-panel figure showing:
- **Top-left**: Full time series of global OHC
- **Top-right**: OHC anomaly (deviation from mean)
- **Bottom-left**: OHC rate of change (monthly differences)
- **Bottom-right**: Summary statistics and methodology

## Calculation Methodology

**Formula:**
```
OHC [J/m²] = Σ(ρ(T,S) × c_p × ΔT × dz)
```

Where:
- `ρ(T,S)` = seawater density (from model RHO field)
- `c_p` = 3850 J/(kg·K) (specific heat capacity)
- `ΔT` = temperature anomaly from reference period
- `dz` = vertical layer thickness

**Key Parameters:**
- Reference period: First 10 years (2035-2044, months 0-119)
- Integration depth: Full ocean depth (60 levels, ~5000m)
- Equation of state: Full (using model RHO field)
- Grid: POP gx1v7 (384 × 320 horizontal, 60 vertical levels)

## Loading in Python

```python
import xarray as xr
import numpy as np

# Load the data
ds = xr.open_dataset('cesm2_ohc_global_timeseries.nc')

# Extract variables
ohc = ds['OHC_global'].values  # numpy array [J/m²]
time = ds['time'].values       # time coordinates

# Compute statistics
print(f"Mean OHC: {np.mean(ohc):.2e} J/m²")
print(f"OHC trend: {np.polyfit(np.arange(len(ohc)), ohc, 1)[0]:.2e} J/m²/month")

# Time averaging
ohc_annual = ohc.reshape(-1, 12).mean(axis=1)
```

## Physical Interpretation

**Positive anomalies** = Warmer than reference period (2035-2044)
**Negative anomalies** = Cooler than reference period

The strong long-term upward trend reflects ocean warming in the SSP245 scenario.
The seasonal cycle (~12-month period) shows natural ocean heat variability.

## Source Data

Input files:
- `/gdex/data/d651059/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI.001/ocn/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI.001.pop.h.TEMP.203501-208412.nc`
- `/gdex/data/d651059/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI.001/ocn/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI.001.pop.h.RHO.203501-208412.nc`

## Code

The calculation is performed by `cesm2_ohc_calculator.py` located in the PRISM root directory.

**Key features:**
- Chunked processing (12 months at a time) for memory efficiency
- Uses pre-calculated model density (RHO field)
- Masks land cells using ocean model mask (KMT)
- Includes spatial OHC calculation function for regional analysis

See `OHC_IMPLEMENTATION_GUIDE.md` for detailed methodology and usage instructions.

## Contact

Generated using OpenCode CESM2 OHC calculator
Date: 2026-04-30
