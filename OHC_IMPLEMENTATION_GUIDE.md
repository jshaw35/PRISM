#!/usr/bin/env python3
"""
CESM2 Ocean Heat Content (OHC) Calculator - Implementation Guide

This document describes the complete workflow for calculating global ocean heat
content from CESM2/POP2 monthly model output using the full equation of state.
"""

# ==============================================================================
# METHODOLOGY
# ==============================================================================

"""
Ocean Heat Content (OHC) Calculation

CESM2 computes ocean temperature and density fields at 60 vertical levels with
varying thickness. This script integrates these fields to produce a global mean
ocean heat content time series.

FORMULA:
--------
OHC [J/m²] = Σ_z (ρ(z,t) × c_p × ΔT(z,t) × dz(z))

Where:
  - ρ(z,t)     = seawater density at depth z, time t [kg/m³]
  - c_p         = specific heat capacity of seawater = 3850 J/(kg·K)
  - ΔT(z,t)    = T(z,t) - T_ref(z)   [K] (temperature anomaly)
  - dz(z)       = thickness of vertical layer z [m]
  - Σ_z         = vertical integration over all ocean levels

REFERENCE STATE:
----------------
The reference (baseline) temperature is computed as the mean over the first
10 years (months 0-119) of simulation. All subsequent OHC values are computed
as anomalies relative to this reference period.

Key feature: Using anomalies instead of absolute values eliminates the large
bias from cold deep ocean water and focuses on changes in ocean heat content.

EQUATION OF STATE:
------------------
This implementation uses the FULL equation of state by utilizing the
pre-calculated RHO (density) fields from CESM2 output. This accounts for
both temperature AND salinity variations, providing a more accurate
representation than simplified constant-density approaches.

Alternative approaches would be:
1. Constant density (ρ = 1025 kg/m³) - simplest but least accurate
2. Compute ρ from TEMP+SALT using UNESCO EOS - more flexible but slower

INTEGRATION DOMAIN:
-------------------
- Vertical: Full ocean depth (all 60 levels, from ~0.5m to ~5000m)
- Spatial:  All grid cells (masked by KMT ocean mask)
- Time:     Monthly resolution, 600 months (50 years: 2035-2084)

UNITS:
------
Input:  TEMP [K], RHO [g/cm³], dz [cm], TAREA [cm²]
Output: OHC [J/m²]

Conversion factors:
  - RHO: multiply by 1000 to convert g/cm³ → kg/m³
  - dz:  divide by 100 to convert cm → m
  - (TAREA is grid cell areas, only used for averaging)
"""

# ==============================================================================
# DATA REQUIREMENTS
# ==============================================================================

"""
CESM2 Output Files Required:
1. Temperature file: *.pop.h.TEMP.*.nc
   - Contains: TEMP (potential temperature [K])
   - Also contains grid metadata: dz, TAREA, KMT, lat/lon
   
2. Density file: *.pop.h.RHO.*.nc  
   - Contains: RHO (seawater density [g/cm³])
   - Same time and spatial dimensions as TEMP

Note: Files are typically separated by variable in CESM2 post-processing.

Grid Information (from TEMP file):
  - dz        : Layer thickness [cm], shape (60,)
  - TAREA     : Cell areas [cm²], shape (384, 320)
  - KMT       : Ocean mask, shape (384, 320)
               (0=land, >0=ocean depth in levels)
  - z_t       : Depth coordinate, shape (60,)
  
Physical constants:
  - cp_seawater = 3850 J/(kg·K)
  - reference period = first 10 years (months 0-119)
"""

# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

"""
Command Line Usage:
-------------------
python cesm2_ohc_calculator.py

This will:
1. Load CESM2 output files from example directory
2. Compute reference climatology (first 10 years)
3. Calculate monthly OHC anomalies for full time series
4. Save results to NetCDF file
5. Generate validation plots

Programmatic Usage:
-------------------
from cesm2_ohc_calculator import CESM2OHCCalculator

# Initialize
calc = CESM2OHCCalculator(
    temp_file='/path/to/file.pop.h.TEMP.*.nc',
    rho_file='/path/to/file.pop.h.RHO.*.nc'  # Optional if using naming convention
)

# Compute time series
time, ohc_global = calc.compute_global_ohc_timeseries(
    output_file='/path/to/output.nc',
    chunk_size=12  # Process 1 year at a time
)

# Generate spatial OHC field for month 0
ohc_spatial = calc.compute_spatial_ohc_field(time_index=0)
# Returns shape (384, 320) with OHC per unit area

Modifying for Different Data:
------------------------------
1. Update input file paths in main() function
2. Adjust reference period if needed:
   - Change ref_end parameter in _get_reference_climatology()
   - Default: 120 months (10 years) = first 10 years
   - Use 240 for first 20 years, etc.
3. Change chunk_size if memory is limited:
   - Smaller chunks (e.g., 6) use less memory
   - Larger chunks (e.g., 24) are faster
"""

# ==============================================================================
# OUTPUT STRUCTURE
# ==============================================================================

"""
NetCDF Output File: cesm2_ohc_global_timeseries.nc

Dimensions:
  - time: 600 (months)

Variables:
  - OHC_global: Global mean OHC time series
    - Shape: (600,)
    - Units: J/m²
    - Description: Temperature anomaly integrated over full ocean depth

Coordinates:
  - time: Monthly time values (cftime.DatetimeNoLeap objects)

Global Attributes:
  - title: "CESM2 Global Ocean Heat Content"
  - method: "OHC = Σ(ρ × c_p × ΔT × dz)"
  - reference_period: "First 10 years (months 0-119)"
  - depth_integration: "Full ocean depth (60 levels)"
  - equation_of_state: "Full (using model RHO field)"
  - source_file: Path to original TEMP file

Variable Attributes:
  - long_name: "Global mean ocean heat content"
  - units: "J/m²"
  - description: "Temperature anomaly integrated over full ocean depth"

Example Reading in Python:
---------------------------
import xarray as xr
import numpy as np

ds = xr.open_dataset('cesm2_ohc_global_timeseries.nc')
ohc = ds['OHC_global'].values  # numpy array [J/m²]
time = ds['time'].values       # time coordinates

# Convert to float64 and compute statistics
ohc_mean = np.mean(ohc)
ohc_std = np.std(ohc)
print(f"OHC: {ohc_mean:.2e} ± {ohc_std:.2e} J/m²")
"""

# ==============================================================================
# COMPUTATIONAL NOTES
# ==============================================================================

"""
Memory Efficiency:
------------------
The script uses chunked processing to fit on login nodes with minimal memory:
- Each chunk loads 12 months of TEMP (~150 MB) and RHO (~150 MB)
- Processes month-by-month (one month loaded at a time)
- Reference climatology computed efficiently in passes
- Total peak memory: ~500 MB

Processing Time:
----------------
On GLADE login nodes:
- 600 months with chunk_size=12: ~5-10 minutes
- Bottleneck: File I/O (netcdf4 library reading from network storage)

Optimization Tips:
------------------
1. Use larger chunk_size if more memory available (e.g., 24)
2. Consider dask.array for parallel computation on compute nodes
3. Pre-compute reference climatology separately if processing many ensembles
4. Use NCO tools (ncra, etc.) for quick temporal averages

Validation:
-----------
The validation plots show:
1. Full time series (top-left): Should show seasonal cycle + long-term trend
2. Anomaly (top-right): Deviations from mean (should oscillate around 0)
3. Change rate (bottom-left): First differences (short-timescale variability)
4. Statistics (bottom-right): Summary of key metrics

Red flags in validation:
- Anomalies drifting consistently positive/negative → likely reference issue
- Extremely high/low spikes → check for NaN/masked cells
- Missing seasonal cycle → verify reference period calculation
"""

# ==============================================================================
# PHYSICAL INTERPRETATION
# ==============================================================================

"""
Understanding the Results:

Mean OHC (~7.4e+08 J/m²):
- Positive: Warmer than reference period (first 10 years)
- Negative: Cooler than reference period
- Magnitude: Indicates how much energy is stored above/below baseline

Seasonal Cycle:
- Monthly values show clear 12-month periodicity
- Driven by ocean surface heating (boreal summer max, winter min)
- Amplitude: ~1-2×10^8 J/m² (typical for ocean seasonal signal)

Long-term Trend:
- Should show warming over SSP245 scenario (moderate warming projection)
- Can extract linear trend via regression on time
- Rate of OHC increase indicates global heat uptake by ocean

Anomaly Time Series:
- Useful for identifying climate modes (ENSO, PDO, etc.)
- Temporal averaging reveals decadal variability
- Compare with observations to validate model performance

Units Context:
- 1 J/m² = 0.001 W⋅s/m² = 0.001 "Joule per meter squared"
- 10^8 J/m² over ocean (3.6×10^8 km²) ≈ 3.6×10^16 J = 36 exajoules
- For reference: annual solar forcing ≈ 170 W/m² = 5.4e+15 J/m² per year
"""

# ==============================================================================
# KNOWN LIMITATIONS & FUTURE WORK
# ==============================================================================

"""
Current Limitations:
1. Uses simple global mean (volume-weighted would be more accurate)
2. No regional breakdown (can implement using REGION_MASK variable)
3. Only processes global values (spatial fields via separate function)
4. Doesn't account for model bias correction
5. Reference period fixed at first 10 years

Future Enhancements:
1. Volume-weighted OHC: OHC_vol = Σ(ρ × c_p × ΔT × dV)
   where dV = dz × TAREA (requires careful grid handling)

2. Regional OHC: Use REGION_MASK variable to compute by ocean basin
   
3. Depth-resolved OHC: Store OHC(depth) profile for all time steps
   
4. Ensemble statistics: Process multiple ensemble members and compute
   mean, std dev, percentiles
   
5. Comparison with observations: Load observed OHC and compute biases
   
6. Vertically integrated anomalies: OHC at different depth levels
   (0-300m, 300-700m, 700-2000m, 2000m+)

7. Derive secondary metrics:
   - Heat content change rate [W/m²]
   - Vertical heat distribution
   - Mixed layer heat content
"""

if __name__ == '__main__':
    print(__doc__)
