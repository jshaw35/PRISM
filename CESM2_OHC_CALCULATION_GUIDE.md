# Ocean Heat Content (OHC) Calculation for CESM2: Research Summary

## 1. TYPICAL METHOD FOR OHC CALCULATION

### Basic Physics
Ocean Heat Content represents the total thermal energy stored in the ocean. The fundamental calculation integrates temperature over volume:

```
OHC = ρ * c_p * ∫∫∫ ΔT dV
```

Where:
- **ρ** = density of seawater ≈ 1025 kg/m³ (varies with temperature/salinity)
- **c_p** = specific heat capacity of seawater ≈ 3850 J/(kg·K)
- **ΔT** = temperature anomaly relative to a reference baseline (K)
- **dV** = volume element (m³)

### For Model Output (Discrete Form)
In CESM2 ocean output, OHC per unit area (J/m²) is calculated as:

```
OHC [J/m²] = Σ_depth (ρ * c_p * ΔT[z] * dz)
```

For a single water column:
- Sum over all vertical levels (z)
- **dz** = thickness of each depth layer (m)
- **ΔT[z]** = temperature anomaly at depth z

### Reference State
- OHC is always calculated as an **anomaly** from a reference state
- Common choices: climatological mean, pre-industrial baseline, or specific period
- Result is change in heat content relative to baseline

---

## 2. REQUIRED OUTPUT FIELDS FROM CESM2

### Primary Variables
1. **TEMP** - Ocean potential temperature (°C)
   - Location: Cell centers on T-grid
   - Dimensions: (time, z_t, nlat, nlon)
   
2. **SALT** - Ocean salinity (PSU)
   - Optional but needed for accurate density calculation
   - Dimensions: (time, z_t, nlat, nlon)

### Grid/Coordinate Variables
1. **z_t** - T-grid depth coordinate (meters)
   
2. **dz** or vertical layer thickness:
   - Usually stored as a variable in POP output
   - Or derived from z_w (w-point depths)
   
3. **TAREA** - T-grid cell area (cm²) or (m²)
   - Typically: (nlat, nlon)
   
4. **KMT** - Number of ocean levels at each T-point
   - Indicates ocean depth mask (0 = land, 1-62 = water)

### Optional but Useful
- **ULAT, ULON** or **TLAT, TLONG** - Grid coordinates
- **REGION_MASK** - For regional analysis
- **TLAT, TLON** - Grid corner coordinates

---

## 3. EXISTING IMPLEMENTATIONS

### A. POP-Tools (NCAR)
**Repository:** https://github.com/NCAR/pop-tools

**Installation:**
```bash
conda install -c conda-forge pop-tools
# or
pip install pop-tools
```

**Key Functions:**
- `pop_tools.get_grid()` - Load POP grid for CESM2
- `pop_tools.eos()` - Equation of state (compute density)
- `pop_tools.compute_pressure()` - Depth to pressure conversion
- `pop_tools.region_mask_3d()` - Generate region masks

**Example Usage:**
```python
import pop_tools
import xarray as xr

# Get grid information
grid = pop_tools.get_grid('POP_gx1v7')

# Compute density
density = pop_tools.eos(salt=salt_data, temp=temp_data, depth=depth)
```

**Advantages:**
- Official NCAR tool for POP/CESM2 ocean analysis
- Handles ocean-specific features (tripole grid, masking)
- Well-documented API
- Actively maintained

### B. xarray + xgcm Approach
**Tools:**
- `xarray` - Data manipulation
- `xgcm` - Generalized Coordinate transformations

**Example Workflow:**
```python
import xarray as xr
import numpy as np

# Load data
ds = xr.open_dataset('CESM2_ocean_output.nc')

# Compute reference state (e.g., mean over first 10 years)
temp_ref = ds['TEMP'].isel(time=slice(0,120)).mean('time')

# Compute anomaly
temp_anom = ds['TEMP'] - temp_ref

# Get density (approximately constant or use pop_tools)
rho = 1025  # kg/m³
cp = 3850   # J/(kg·K)

# OHC calculation
dz = xr.open_dataset('grid_info.nc')['dz']  # Layer thickness
ohc = rho * cp * (temp_anom * dz).sum('z_t')

# Result: ohc has dimensions (time, nlat, nlon)
```

### C. Intake-ESM Integration
**For CESM2-LE and large ensemble data:**

```python
import intake

# Load CESM2 data catalogs
cat = intake.open_esm_datastore(...)
ds = cat.to_dask()

# Apply OHC calculation across ensemble members
```

---

## 4. PYTHON PACKAGES FOR CESM2 OCEAN ANALYSIS

### Essential
1. **pop-tools** (NCAR)
   - Purpose: POP/CESM2 ocean utilities
   - Functions: Grid, EOS, region masks
   - Status: Active, well-supported

2. **xarray**
   - Purpose: N-dimensional data handling
   - Essential for NetCDF work
   - Status: Standard in climate science

3. **xgcm**
   - Purpose: General circulation model utilities
   - Functions: Grid-aware calculations, transformations
   - Status: Active

### Supporting
4. **dask** - Parallel/lazy computation for large datasets
5. **scipy** - Scientific computing routines
6. **numpy** - Numerical operations
7. **xesmf** - Regridding (ESM version)
8. **netCDF4** - Low-level NetCDF access
9. **pandas** - Time series handling

### Data Access
10. **intake-esm** - Browse/load CMIP6/CESM2-LE data
11. **pooch** - Data downloading/caching

---

## 5. ACADEMIC REFERENCES & METHODS

### Key References
- **Levitus et al. (2009)** - Ocean heat content analysis methodology
- **Roemmich & Gilson (2009)** - Argo-based OHC calculations
- **Cheng et al. (2016+)** - Global ocean heat content trends

### Typical Analysis Approaches

**Method 1: Layer-integrated OHC (0-700m, 0-2000m, etc.)**
```
OHC[0-700m] = Σ(ρ * cp * ΔT * dz) for z=0 to 700m
```

**Method 2: Full-depth OHC**
```
OHC[full] = Σ(ρ * cp * ΔT * dz) for all vertical levels
```

**Method 3: Regional OHC**
```
OHC[region] = Σ_time,lat,lon,depth (ρ * cp * ΔT * dz * area)
Result in Joules or Watts (W = J/time)
```

---

## 6. CESM2 SPECIFIC CONSIDERATIONS

### File Organization
- CESM2 ocean output: `case.pop.h.YYYY-MM.nc` (history files)
- Multiple output frequencies available: yearly, monthly, daily
- Grid information often in separate file: `*.gx1v7_grid.nc`

### Key Dimensions
- **z_t**: 62 vertical levels (depth centers)
- **z_w**: 62 vertical interfaces (depth edges)
- **nlat, nlon**: 384 × 320 (gx1v7 resolution)

### POP Grid Features
- **Tripole grid** - Special handling at poles
- **Irregular spacing** - Refinement near equator
- **Cell areas vary** - Must use TAREA for proper weighting
- **Ocean mask** - KMT indicates ocean vs land

### Output Conventions
- Temperature: Potential Temperature (°C)
- Salinity: Practical Salinity Units (PSU)
- Depth: meters
- Area: cm² (need conversion to m²)
- Velocity: cm/s (need conversion to m/s)

---

## 7. TYPICAL WORKFLOW FOR OHC CALCULATION

```python
import xarray as xr
import numpy as np
import pop_tools

# Step 1: Load data
temp = xr.open_dataset('CESM2_TEMP.nc')['TEMP']
grid = pop_tools.get_grid('POP_gx1v7')

# Step 2: Define reference period
temp_ref = temp.sel(time=slice('1900', '1920')).mean('time')

# Step 3: Compute anomaly
temp_anom = temp - temp_ref

# Step 4: Get layer thicknesses and areas
dz = grid['dz']
tarea = grid['TAREA'] / 1e4  # Convert cm² to m²

# Step 5: Compute OHC (J/m²)
rho = 1025  # kg/m³
cp = 3850   # J/(kg·K)
ohc_per_m2 = rho * cp * (temp_anom * dz).sum('z_t')

# Step 6: Global OHC (Joules)
ohc_global = (ohc_per_m2 * tarea).sum(['nlat', 'nlon'])

# Step 7: Regional OHC (optional)
region_mask = pop_tools.region_mask_3d('POP_gx1v7')
# ... apply masking ...
```

---

## 8. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Basic Formula** | OHC = ρ × c_p × Σ(ΔT × dz) |
| **Units** | J/m² (per unit area) or J (total) |
| **ρ (density)** | 1025 kg/m³ (or variable with S,T) |
| **c_p (heat capacity)** | 3850 J/(kg·K) |
| **Reference State** | Climatology or baseline period |
| **Primary Output Fields** | TEMP, SALT, z_t, dz, TAREA, KMT |
| **Main Tool** | pop-tools (NCAR) |
| **Supporting Tools** | xarray, xgcm, dask, scipy |
| **Grid Type** | Tripole (needs special handling) |
| **Time Integration** | Sum over all vertical levels |
| **Spatial Integration** | Multiply by grid cell area |

---

## 9. KEY TAKEAWAYS FOR PRISM PROJECT

1. **Primary Tool**: Use `pop-tools` for grid handling and EOS calculations
2. **Core Calculation**: OHC = ρ × c_p × Σ(ΔT × dz) where ΔT is anomaly from baseline
3. **Required Files**: TEMP (3D temperature), grid information (dz, TAREA, KMT)
4. **Reference State**: Define baseline period carefully (typically pre-industrial or early control)
5. **Units**: Result is J/m² (per unit ocean area) - multiply by TAREA for total
6. **Regional Analysis**: Use pop_tools.region_mask_3d() for ocean regions
7. **Performance**: Use dask/xarray for efficient computation on large datasets
8. **Validation**: Compare with published Argo-based OHC estimates or CESM2 model diagnostics

