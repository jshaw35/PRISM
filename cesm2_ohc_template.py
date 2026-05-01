"""
Quick reference template for calculating Ocean Heat Content (OHC) from CESM2 output.

This module demonstrates the basic workflow for OHC computation.
For detailed methodology, see: CESM2_OHC_CALCULATION_GUIDE.md
"""

import xarray as xr
import numpy as np
import pop_tools
from pathlib import Path


def compute_ohc_from_cesm2(
    temp_file: str,
    reference_period: tuple = (0, 240),
    depth_range: tuple = (0, 62),
    output_file: str = None,
) -> xr.Dataset:
    """
    Calculate Ocean Heat Content from CESM2 temperature output.

    Parameters
    ----------
    temp_file : str
        Path to CESM2 temperature file (case.pop.h.*.nc)
    reference_period : tuple
        Time indices for reference/baseline period (start, end)
        Default: (0, 240) = first 20 years for 12 times/year
    depth_range : tuple
        Depth level indices to integrate over (start, end)
        Default: (0, 62) = all 62 levels (full depth)
    output_file : str, optional
        Path to save output. If None, only returns dataset.

    Returns
    -------
    ds_ohc : xr.Dataset
        Dataset containing:
        - 'OHC_per_m2' : OHC per unit area [J/m²]
        - 'OHC_global' : Global OHC time series [J]
        - Metadata with calculation parameters

    Example
    -------
    >>> ohc = compute_ohc_from_cesm2(
    ...     'CESM2_TEMP.nc',
    ...     reference_period=(0, 240),
    ...     depth_range=(0, 62)
    ... )
    >>> ohc['OHC_global'].plot()  # Plot global OHC time series
    """

    # =========================================================================
    # STEP 1: Load temperature data
    # =========================================================================
    print(f"Loading temperature data from: {temp_file}")
    temp = xr.open_dataset(temp_file)["TEMP"]
    print(f"  Shape: {temp.shape}")
    print(f"  Dims: {list(temp.dims)}")

    # =========================================================================
    # STEP 2: Load grid information
    # =========================================================================
    print("Loading POP grid information...")
    grid = pop_tools.get_grid("POP_gx1v7")  # Or appropriate resolution
    dz = grid["dz"]  # Layer thicknesses [m]
    tarea = grid["TAREA"]  # Cell areas [cm²]
    kmt = grid["KMT"]  # Ocean mask

    # Convert TAREA from cm² to m²
    tarea_m2 = tarea / 1e4

    # =========================================================================
    # STEP 3: Define reference period (baseline)
    # =========================================================================
    print(f"Computing reference state from times {reference_period}")
    temp_ref = temp.isel(time=slice(*reference_period)).mean("time")
    print(f"  Reference climatology shape: {temp_ref.shape}")

    # =========================================================================
    # STEP 4: Compute temperature anomaly
    # =========================================================================
    print("Computing temperature anomaly...")
    temp_anom = temp - temp_ref

    # =========================================================================
    # STEP 5: Select depth range for integration
    # =========================================================================
    if depth_range != (0, 62):
        print(f"Integrating depths: {depth_range}")
        temp_anom = temp_anom.isel(z_t=slice(*depth_range))
        dz_selected = dz.isel(z_t=slice(*depth_range))
    else:
        print("Integrating full depth (62 levels)")
        dz_selected = dz

    # =========================================================================
    # STEP 6: Calculate OHC per unit area [J/m²]
    # =========================================================================
    print("Computing OHC per unit area...")
    rho = 1025.0  # kg/m³ (seawater density)
    cp = 3850.0  # J/(kg·K) (specific heat capacity)

    ohc_per_m2 = rho * cp * (temp_anom * dz_selected).sum("z_t")
    print(f"  OHC per m² shape: {ohc_per_m2.shape}")

    # =========================================================================
    # STEP 7: Calculate global OHC [Joules]
    # =========================================================================
    print("Computing global OHC...")
    ohc_global = (ohc_per_m2 * tarea_m2).sum(["nlat", "nlon"])
    print(f"  Global OHC shape: {ohc_global.shape}")
    print(f"  Global OHC range: {float(ohc_global.min()):.2e} to {float(ohc_global.max()):.2e} J")

    # =========================================================================
    # STEP 8: Package results
    # =========================================================================
    print("Packaging results...")
    ds_ohc = xr.Dataset(
        {
            "OHC_per_m2": ohc_per_m2,
            "OHC_global": ohc_global,
        },
        attrs={
            "description": "Ocean Heat Content from CESM2",
            "formula": "OHC = rho * cp * sum(dT * dz)",
            "rho": f"{rho} kg/m³",
            "cp": f"{cp} J/(kg·K)",
            "reference_period": f"times {reference_period}",
            "depth_range_indices": f"{depth_range}",
            "units_per_m2": "J/m²",
            "units_global": "J",
        },
    )

    # =========================================================================
    # STEP 9: Optionally save to file
    # =========================================================================
    if output_file is not None:
        print(f"Saving to: {output_file}")
        ds_ohc.to_netcdf(output_file)
        print("  ✓ Saved successfully")

    return ds_ohc


def compute_regional_ohc(ohc_per_m2: xr.DataArray, mask_name: str = "default") -> dict:
    """
    Compute regional OHC time series from per-unit-area OHC.

    Parameters
    ----------
    ohc_per_m2 : xr.DataArray
        OHC per unit area [J/m²] with dims (time, nlat, nlon)
    mask_name : str
        Name of POP region mask to use

    Returns
    -------
    regional_ohc : dict
        Dictionary with regional time series
    """
    print(f"Computing regional OHC using mask: {mask_name}")

    # Load grid and region mask
    grid = pop_tools.get_grid("POP_gx1v7")
    tarea = grid["TAREA"] / 1e4  # Convert to m²
    region_mask = pop_tools.region_mask_3d("POP_gx1v7", mask_name=mask_name)

    regional_ohc = {}
    for region_name in region_mask.region.values:
        # Get mask for this region
        mask = region_mask.sel(region=region_name) == 1

        # Compute regional OHC
        ohc_region = (ohc_per_m2.where(mask) * tarea).sum(["nlat", "nlon"])
        regional_ohc[region_name] = ohc_region
        print(f"  {region_name}: mean = {float(ohc_region.mean()):.2e} J")

    return regional_ohc


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Main calculation
    ohc = compute_ohc_from_cesm2(
        temp_file="path/to/CESM2_TEMP.nc",
        reference_period=(0, 240),  # First 20 years
        depth_range=(0, 62),  # Full depth
        output_file="OHC_output.nc",
    )

    # Compute regional OHC
    ohc_regional = compute_regional_ohc(ohc["OHC_per_m2"], mask_name="default")

    # Plot results
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Global OHC time series
    ohc["OHC_global"].plot(ax=ax1, marker="o", linewidth=2)
    ax1.set_ylabel("Global OHC [J]")
    ax1.set_title("Ocean Heat Content Evolution")
    ax1.grid(True, alpha=0.3)

    # Regional OHC
    for region, ts in ohc_regional.items():
        ts.plot(ax=ax2, label=region)
    ax2.set_ylabel("Regional OHC [J]")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ohc_analysis.png", dpi=150)
    print("\n✓ Plots saved to: ohc_analysis.png")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Global OHC [J]:")
    print(f"  Mean:  {float(ohc['OHC_global'].mean()):.3e}")
    print(f"  Min:   {float(ohc['OHC_global'].min()):.3e}")
    print(f"  Max:   {float(ohc['OHC_global'].max()):.3e}")
    print(f"  Trend: {float(np.polyfit(range(len(ohc['OHC_global'])), ohc['OHC_global'], 1)[0]):.3e} J/month")
