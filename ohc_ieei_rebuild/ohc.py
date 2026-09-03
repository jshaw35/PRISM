"""Independent Ocean Heat Content (OHC) calculation for POP2 monthly output.

See New_OHC_iEEI.md, section 2.1, for the governing equation and the
constant/convention choices this implementation makes:
  - density and specific heat: POP2's own internal reference constants
  - temperature convention: absolute TEMP, not an anomaly
"""
import pandas as pd
import xarray as xr

from grid_utils import area_weighted_global_mean, depth_bin_masks, ocean_column_mask, ocean_level_mask

# POP2's own internal reference constants (constants.F90):
#   rho_sw = 4.1 / 3.996 g/cm^3
#   cp_sw  = 3.996e7 erg/g/K = 3996 J/(kg*K)
RHO_SW_KG_M3 = 1000.0 * 4.1 / 3.996
CP_SW_J_KG_K = 3996.0

DEPTH_BINS_M = (300, 700, 2000)

REQUIRED_OCEAN_VARS = ["TEMP", "dz", "TAREA", "KMT"]
REQUIRED_OCEAN_COORDS = ["z_t", "z_w_bot"]


def shift_noleap_time_back_one_month(ds):
    """POP2 monthly-mean history files stamp each month's average with a
    timestamp at the first instant of the *following* month (CESM's period-
    average convention: the time coordinate marks the end of the averaging
    interval). Shift every timestamp back by one month so `time.dt.month`
    reflects the month the data actually describes.
    """
    old_time = ds["time"].values
    date_type = type(old_time[0])
    new_time = []
    for t in old_time:
        year, month = t.year, t.month - 1
        if month == 0:
            month = 12
            year -= 1
        new_time.append(date_type(year, month, t.day, t.hour, t.minute, t.second))
    return ds.assign_coords(time=("time", new_time))


def open_ocean_dataset(temp_paths, grid_path=None, chunks=None):
    """Open POP2 TEMP history file(s) and ensure the grid variables/
    coordinates OHC needs are present, merging them in from `grid_path` (any
    POP2 history file containing the static grid fields) if they are missing
    from `temp_paths` itself.
    """
    ds = xr.open_mfdataset(temp_paths, combine="by_coords", chunks=chunks, decode_timedelta=True)
    missing = [v for v in REQUIRED_OCEAN_VARS if v not in ds]
    missing += [c for c in REQUIRED_OCEAN_COORDS if c not in ds.coords and c not in ds]
    if missing:
        if grid_path is None:
            raise ValueError(
                f"Missing required ocean grid variables/coordinates {missing}; "
                "set config.OCN_GRID_FILE to a POP2 history file that contains them."
            )
        grid_ds = xr.open_dataset(grid_path)
        ds = ds.merge(grid_ds[missing], compat="override", join="override")
    return ds


def compute_ohc(ds, rho=RHO_SW_KG_M3, cp=CP_SW_J_KG_K, depth_bins_m=DEPTH_BINS_M):
    """Compute Ocean Heat Content (New_OHC_iEEI.md sec. 2.1).

    Requires `ds` to contain: TEMP [degC], dz [cm], TAREA [cm^2], KMT,
    z_t, z_w_bot [cm].

    Returns a Dataset with:
      - OHC: per-column-area OHC [J/m^2], dims (..., ohc_depth, nlat, nlon)
        where ohc_depth in {-1 (full column), 300, 700, 2000} (meters)
      - OHC_global_mean: TAREA-weighted global mean [J/m^2], dims (..., ohc_depth)
    and attrs global_area_m2, ocean_area_m2, rho_kg_m3, cp_j_kg_k.
    """
    dz_m = ds["dz"] / 100.0
    tarea_m2 = ds["TAREA"] / 1e4
    level_mask = ocean_level_mask(ds["KMT"], ds["z_t"])
    column_mask = ocean_column_mask(ds["KMT"])

    ohc_per_vol = (rho * cp * ds["TEMP"]).where(level_mask)  # J/m^3
    ohc_layers = ohc_per_vol * dz_m  # J/m^2 per vertical layer

    full_depth = ohc_layers.sum(dim="z_t", skipna=True)
    depth_masks = depth_bin_masks(ds["z_w_bot"], depth_bins_m)
    binned = [ohc_layers.where(mask).sum(dim="z_t", skipna=True) for mask in depth_masks.values()]

    ohc_depth_index = pd.Index([-1, *depth_bins_m], name="ohc_depth")
    ohc_per_area = xr.concat([full_depth, *binned], dim=ohc_depth_index)
    ohc_per_area = ohc_per_area.where(column_mask)

    global_area_m2 = tarea_m2.sum(dim=["nlat", "nlon"])
    ocean_area_m2 = tarea_m2.where(column_mask, other=0.0).sum(dim=["nlat", "nlon"])
    ohc_global_mean = area_weighted_global_mean(ohc_per_area, tarea_m2, column_mask)

    out = xr.Dataset({"OHC": ohc_per_area, "OHC_global_mean": ohc_global_mean})
    out.attrs["global_area_m2"] = float(global_area_m2.values)
    out.attrs["ocean_area_m2"] = float(ocean_area_m2.values)
    out.attrs["rho_kg_m3"] = rho
    out.attrs["cp_j_kg_k"] = cp
    out["OHC"].attrs.update(units="J m-2", long_name="Ocean heat content per unit column area, by depth bin (-1 = full column)")
    out["OHC_global_mean"].attrs.update(units="J m-2", long_name="Area-weighted global-mean ocean heat content, by depth bin")
    return out
