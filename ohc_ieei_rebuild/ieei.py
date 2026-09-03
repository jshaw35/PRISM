"""Independent integrated Earth Energy Imbalance (iEEI) calculation for CAM
monthly output.

See New_OHC_iEEI.md, section 2.3, for the governing equations and the
ASR/OLR variable choice this implementation makes (FSNT / FLNT, i.e. net
shortwave/longwave at the top of the model — set in config.py).
"""
import numpy as np

from grid_utils import atmosphere_area_weights

# CAM's internal planetary radius constant (shr_const_rearth), used so the
# total-Joules conversion is consistent with the model's own physics rather
# than the literal mean Earth radius.
CAM_EARTH_RADIUS_M = 6.37122e6


def global_mean(da, ds_for_weights, weight_func=atmosphere_area_weights):
    """Area-weighted horizontal mean of a CAM lat/lon field.

    Uses `gw`-based weights by default, which raises if `gw` is missing.
    Pass `weight_func=grid_utils.cos_lat_area_weights` explicitly to opt
    into the cos(latitude) approximation instead.
    """
    weights = weight_func(ds_for_weights)
    return da.weighted(weights).mean(dim=["lat", "lon"])


def compute_eei(asr, olr):
    """EEI = ASR - OLR [W/m^2]."""
    eei = asr - olr
    eei.attrs["units"] = "W m-2"
    eei.attrs["long_name"] = "Earth energy imbalance (ASR - OLR)"
    return eei


def month_length_weights(time):
    """Seconds in each month, calendar-aware: uses the time coordinate's own
    `days_in_month`, so it respects whatever calendar (noleap, 365_day,
    standard, ...) the simulation actually uses."""
    seconds = time.dt.days_in_month * 86400.0
    seconds.attrs["units"] = "s"
    return seconds


def compute_ieei(eei, start_time=None):
    """Cumulative time integral of EEI, in J/m^2, optionally zeroed starting
    at `start_time` (anything comparable to the time coordinate's labels,
    e.g. an ISO date string)."""
    if start_time is not None:
        eei = eei.sel(time=slice(start_time, None))
    weights = month_length_weights(eei["time"])
    ieei = (eei * weights).cumsum(dim="time")
    ieei.attrs["units"] = "J m-2"
    ieei.attrs["long_name"] = "Integrated Earth energy imbalance since start_time"
    return ieei


def total_joules(ieei_per_area, earth_radius_m=CAM_EARTH_RADIUS_M):
    """Convert a per-area iEEI [J/m^2] to total Joules using the Earth
    surface area implied by `earth_radius_m`."""
    earth_surface_area_m2 = 4.0 * np.pi * earth_radius_m**2
    out = ieei_per_area * earth_surface_area_m2
    out.attrs["units"] = "J"
    out.attrs["long_name"] = "Integrated Earth energy imbalance since start_time, total"
    return out
