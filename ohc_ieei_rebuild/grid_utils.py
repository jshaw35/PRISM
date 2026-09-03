"""Grid geometry helpers for POP2 ocean output and CAM atmosphere output."""
import numpy as np
import xarray as xr


def ocean_column_mask(kmt):
    """Boolean mask (nlat, nlon): True where the column has at least one
    active ocean level."""
    return kmt > 0


def ocean_level_mask(kmt, z_t):
    """Boolean mask (z_t, nlat, nlon): True where vertical level k is an
    active ocean level in that column.

    POP2's KMT gives the 1-indexed count of active vertical levels in each
    water column, so level k (1-indexed) is active where k <= KMT.
    """
    level_index = xr.DataArray(
        np.arange(1, z_t.sizes["z_t"] + 1), dims="z_t", coords={"z_t": z_t["z_t"]}
    )
    return level_index <= kmt


def depth_bin_masks(z_w_bot, depths_m=(300, 700, 2000)):
    """Boolean masks (z_t) selecting vertical levels whose bottom lies within
    each depth range, keyed by the depth bound in meters. `z_w_bot` is
    expected in centimeters (POP2 native units)."""
    return {depth_m: (z_w_bot < depth_m * 100.0) for depth_m in depths_m}


def area_weighted_global_mean(data, area, valid_mask, dims=("nlat", "nlon")):
    """Area-weighted mean of `data` over `dims`, restricted to `valid_mask`.

    Cells outside `valid_mask` are given zero weight and their (possibly NaN)
    data values are replaced with zero, so they contribute nothing to either
    the numerator or denominator instead of turning the whole reduction into
    NaN.
    """
    weights = area.where(valid_mask, other=0.0)
    data_filled = data.where(valid_mask, other=0.0)
    return data_filled.weighted(weights).mean(dim=list(dims))


def atmosphere_area_weights(ds):
    """Horizontal area weights for a CAM lat/lon grid, from the Gaussian
    latitude-weight variable `gw` (present on most CAM history files and
    consistent with the model's own grid quadrature), broadcast evenly over
    longitude. This is a relative weight only (not a physical area in m^2)
    and is only valid for computing area-weighted means, not absolute areas.

    Raises ValueError if `gw` is not present. Call `cos_lat_area_weights`
    explicitly instead if a cos(latitude) approximation is acceptable.
    """
    if "gw" not in ds:
        raise ValueError(
            "'gw' (Gaussian latitude weights) not found in dataset; call "
            "cos_lat_area_weights(ds) explicitly if a cos(latitude) "
            "approximation is acceptable instead."
        )
    return ds["gw"] * xr.ones_like(ds["lon"])


def cos_lat_area_weights(ds):
    """Approximate horizontal area weights for a CAM lat/lon grid, using
    cos(latitude). This is a relative weight only (not a physical area in
    m^2) and is only valid for computing area-weighted means, not absolute
    areas. Use only when `gw` is unavailable and this approximation has been
    deliberately chosen -- it is never used automatically."""
    lat_weight = np.cos(np.deg2rad(ds["lat"]))
    return lat_weight * xr.ones_like(ds["lon"])
