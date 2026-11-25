"""Annual zonal TOA radiative fluxes (ASR, OLR, EEI, iEEI) in latitude bands for SSP2-4.5, ARISE-1.5 and ARISE-1.0.

Usage:
    python zonal_toa_fluxes.py [--start 2035] [--end 2069] [--band-width 5] [--out PATH] [--workers N]

Reads monthly lat x lon FSNT (ASR) and FLNT (OLR) for members 001-010 of each condition, and writes one NetCDF file
with dims (condition, member, year, lat), lat being the band centers, for zonal_means.ipynb. The notebook only has to
reduce the annual values over 2035-2069: sum iEEI_* over year for the period total, and average the other variables
over year for the period mean (noleap years all have 365 days, so the mean of the annual means is the period mean).

Conventions, matching J04_CESM_EEI_timeseries_multiensemble.py and ohc_cases.py:
  * EEI = FSNT - FLNT (top of model), positive = the system gains energy. OLR keeps CAM's upward-positive sign.
  * iEEI for a year is the energy the band gained over that year: integral of EEI dt with each month weighted by its
    length (noleap, no leap days) -- J04's compute_iEEI summed per year instead of accumulated.
  * Months are labeled by the start of their averaging interval (time_bnds[:, 0]); CESM stamps each monthly mean at
    the end of the interval. The ARISE-1.0 yearly files repeat December at each file boundary; repeats are checked to
    hold the same values before one copy is kept. Annual means weight months by their length.
  * Cell areas come from CAM's Gaussian weights: area = gw / sum(gw) * 4 pi R^2 / nlon, R = 6.37122e6 m, so band
    areas sum to CAM's Earth area and the bands sum exactly to the global total.
  * Bands are built conservatively: each CAM latitude row (edges midway between row centers; sin(edge) is exactly
    -1 + cumsum(gw)) is split between the bands it straddles by area, assuming its zonal value is uniform within
    the row. Each band's area is then exactly the spherical band area 2 pi R^2 (sin(top) - sin(bottom)). Binning
    whole rows by their center latitude (as ohc_cases.latitude_band_sums does on the much finer POP grid) would give
    5 rows to some 5-degree bands and 6 to others at CAM's 0.94-degree spacing, making band areas and totals -8% to
    +13% wrong. Zonal means are over the whole band (land and ocean), unlike the ocean-only OHC zonal means.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

CONDITIONS = {  # display name (as in zonal_means.ipynb): (data root, experiment = case name without member)
    "SSP2-4.5": ("/glade/campaign/cesm/collections/CESM2-WACCM-SSP245/",
                 "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM"),
    "ARISE-1.5": ("/glade/campaign/cesm/collections/ARISE-SAI-1.5/",
                  "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT"),
    "ARISE-1.0": ("/glade/work/jonahshaw/PRISM_data/ARISE-1.0/",
                  "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5"),
}
MEMBERS = [f"{i:03d}" for i in range(1, 11)]  # the members ohc_cases.load_experiments keeps for all three conditions
VARS = {"ASR": "FSNT", "OLR": "FLNT"}

CAM_EARTH_RADIUS_M = 6.37122e6
EARTH_AREA_M2 = 4.0 * np.pi * CAM_EARTH_RADIUS_M**2
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)  # noleap
SECONDS_PER_YEAR = DAYS_IN_MONTH.sum() * 86400.0
DEFAULT_OUT = "/glade/work/wkamp/GEO/PRISM/PRISM/OHC_levels/data/zonal_toa_fluxes_annual_{start}-{end}.nc"


def case_files(condition, member, var):
    root, experiment = CONDITIONS[condition]
    case = f"{experiment}.{member}"
    pattern = os.path.join(root, case, "atm", "proc", "tseries", "month_1", f"{case}.cam.h0.{var}.*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match {pattern}")
    return case, files


def row_areas_m2(gw, nlon):
    """Area of one grid cell in each CAM latitude row [m^2]."""
    return gw / gw.sum() * EARTH_AREA_M2 / nlon


def monthly_row_totals(case, files, var, start, end):
    """Monthly totals of `var` over each CAM latitude row [W] for years start..end, as (months, lat) with months
    ordered Jan start .. Dec end. Also returns lat and gw (checked identical across files)."""
    keys, rows, lat, gw = [], [], None, None
    for f in files:
        with xr.open_dataset(f, use_cftime=True, decode_timedelta=False) as ds:
            lo, hi = ds["time_bnds"].values[:, 0], ds["time_bnds"].values[:, 1]
            months = np.array([t.year * 12 + t.month - 1 for t in lo])
            widths = np.array([b.year * 12 + b.month - 1 for b in hi]) - months
            if not (all(t.day == 1 for t in lo) and np.all(widths == 1)):
                raise ValueError(f"{f}: time_bnds are not single calendar months")
            keep = (months >= start * 12) & (months <= end * 12 + 11)
            if not keep.any():
                continue
            if lat is None:
                lat, gw = ds["lat"].values, ds["gw"].values
            elif not (np.array_equal(ds["lat"].values, lat) and np.allclose(ds["gw"].values, gw)):
                raise ValueError(f"{f}: lat/gw differ from the case's other files")
            field = ds[var].isel(time=np.flatnonzero(keep)).transpose("time", "lat", "lon").values.astype(np.float64)
            if np.isnan(field).any():
                raise ValueError(f"{f}: {var} has missing values")
            rows.append(field.sum(axis=2) * row_areas_m2(gw, field.shape[2]))
            keys.append(months[keep])
    if not rows:
        raise ValueError(f"{case}: no {var} data in {start}-{end}")
    keys, rows = np.concatenate(keys), np.concatenate(rows)
    order = np.argsort(keys, kind="stable")
    keys, rows = keys[order], rows[order]
    repeats = np.flatnonzero(np.diff(keys) == 0)
    if repeats.size and not np.allclose(rows[repeats], rows[repeats + 1], rtol=1e-6, atol=0):
        raise ValueError(f"{case}: files repeat months with different {var} values")
    first = np.r_[True, np.diff(keys) != 0]
    keys, rows = keys[first], rows[first]
    expected = np.arange(start * 12, end * 12 + 12)
    if not np.array_equal(keys, expected):
        missing = sorted(set(expected) - set(keys))
        raise ValueError(f"{case}: {var} does not cover every month of {start}-{end} "
                         f"(missing {len(missing)}, first {divmod(missing[0], 12) if missing else None})")
    return rows, lat, gw


def annual_mean(monthly):
    """(months, ...) with whole years starting in January -> month-length-weighted annual means (years, ...)."""
    by_year = monthly.reshape(-1, 12, *monthly.shape[1:])
    w = DAYS_IN_MONTH.reshape(1, 12, *([1] * (monthly.ndim - 1)))
    return (by_year * w).sum(axis=1) / DAYS_IN_MONTH.sum()


def band_weights(lat, gw, edges):
    """(n_bands, nlat) fraction of each CAM row's area lying in each band. Row edges in sin(latitude) come from the
    Gaussian weights (gw is the sin(lat) width of each row), and area on the sphere is linear in sin(latitude), so
    the overlap in sin(latitude) is the exact area fraction. Every column sums to 1."""
    s_row = -1.0 + np.concatenate([[0.0], np.cumsum(gw / gw.sum() * 2.0)])
    if not np.allclose(np.sin(np.deg2rad(lat)), 0.5 * (s_row[:-1] + s_row[1:]), atol=0.02):
        raise ValueError("gw-derived row edges don't bracket the row latitudes")
    s_band = np.sin(np.deg2rad(edges))
    lo = np.maximum(s_row[None, :-1], s_band[:-1, None])
    hi = np.minimum(s_row[None, 1:], s_band[1:, None])
    w = np.clip(hi - lo, 0.0, None) / np.diff(s_row)[None, :]
    if not np.allclose(w.sum(axis=0), 1.0, rtol=0, atol=1e-12):
        raise ValueError("band weights don't partition every row")
    return w


def to_bands(row_values, weights):
    """Row totals (..., lat) -> band totals (..., n_bands), splitting straddling rows by area."""
    return row_values @ weights.T


def process_member(job):
    condition, member, start, end, edges = job
    out, lat_ref, gw_ref = {}, None, None
    for name, var in VARS.items():
        case, files = case_files(condition, member, var)
        rows, lat, gw = monthly_row_totals(case, files, var, start, end)
        if lat_ref is None:
            lat_ref, gw_ref = lat, gw
        elif not (np.array_equal(lat, lat_ref) and np.allclose(gw, gw_ref)):
            raise ValueError(f"{case}: FSNT and FLNT grids differ")
        out[name] = annual_mean(rows)  # (year, lat) [W]
    weights = band_weights(lat_ref, gw_ref, edges)
    banded = {name: to_bands(v, weights) for name, v in out.items()}
    # Bands must sum to the global total computed on the native rows.
    for name, v in out.items():
        if not np.allclose(banded[name].sum(axis=-1), v.sum(axis=-1), rtol=1e-12):
            raise ValueError(f"{condition} {member}: {name} bands don't sum to the global total")
    print(f"  {condition} {member}: global-mean EEI {start}-{end} = "
          f"{(banded['ASR'] - banded['OLR']).sum(axis=-1).mean() / EARTH_AREA_M2:+.3f} W m-2", flush=True)
    return condition, member, banded, lat_ref, gw_ref


def main(start, end, band_width, out_path, workers):
    edges = np.arange(-90.0, 90.0 + band_width / 2, band_width)
    if not np.isclose(edges[-1], 90.0):
        raise ValueError(f"band width {band_width} doesn't divide 180")
    jobs = [(c, m, start, end, edges) for c in CONDITIONS for m in MEMBERS]
    print(f"Processing {len(jobs)} members with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_member, jobs))

    lat, gw = results[0][3], results[0][4]
    for c, m, _, lat_i, gw_i in results:
        if not (np.array_equal(lat_i, lat) and np.allclose(gw_i, gw)):
            raise ValueError(f"{c} {m}: CAM grid differs from the other cases")
    n_bands = edges.size - 1
    band_area = to_bands(row_areas_m2(gw, 1), band_weights(lat, gw, edges))  # nlon=1: area of each whole row
    exact_area = 0.5 * EARTH_AREA_M2 * np.diff(np.sin(np.deg2rad(edges)))
    if not np.allclose(band_area, exact_area, rtol=1e-9):
        raise ValueError("band areas differ from the exact spherical band areas")

    conds, years = list(CONDITIONS), np.arange(start, end + 1)
    shape = (len(conds), len(MEMBERS), years.size, n_bands)
    total = {name: np.full(shape, np.nan) for name in VARS}
    for c, m, banded, _, _ in results:
        for name, v in banded.items():
            total[name][conds.index(c), MEMBERS.index(m)] = v
    total["EEI"] = total["ASR"] - total["OLR"]

    dims = ("condition", "member", "year", "lat")
    coords = {
        "condition": conds, "member": MEMBERS, "year": years, "lat": 0.5 * (edges[:-1] + edges[1:]),
        "lat_bnds": (("lat", "bnds"), np.stack([edges[:-1], edges[1:]], axis=-1)),
        "band_area_m2": ("lat", band_area, dict(units="m2", long_name="Area of the latitude band (land + ocean)")),
    }
    long = {"ASR": "absorbed shortwave (FSNT)", "OLR": "outgoing longwave (FLNT, upward positive)",
            "EEI": "Earth energy imbalance (FSNT - FLNT, positive into the system)"}
    data_vars = {}
    for name in ("ASR", "OLR", "EEI"):
        data_vars[f"{name}_total"] = (dims, total[name], dict(
            units="W", long_name=f"Annual-mean {long[name]}, band total"))
        data_vars[f"{name}_mean"] = (dims, total[name] / band_area, dict(
            units="W m-2", long_name=f"Annual-mean {long[name]}, band area-weighted mean"))
    data_vars["iEEI_total"] = (dims, total["EEI"] * SECONDS_PER_YEAR, dict(
        units="J", long_name="Energy gained by the band during the year (EEI integrated over the year), band total"))
    data_vars["iEEI_mean"] = (dims, total["EEI"] * SECONDS_PER_YEAR / band_area, dict(
        units="J m-2", long_name="Energy gained during the year per m2 of band (EEI integrated over the year)"))

    ds = xr.Dataset(data_vars, coords=coords)
    ds["lat"].attrs = dict(units="degrees_north", long_name="latitude band center", bounds="lat_bnds")
    ds["year"].attrs = dict(long_name="calendar year (noleap)")
    ds.attrs = dict(
        title="Annual zonal TOA radiative fluxes in latitude bands",
        cases="; ".join(f"{c}: {e}.<member> under {r}" for c, (r, e) in CONDITIONS.items()),
        lat_edges=edges,
        banding=("conservative: CAM rows split between bands by area overlap (uniform value within a row assumed); "
                 "band_area_m2 equals the exact spherical band area"),
        earth_area_m2=EARTH_AREA_M2,
        seconds_per_year=SECONDS_PER_YEAR,
        period_reduction=("iEEI_*: sum over year for the period total; ASR/OLR/EEI_*: mean over year for the period "
                          "mean. Sum over lat of *_total gives the global total."),
        created_by=os.path.abspath(__file__),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write a temporary file and rename it over the old one. Overwriting in place would truncate a file that an open
    # notebook kernel may still hold; HDF5 then keeps serving that kernel the old cached contents even on a reload.
    tmp_path = f"{out_path}.tmp{os.getpid()}"
    ds.to_netcdf(tmp_path)
    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=2035)
    parser.add_argument("--end", type=int, default=2069)
    parser.add_argument("--band-width", type=float, default=5.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("NCPUS", 1)))
    args = parser.parse_args()
    out = args.out or DEFAULT_OUT.format(start=args.start, end=args.end)
    main(args.start, args.end, args.band_width, out, args.workers)
