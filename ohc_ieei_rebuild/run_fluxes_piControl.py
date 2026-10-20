"""Compute global heat-budget flux terms for the 1850 piControl simulation.

Usage:
    python run_fluxes_piControl.py [--start-year YEAR] [--end-year YEAR]

Reads monthly POP2, CICE, CLM and CAM time series for the piControl case
(see config.py), computes every term in energy_fluxes.py as a global-total
time series, and writes one NetCDF file under config.OUTPUT_ROOT. Year
filtering uses the shifted (true-calendar-month) time axis, matching
run_ohc_piControl.py and run_ieei_piControl.py.
"""
import argparse
import glob
import os
import re

import xarray as xr

import config
from energy_fluxes import (
    EARTH_AREA_M2,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_VAPOR,
    RHO_ICE,
    SOIL_HEAT_CAPACITY,
    compute_atmosphere_terms,
    compute_land_terms,
    compute_ocean_terms,
    compute_seaice_terms,
)
from ohc import shift_noleap_time_back_one_month

MFDATASET_KWARGS = dict(combine="by_coords", data_vars="minimal", coords="minimal", compat="override")

# CESM time-series file names end in .YYYYMM-YYYYMM.nc, the calendar months the file covers.
FILE_RANGE_RE = re.compile(r"\.(\d{4})\d{2}-(\d{4})\d{2}\.nc$")


def files_overlapping_years(files, start_year, end_year):
    keep = []
    for f in files:
        m = FILE_RANGE_RE.search(f)
        if m is None:
            raise ValueError(f"Can't parse the year range from file name {f}")
        first, last = int(m.group(1)), int(m.group(2))
        if (start_year is None or last >= start_year) and (end_year is None or first <= end_year):
            keep.append(f)
    return keep


def make_opener(tseries_dir, stream, start_year=None, end_year=None, max_files=None, n_months=None):
    """Return open_var(var) -> time-shifted, year-filtered Dataset for one
    model component. max_files/n_months only exist for the smoke test."""
    def open_var(var):
        pattern = os.path.join(tseries_dir, f"{config.CASE_STR}.{stream}.{var}.*.nc")
        files = files_overlapping_years(sorted(glob.glob(pattern)), start_year, end_year)[:max_files]
        if not files:
            raise FileNotFoundError(f"No files found at {pattern} covering years {start_year}-{end_year}")
        ds = xr.open_mfdataset(files, chunks={}, decode_timedelta=True, **MFDATASET_KWARGS)
        ds = shift_noleap_time_back_one_month(ds)
        if start_year is not None:
            ds = ds.isel(time=(ds["time"].dt.year >= start_year).values)
        if end_year is not None:
            ds = ds.isel(time=(ds["time"].dt.year <= end_year).values)
        if n_months is not None:
            ds = ds.isel(time=slice(0, n_months))
        return ds
    return open_var


def compute_all_terms(**opener_kwargs):
    open_pop = make_opener(config.OCN_TSERIES_DIR, "pop.h", **opener_kwargs)
    terms = {}
    terms.update(compute_ocean_terms(open_pop))
    terms.update(compute_seaice_terms(make_opener(config.ICE_TSERIES_DIR, "cice.h", **opener_kwargs)))
    terms.update(compute_land_terms(make_opener(config.LND_TSERIES_DIR, "clm2.h0", **opener_kwargs)))
    terms.update(compute_atmosphere_terms(make_opener(config.ATM_TSERIES_DIR, "cam.h0", **opener_kwargs)))
    out = xr.Dataset(terms)

    grid = open_pop("SHF")
    out.attrs["ocean_area_m2"] = float(((grid["TAREA"] / 1e4).where(grid["KMT"] > 0, 0.0)).sum().values)
    out.attrs["earth_area_m2"] = EARTH_AREA_M2
    out.attrs["latent_heat_vapor_j_kg"] = LATENT_HEAT_VAPOR
    out.attrs["latent_heat_fusion_j_kg"] = LATENT_HEAT_FUSION
    out.attrs["rho_ice_kg_m3"] = RHO_ICE
    out.attrs["soil_heat_capacity_j_m3_k"] = SOIL_HEAT_CAPACITY
    out.attrs["sign_convention"] = (
        "Flux terms [W] are global totals, positive = energy gained by the receiving component "
        "(ocean for POP terms, system for TOA terms, surface for SFC_NET); CAM FLNT/FLUT/FLNS/LHFLX/SHFLX "
        "keep their native upward-positive sign. Divide by earth_area_m2 for W per m^2 of Earth."
    )
    return out


def main(start_year=None, end_year=None):
    out = compute_all_terms(start_year=start_year, end_year=end_year)
    out.attrs["case"] = config.CASE_STR
    if start_year is not None:
        out.attrs["start_year"] = start_year
    if end_year is not None:
        out.attrs["end_year"] = end_year

    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    suffix = f".y{start_year or 1}-{end_year}" if end_year is not None else ""
    out_path = os.path.join(config.OUTPUT_ROOT, f"{config.CASE_STR}.fluxes{suffix}.nc")
    out.to_netcdf(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)
