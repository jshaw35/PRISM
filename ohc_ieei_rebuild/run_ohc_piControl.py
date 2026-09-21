"""Compute Ocean Heat Content for the 1850 piControl simulation on NCAR Glade.

Usage:
    python run_ohc_piControl.py [--start-year YEAR] [--end-year YEAR]

Reads POP2 monthly TEMP output for the piControl case (see config.py),
computes OHC via ohc.compute_ohc, and writes a single NetCDF output file
under config.OUTPUT_ROOT. Both --start-year and --end-year are inclusive
and are applied to the shifted (true-calendar-month) time axis.
"""
import argparse
import glob
import os

import config
from ohc import compute_ohc, open_ocean_dataset, shift_noleap_time_back_one_month


def find_temp_files():
    pattern = os.path.join(config.OCN_TSERIES_DIR, config.TEMP_FILE_GLOB)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No piControl TEMP files found at {pattern}")
    return files


def main(start_year=None, end_year=None):
    temp_files = find_temp_files()
    ds = open_ocean_dataset(temp_files, grid_path=config.OCN_GRID_FILE, chunks={"time": 24})
    ds = shift_noleap_time_back_one_month(ds)
    if start_year is not None:
        ds = ds.where(ds["time"].dt.year >= start_year, drop=True)
    if end_year is not None:
        ds = ds.where(ds["time"].dt.year <= end_year, drop=True)

    ohc_ds = compute_ohc(ds)
    ohc_ds.attrs["case"] = config.CASE_STR
    ohc_ds.attrs["source_files"] = ", ".join(temp_files)
    if start_year is not None:
        ohc_ds.attrs["start_year"] = start_year
    if end_year is not None:
        ohc_ds.attrs["end_year"] = end_year

    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    suffix = f".y{start_year or 1}-{end_year}" if end_year is not None else ""
    out_path = os.path.join(config.OUTPUT_ROOT, f"{config.CASE_STR}.OHC{suffix}.nc")
    ohc_ds.to_netcdf(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)
