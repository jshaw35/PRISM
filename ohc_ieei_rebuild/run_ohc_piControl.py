"""Compute Ocean Heat Content for the 1850 piControl simulation on NCAR Glade.

Usage:
    python run_ohc_piControl.py [--start-year YEAR]

Reads POP2 monthly TEMP output for the piControl case (see config.py),
computes OHC via ohc.compute_ohc, and writes a single NetCDF output file
under config.OUTPUT_ROOT.
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


def main(start_year=None):
    temp_files = find_temp_files()
    ds = open_ocean_dataset(temp_files, grid_path=config.OCN_GRID_FILE, chunks={"time": 24})
    ds = shift_noleap_time_back_one_month(ds)
    if start_year is not None:
        ds = ds.where(ds["time"].dt.year >= start_year, drop=True)

    ohc_ds = compute_ohc(ds)
    ohc_ds.attrs["case"] = config.CASE_STR
    ohc_ds.attrs["source_files"] = ", ".join(temp_files)

    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_ROOT, f"{config.CASE_STR}.OHC.nc")
    ohc_ds.to_netcdf(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year)
