"""Compute integrated EEI for the 1850 piControl simulation on NCAR Glade.

Usage:
    python run_ieei_piControl.py [--start-year YEAR]

Reads CAM monthly FSNT/FLNT output for the piControl case (see config.py),
computes the area-weighted global-mean EEI and its running time integral
(iEEI) via ieei.compute_ieei, and writes a single NetCDF output file under
config.OUTPUT_ROOT.
"""
import argparse
import glob
import os

import xarray as xr

import config
from ieei import compute_eei, compute_ieei, global_mean, total_joules


def find_files(var_glob):
    pattern = os.path.join(config.ATM_TSERIES_DIR, var_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No piControl files found at {pattern}")
    return files


def main(start_year=None):
    fsnt_files = find_files(config.FSNT_FILE_GLOB)
    flnt_files = find_files(config.FLNT_FILE_GLOB)

    fsnt_ds = xr.open_mfdataset(fsnt_files, combine="by_coords")
    flnt_ds = xr.open_mfdataset(flnt_files, combine="by_coords")

    asr_global = global_mean(fsnt_ds[config.ASR_VAR], fsnt_ds)
    olr_global = global_mean(flnt_ds[config.OLR_VAR], flnt_ds)

    start_time = f"{start_year}-01-01" if start_year is not None else None
    eei = compute_eei(asr_global, olr_global)
    ieei = compute_ieei(eei, start_time=start_time)
    ieei_total_j = total_joules(ieei)

    out = xr.Dataset(
        {
            "EEI_global_mean": eei,
            "iEEI_global_mean": ieei,
            "iEEI_total_joules": ieei_total_j,
        }
    )
    out.attrs["case"] = config.CASE_STR
    out.attrs["asr_variable"] = config.ASR_VAR
    out.attrs["olr_variable"] = config.OLR_VAR
    if start_time is not None:
        out.attrs["ieei_start_time"] = start_time

    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_ROOT, f"{config.CASE_STR}.iEEI.nc")
    out.to_netcdf(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year)
