"""Compute integrated EEI for the 1850 piControl simulation on NCAR Glade.

Usage:
    python run_ieei_piControl.py [--start-year YEAR] [--end-year YEAR]

Reads CAM monthly FSNT/FLNT output for the piControl case (see config.py),
computes the area-weighted global-mean EEI and its running time integral
(iEEI) via ieei.compute_ieei, and writes a single NetCDF output file under
config.OUTPUT_ROOT. Both --start-year and --end-year are inclusive and are
applied to the shifted (true-calendar-month) time axis, the same convention
run_ohc_piControl.py uses, so a shared --start-year/--end-year pair selects
the same calendar months from both pipelines.
"""
import argparse
import glob
import os

import xarray as xr

import config
from ieei import compute_eei, compute_ieei, global_mean, total_joules
from ohc import shift_noleap_time_back_one_month


def find_files(var_glob):
    pattern = os.path.join(config.ATM_TSERIES_DIR, var_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No piControl files found at {pattern}")
    return files


def main(start_year=None, end_year=None):
    fsnt_files = find_files(config.FSNT_FILE_GLOB)
    flnt_files = find_files(config.FLNT_FILE_GLOB)

    # data_vars="minimal"/coords="minimal"+compat="override": without this,
    # open_mfdataset's default data_vars="all" broadcasts the time-invariant
    # `gw` (Gaussian latitude weights) across the new time dimension when
    # concatenating 2+ files (see ohc.open_ocean_dataset for the same issue
    # on the ocean side -- it silently corrupted OHC's global_area_m2 there).
    mfdataset_kwargs = dict(combine="by_coords", data_vars="minimal", coords="minimal", compat="override")
    fsnt_ds = xr.open_mfdataset(fsnt_files, **mfdataset_kwargs)
    flnt_ds = xr.open_mfdataset(flnt_files, **mfdataset_kwargs)
    # CAM monthly history files use the same end-of-averaging-period time
    # stamp convention as POP2 (see ohc.shift_noleap_time_back_one_month) --
    # shift both so time.dt.year matches the calendar year the data actually
    # describes, and lines up with run_ohc_piControl.py's year filtering.
    fsnt_ds = shift_noleap_time_back_one_month(fsnt_ds)
    flnt_ds = shift_noleap_time_back_one_month(flnt_ds)
    # isel() with a boolean mask, not where(cond, drop=True): the latter
    # broadcasts every variable against cond's dimension, even ones (gw) that
    # don't have a time dimension at all (see run_ohc_piControl.py, which hit
    # the identical bug on TAREA).
    if end_year is not None:
        fsnt_ds = fsnt_ds.isel(time=(fsnt_ds["time"].dt.year <= end_year).values)
        flnt_ds = flnt_ds.isel(time=(flnt_ds["time"].dt.year <= end_year).values)

    asr_global = global_mean(fsnt_ds[config.ASR_VAR], fsnt_ds)
    olr_global = global_mean(flnt_ds[config.OLR_VAR], flnt_ds)

    # Zero-padded to 4 digits: xarray's cftime ISO-8601 parser (used by
    # compute_ieei's eei.sel(time=slice(start_time, ...))) rejects "1-01-01",
    # requiring "0001-01-01".
    start_time = f"{start_year:04d}-01-01" if start_year is not None else None
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
    if end_year is not None:
        out.attrs["end_year"] = end_year

    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    suffix = f".y{start_year or 1}-{end_year}" if end_year is not None else ""
    out_path = os.path.join(config.OUTPUT_ROOT, f"{config.CASE_STR}.iEEI{suffix}.nc")
    out.to_netcdf(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)
