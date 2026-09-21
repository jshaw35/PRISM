"""Minimal smoke test for the OHC/iEEI pipelines.

Not a substitute for the full 100-year runs -- this only checks that the
input files resolve, load, and that every function in the OHC and iEEI
pipelines runs end-to-end without erroring, on a tiny time slice (the first
few months of the earliest piControl file for each variable). It also
sanity-checks that the outputs aren't all-NaN/non-finite and exercises the
NetCDF write path, then deletes the small files it wrote.

Usage:
    python test_pipeline_smoke.py [--n-months N]

Exits non-zero (via an uncaught exception) on any failure, so a PBS job
running this script shows a clear failed/succeeded status.
"""
import argparse
import os
import shutil

import numpy as np
import xarray as xr

import config
from ieei import compute_eei, compute_ieei, global_mean, total_joules
from ohc import compute_ohc, open_ocean_dataset, shift_noleap_time_back_one_month
from run_ieei_piControl import find_files as find_atm_files
from run_ohc_piControl import find_temp_files


def check_finite(name, da):
    values = da.values
    n_finite = np.isfinite(values).sum()
    if n_finite == 0:
        raise AssertionError(f"{name}: all values are non-finite (NaN/inf) -- pipeline likely broken")
    print(f"  {name}: shape={values.shape}, {n_finite}/{values.size} finite, "
          f"min={np.nanmin(values):.4g}, max={np.nanmax(values):.4g}")


def test_ohc(n_months):
    print("--- OHC pipeline ---")
    temp_files = find_temp_files()
    print(f"  found {len(temp_files)} TEMP file(s); using first file only for the smoke test")
    ds = open_ocean_dataset([temp_files[0]], grid_path=config.OCN_GRID_FILE)
    ds = shift_noleap_time_back_one_month(ds)
    ds = ds.isel(time=slice(0, n_months))
    print(f"  loaded {ds.sizes['time']} month(s): {ds['time'].values[0]} .. {ds['time'].values[-1]}")

    ohc_ds = compute_ohc(ds)
    check_finite("OHC_global_mean", ohc_ds["OHC_global_mean"])
    ocean_frac = ohc_ds.attrs["ocean_area_m2"] / ohc_ds.attrs["global_area_m2"]
    print(f"  ocean_area_m2/global_area_m2 = {ocean_frac:.3f} (expect close to ~0.7)")
    if not (0.5 < ocean_frac < 0.85):
        raise AssertionError(f"ocean fraction {ocean_frac:.3f} is outside a plausible range -- masking/unit bug")

    out_dir = os.path.join(config.OUTPUT_ROOT, "smoke_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke_test.OHC.nc")
    ohc_ds.to_netcdf(out_path)
    reloaded = xr.open_dataset(out_path)
    assert "OHC_global_mean" in reloaded, "OHC_global_mean missing after round-trip to NetCDF"
    reloaded.close()
    os.remove(out_path)
    print("  NetCDF write/read round-trip OK")


def test_ieei(n_months):
    print("--- iEEI pipeline ---")
    fsnt_files = find_atm_files(config.FSNT_FILE_GLOB)
    flnt_files = find_atm_files(config.FLNT_FILE_GLOB)
    print(f"  found {len(fsnt_files)} FSNT file(s), {len(flnt_files)} FLNT file(s); using first file of each")

    fsnt_ds = xr.open_dataset(fsnt_files[0])
    flnt_ds = xr.open_dataset(flnt_files[0])
    fsnt_ds = shift_noleap_time_back_one_month(fsnt_ds).isel(time=slice(0, n_months))
    flnt_ds = shift_noleap_time_back_one_month(flnt_ds).isel(time=slice(0, n_months))
    print(f"  loaded {fsnt_ds.sizes['time']} month(s): {fsnt_ds['time'].values[0]} .. {fsnt_ds['time'].values[-1]}")

    asr_global = global_mean(fsnt_ds[config.ASR_VAR], fsnt_ds)
    olr_global = global_mean(flnt_ds[config.OLR_VAR], flnt_ds)
    eei = compute_eei(asr_global, olr_global)
    check_finite("EEI_global_mean", eei)

    ieei = compute_ieei(eei)
    check_finite("iEEI_global_mean", ieei)
    ieei_total_j = total_joules(ieei)
    check_finite("iEEI_total_joules", ieei_total_j)

    out = xr.Dataset({"EEI_global_mean": eei, "iEEI_global_mean": ieei, "iEEI_total_joules": ieei_total_j})
    out_dir = os.path.join(config.OUTPUT_ROOT, "smoke_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke_test.iEEI.nc")
    out.to_netcdf(out_path)
    reloaded = xr.open_dataset(out_path)
    assert "iEEI_global_mean" in reloaded, "iEEI_global_mean missing after round-trip to NetCDF"
    reloaded.close()
    os.remove(out_path)
    print("  NetCDF write/read round-trip OK")


def main(n_months):
    test_ohc(n_months)
    test_ieei(n_months)
    smoke_dir = os.path.join(config.OUTPUT_ROOT, "smoke_test")
    if os.path.isdir(smoke_dir) and not os.listdir(smoke_dir):
        shutil.rmtree(smoke_dir)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-months", type=int, default=3, help="Number of leading months to load per file")
    args = parser.parse_args()
    main(n_months=args.n_months)
