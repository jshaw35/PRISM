"""Minimal smoke test for the OHC, iEEI and energy-flux pipelines.

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
from run_fluxes_piControl import compute_all_terms
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
    # Use the first 2 files (if available), not just 1: open_mfdataset's
    # default data_vars="all" only broadcasts time-invariant grid variables
    # (TAREA, KMT, ...) across a spurious new time dimension when it actually
    # has to concatenate multiple files -- a single-file smoke test can't
    # catch that class of bug at all.
    use_files = temp_files[:2]
    print(f"  found {len(temp_files)} TEMP file(s); using first {len(use_files)} for the smoke test")
    ds = open_ocean_dataset(use_files, grid_path=config.OCN_GRID_FILE)
    assert "time" not in ds["TAREA"].dims, (
        "TAREA picked up a spurious time dimension from concatenating files -- "
        "open_mfdataset's data_vars='all' default broadcasting is back"
    )
    ds = shift_noleap_time_back_one_month(ds)

    # Exercise the exact --start-year/--end-year filtering run_ohc_piControl.py
    # does (isel with a boolean mask), not just positional slicing below --
    # this is the code path that broke when it used where(cond, drop=True)
    # instead, which (like data_vars="all" above) broadcasts TAREA/KMT across
    # a spurious time dimension.
    first_year = int(ds["time"].dt.year.values[0])
    ds = ds.isel(time=(ds["time"].dt.year >= first_year).values)
    ds = ds.isel(time=(ds["time"].dt.year <= first_year).values)
    assert "time" not in ds["TAREA"].dims, (
        "TAREA picked up a spurious time dimension from year filtering -- "
        "Dataset.where(cond, drop=True) broadcasting is back"
    )

    ds = ds.isel(time=slice(0, n_months))
    print(f"  loaded {ds.sizes['time']} month(s): {ds['time'].values[0]} .. {ds['time'].values[-1]}")

    ohc_ds = compute_ohc(ds)
    # Catches the z_w_bot/z_t dimension-mismatch bug: z_w_bot lives on its own
    # "z_w_bot" dimension (not "z_t") in the source files, so where(mask)
    # silently broadcast a 60x60 outer product instead of aligning layers,
    # leaving a bogus leftover "z_w_bot" axis in every output array (60x the
    # expected data volume -- this is what blew up a 100-year OHC run from an
    # expected few GB to a 141GB file that then ran out of walltime).
    assert ohc_ds["OHC"].dims == ("ohc_depth", "time", "nlat", "nlon"), (
        f"OHC has unexpected dims {ohc_ds['OHC'].dims} -- a leftover vertical "
        "axis (e.g. z_w_bot) likely didn't get reduced"
    )
    assert ohc_ds["OHC_global_mean"].dims == ("ohc_depth", "time"), (
        f"OHC_global_mean has unexpected dims {ohc_ds['OHC_global_mean'].dims} "
        "-- a leftover vertical axis (e.g. z_w_bot) likely didn't get reduced"
    )
    check_finite("OHC_global_mean", ohc_ds["OHC_global_mean"])
    d_ohc = ohc_ds["dOHC_total_joules"]
    assert d_ohc.dims == ("ohc_depth", "time"), f"dOHC_total_joules has unexpected dims {d_ohc.dims}"
    assert np.allclose(d_ohc.isel(time=0).values, 0.0), "dOHC_total_joules should be zero at the first time step"
    check_finite("dOHC_total_joules", d_ohc)
    vol = ohc_ds["ocean_volume_m3"]
    assert vol.dims == ("ohc_depth",), f"ocean_volume_m3 has unexpected dims {vol.dims}"
    full_vol = float(vol.sel(ohc_depth=-1))
    print(f"  ocean volume = {full_vol:.4e} m^3 (expect ~1.3e18); bins: {vol.values}")
    if not (1.1e18 < full_vol < 1.5e18):
        raise AssertionError(f"full-depth ocean volume {full_vol:.3e} m^3 is implausible -- dz/TAREA unit or mask bug")
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
    fsnt_files = find_atm_files(config.FSNT_FILE_GLOB)[:2]
    flnt_files = find_atm_files(config.FLNT_FILE_GLOB)[:2]
    print(f"  using first {len(fsnt_files)} FSNT file(s), {len(flnt_files)} FLNT file(s) for the smoke test")

    # Same data_vars/coords="minimal"+compat="override" open_mfdataset needs
    # in run_ieei_piControl.py to keep `gw` from picking up a spurious time
    # dimension when concatenating multiple files.
    mfdataset_kwargs = dict(combine="by_coords", data_vars="minimal", coords="minimal", compat="override")
    fsnt_ds = xr.open_mfdataset(fsnt_files, **mfdataset_kwargs)
    flnt_ds = xr.open_mfdataset(flnt_files, **mfdataset_kwargs)
    assert "time" not in fsnt_ds["gw"].dims, (
        "gw picked up a spurious time dimension from concatenating files -- "
        "open_mfdataset's data_vars='all' default broadcasting is back"
    )
    fsnt_ds = shift_noleap_time_back_one_month(fsnt_ds)
    flnt_ds = shift_noleap_time_back_one_month(flnt_ds)

    # Exercise the exact --end-year filtering run_ieei_piControl.py does
    # (isel with a boolean mask), matching the OHC-side check above.
    first_year = int(fsnt_ds["time"].dt.year.values[0])
    fsnt_ds = fsnt_ds.isel(time=(fsnt_ds["time"].dt.year <= first_year).values)
    flnt_ds = flnt_ds.isel(time=(flnt_ds["time"].dt.year <= first_year).values)
    assert "time" not in fsnt_ds["gw"].dims, (
        "gw picked up a spurious time dimension from year filtering -- "
        "Dataset.where(cond, drop=True) broadcasting is back"
    )

    fsnt_ds = fsnt_ds.isel(time=slice(0, n_months))
    flnt_ds = flnt_ds.isel(time=slice(0, n_months))
    print(f"  loaded {fsnt_ds.sizes['time']} month(s): {fsnt_ds['time'].values[0]} .. {fsnt_ds['time'].values[-1]}")

    asr_global = global_mean(fsnt_ds[config.ASR_VAR], fsnt_ds)
    olr_global = global_mean(flnt_ds[config.OLR_VAR], flnt_ds)
    eei = compute_eei(asr_global, olr_global)
    check_finite("EEI_global_mean", eei)

    ieei = compute_ieei(eei)
    check_finite("iEEI_global_mean", ieei)
    ieei_total_j = total_joules(ieei)
    check_finite("iEEI_total_joules", ieei_total_j)

    # Exercise the same start_time string the run_*_piControl.py CLIs build
    # from --start-year, including a single-digit year (the run scripts'
    # actual first piControl year), since compute_ieei's cftime parsing
    # rejects a non-zero-padded year like "1-01-01".
    start_year = int(eei["time"].dt.year.values[0])
    start_time = f"{start_year:04d}-01-01"
    ieei_from_start = compute_ieei(eei, start_time=start_time)
    check_finite("iEEI_global_mean (start_time path)", ieei_from_start)

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


def test_fluxes(n_months):
    print("--- energy-flux pipeline ---")
    # Same opener as run_fluxes_piControl.py, limited to the first 2 files per
    # variable (so multi-file concatenation is exercised) and n_months.
    out = compute_all_terms(max_files=2, n_months=n_months).compute()
    for name, da in out.data_vars.items():
        assert da.dims == ("time",), f"{name} has unexpected dims {da.dims} -- a spatial axis didn't get reduced"
        if not np.all(np.isfinite(da.values)):
            raise AssertionError(
                f"{name} has non-finite values -- likely a time-axis mismatch between model components"
            )
    assert out.sizes["time"] == n_months, f"expected {n_months} months, got {out.sizes['time']} (component time axes misaligned?)"
    print(f"  {len(out.data_vars)} terms, {out.sizes['time']} month(s): {out['time'].values[0]} .. {out['time'].values[-1]}")
    earth = out.attrs["earth_area_m2"]
    for name in ("SHF", "SHF_RESIDUAL", "QFLUX", "TFW_T", "RESID_T", "FHOCN_AI", "MELTH_F", "FGR", "TOA_NET_MODEL", "SFC_NET"):
        print(f"  {name:14s} mean = {float(out[name].mean()) / earth:9.4f} W/m^2 (Earth area)")
    print(f"  TS_GLOBAL_MEAN mean = {float(out['TS_GLOBAL_MEAN'].mean()):.2f} K")
    ocean_frac = out.attrs["ocean_area_m2"] / earth
    print(f"  ocean_area/earth_area = {ocean_frac:.3f}")

    out_dir = os.path.join(config.OUTPUT_ROOT, "smoke_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke_test.fluxes.nc")
    out.to_netcdf(out_path)
    reloaded = xr.open_dataset(out_path)
    assert "SHF" in reloaded, "SHF missing after round-trip to NetCDF"
    reloaded.close()
    os.remove(out_path)
    print("  NetCDF write/read round-trip OK")


def main(n_months):
    test_ohc(n_months)
    test_ieei(n_months)
    test_fluxes(n_months)
    smoke_dir = os.path.join(config.OUTPUT_ROOT, "smoke_test")
    if os.path.isdir(smoke_dir) and not os.listdir(smoke_dir):
        shutil.rmtree(smoke_dir)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-months", type=int, default=3, help="Number of leading months to load per file")
    args = parser.parse_args()
    main(n_months=args.n_months)
