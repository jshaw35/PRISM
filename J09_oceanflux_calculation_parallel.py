#!/usr/bin/env python3
"""
CESM2 Ocean Heat flux claculator

Computes global ocean heat content from CESM2/POP2 monthly output using constant density and heat capacity (per CESM examples below).

https://ncar.github.io/CESM-Tutorial/notebooks/diagnostics/pop/advanced_pop.html
https://ncar.github.io/osdf-examples/jetstream-cesm-oceanheat/

Author: Jonah Shaw
Date: 2026-07-01
"""
# %%
import os
import logging
import numpy as np
import xarray as xr
from pathlib import Path
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
warnings.filterwarnings('ignore', category=DeprecationWarning)

ancillary_dir = "/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/ancillary_files/"
ancillary_file = os.path.join(ancillary_dir, "ohc_ancillary_data.nc")

# %%
def crawl_and_process(
    input_dir,
    output_dir,
    match_pattern: str,
    process_fn,
    process_path_fn: callable = None,
    **fn_args,
):
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        if output_dir is not None:
            out_root = output_dir if rel_root == "." else os.path.join(output_dir, rel_root)
        for name in files:
            src = os.path.join(root, name)
            if Path(src).match("**/" + match_pattern + "**.QFLUX.*.nc"):
                dst = os.path.join(out_root, name)
                dst = process_path_fn(dst) if process_path_fn is not None else dst
                if os.path.exists(dst):
                    logging.info(f"{dst} already exists")
                    continue
                process_fn(src, dst, **fn_args)


def submit_oceanflux_job(filepath, savepath):
    """
    Call an existing bash script and supply filepath and savepath as arguments.

    Args:
        filepath (_type_): _description_
        savepath (_type_): _description_
    """

    os.system(f"qsub -v INPUT_FILE={filepath},OUTPUT_FILE={savepath} /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_scripts/compute_oceanflux_single.sh")


def compute_oceanflux_single_file(filepath, savepath):

    # Get the correspond SHF file path
    logging.info(f"Starting ocean flux computation for to {filepath}")
    shf_filepath = filepath.replace("QFLUX", "SHF")
    assert os.path.exists(shf_filepath), f"SHF file not found for {filepath}"

    ds_merged = xr.open_mfdataset([filepath, shf_filepath], chunks={"time":1}, decode_timedelta=True, combine="by_coords")

    # Check that the needed variables are a subset of the merged dataset variables
    required_vars = ["QFLUX", "SHF", "dz", "TAREA", "KMT"]
    required_coords = ["z_t", "z_w_bot"]
    missing_vars = [var for var in required_vars if var not in ds_merged.data_vars]
    missing_coords = [coord for coord in required_coords if coord not in ds_merged.coords]
    if missing_vars or missing_coords:
        ancillary_ds = xr.open_dataset(ancillary_file)
        ancillary_subset = ancillary_ds[missing_vars + missing_coords].drop_attrs()
        ds_merged = xr.merge([ds_merged, ancillary_subset], combine_attrs="override", compat='override')
    ds_merged = ds_merged[required_vars + required_coords]

    # Correct CESM time if necessary
    if ds_merged["time"][0]["time.month"] == 2:
        ds_merged = ds_merged.assign_coords(
            time=shift_noleap_time_back_one_month(ds_merged["time"].values)
        )
    oceanflux_ds = compute_oceanflux(ds_merged)
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    logging.info(f"Saving ocean flux data to {savepath}")
    oceanflux_ds.to_netcdf(savepath)


def compute_oceanflux(
    ds,
):
    """
    Compute the heat flux into the ocean (sum of SHF and QFLUX).
    """
    
    # Convert TAREA from cm² to m²
    tarea_m2 = ds['TAREA'] / 1e4  # [m²]

    # Ocean mask (0=land, >0=ocean)
    kmt = ds['KMT']
    kmt_mask = kmt > 0  # True for ocean points, False for land points

    oceanflux_ds = (ds['QFLUX'] + ds['SHF']).where(kmt_mask)  # [W/m²] JKS assuming sign conventions are consistent here.

    global_mean_oceanflux = oceanflux_ds.weighted(tarea_m2).mean(dim=['nlat', 'nlon'], skipna=True)  # [W/m²]
    global_mean_oceanflux.name = "OHF_global_mean"

    # Compute global and ocean surface areas to include in metadata
    global_area = (tarea_m2).sum(["nlat", "nlon"])
    ocean_area = (tarea_m2.where(ds["KMT"]>0)).sum(["nlat", "nlon"])

    oceanflux_ds.attrs["long_name"] = "Total Ocean Heat Flux per unit area"
    oceanflux_ds.attrs["units"] = "W/m²"
    oceanflux_ds.name = "OHF"

    ohf_ds = xr.merge([oceanflux_ds, global_mean_oceanflux])
    ohf_ds.attrs["description"] = "Total Ocean Heat Flux calculated from CESM2/POP2 output (SHF + QFLUX). The global mean Ocean Heat Flux is also provided as a separate variable."
    ohf_ds.attrs["global_area_m2"] = global_area.values
    ohf_ds.attrs["ocean_area_m2"] = ocean_area.values

    return ohf_ds.compute()


def shift_noleap_time_back_one_month(time_values):
    t = np.asarray(time_values)
    n = t.size

    years = np.fromiter((v.year for v in t), dtype=np.int32, count=n)
    months = np.fromiter((v.month for v in t), dtype=np.int16, count=n)
    days = np.fromiter((v.day for v in t), dtype=np.int16, count=n)
    hours = np.fromiter((v.hour for v in t), dtype=np.int16, count=n)
    minutes = np.fromiter((v.minute for v in t), dtype=np.int16, count=n)
    seconds = np.fromiter((v.second for v in t), dtype=np.int16, count=n)
    microseconds = np.fromiter((v.microsecond for v in t), dtype=np.int32, count=n)

    months = months - 1
    jan_mask = months == 0
    months[jan_mask] = 12
    years[jan_mask] = years[jan_mask] - 1

    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int16)
    days = np.minimum(days, days_in_month[months - 1])

    dt_type = type(t[0])
    return np.array(
        [
            dt_type(int(y), int(m), int(d), int(h), int(mi), int(s), int(us))
            for y, m, d, h, mi, s, us in zip(years, months, days, hours, minutes, seconds, microseconds)
        ],
        dtype=object,
    )


def get_weights_by_month2(
    time_ds,
    account_for_leap: bool = False,
):

    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_per_month_leap = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    seconds_per_month = 60 * 60 * 24 * days_per_month
    seconds_per_month_leap = 60 * 60 * 24 * days_per_month_leap

    weights = xr.DataArray(
        data=seconds_per_month,
        dims=["month"],
        coords={
            "month": np.arange(1,13),
        }
    )
    weights_leap = xr.DataArray(
        data=seconds_per_month_leap,
        dims=["month"],
        coords={
            "month": np.arange(1,13),
        }
    )

    month_values = time_ds.dt.month
    year_values = time_ds.dt.year

    # Vectorized selection of the appropriate weights for each time point
    if account_for_leap:
        weights = xr.where((year_values % 4 == 0), weights_leap.sel(month=month_values), weights.sel(month=month_values))
    else:
        weights = weights.sel(month=month_values)
    return weights

# %%
testing = False

if testing:
    import glob
    ohc_files = glob.glob("/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/CESM2_LME/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/ocn/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.pop.h.OHC.??????-??????.nc")
    # ohc_files = glob.glob("/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/CESM2_LME/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/ocn/proc/tseries/month_1/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.pop.h.OHC.??????-??????.nc")
    ohc_ds = xr.open_mfdataset(ohc_files, chunks={"time": 1}, preprocess=lambda ds: ds[["OHC_global_mean"]].sel(ohc_depth=-1)).compute()
    ohf_files = glob.glob("/glade/work/jonahshaw/PRISM_data/spatial_oceanflux_data/CESM2_LME/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/ocn/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.pop.h.OHF.??????-??????.nc")
    ohf_files.sort()
    ohf_ds = xr.open_mfdataset(ohf_files, chunks={"time": 1}, preprocess=lambda ds: ds[["OHF_global_mean"]]).compute()

# %%
if testing:
    test_ohc_zeroed = ohc_ds["OHC_global_mean"]
    test_ohc_zeroed = (test_ohc_zeroed - test_ohc_zeroed.isel(time=0)).compute()

    # Convert units
    month_weights = get_weights_by_month2(ohf_ds["time"], account_for_leap=False)
    test_ohf_weighted = ohf_ds["OHF_global_mean"] * month_weights # [W/m²] * [s] = [J/m²]
    test_ohf_unweighted = ohf_ds["OHF_global_mean"] * 365 * 24 * 60 * 60 / 12 # [W/m²] * [s] = [J/m²]

# %%
if testing:
    import matplotlib.pyplot as plt
    plt.figure()
    test_ohf_weighted.cumsum(dim="time").plot(label="Integrated OHF with monthly-weighting")
    test_ohf_unweighted.cumsum(dim="time").plot(label="Integrated OHF no monthly-weighting")
    test_ohc_zeroed.plot(label="OHC from CESM2", linestyle="--", alpha=0.5, linewidth=0.5)
    plt.legend()

    plt.figure()
    test_ohf_weighted.cumsum(dim="time").groupby("time.year").mean().plot(label="Integrated OHF with monthly-weighting")
    test_ohf_unweighted.cumsum(dim="time").groupby("time.year").mean().plot(label="Integrated OHF no monthly-weighting")
    test_ohc_zeroed.groupby("time.year").mean().plot(label="OHC from CESM2", linestyle="--", alpha=0.5, linewidth=0.5)
    plt.legend()

    plt.figure()
    (test_ohf_weighted.cumsum(dim="time") - test_ohc_zeroed).plot(label="Weighted OHF - OHC")
    (test_ohf_unweighted.cumsum(dim="time") - test_ohc_zeroed).plot(label="Unweighted OHF - OHC")
    plt.legend()

    plt.figure()
    (test_ohf_weighted.cumsum(dim="time") - test_ohc_zeroed).groupby("time.year").mean().plot(label="Weighted OHF minus OHC")
    # (test_ohf_unweighted.cumsum(dim="time") - test_ohc_zeroed).groupby("time.year").mean().plot(label="Unweighted OHF - OHC")
    plt.legend()
    # (test_ohf_weighted.cumsum(dim="time") - test_ohc_zeroed).cumsum(dim="time").plot()
    # test_ohc_zeroed.plot()

    fig, ax = plt.subplots(1, 1)
    (test_ohf_weighted.cumsum(dim="time") / test_ohc_zeroed).plot(label="Weighted OHF / OHC")
    (test_ohf_unweighted.cumsum(dim="time") / test_ohc_zeroed).plot(label="Unweighted OHF / OHC")
    ax.set_ylim(0.8, 1.8)
    ax.hlines(1, ax.get_xlim()[0], ax.get_xlim()[1], colors="k", linestyles="dashed")
    plt.legend()

    fig, ax = plt.subplots(1, 1)
    (test_ohf_weighted.cumsum(dim="time") / test_ohc_zeroed).groupby("time.year").mean().plot(label="Weighted OHF / OHC")
    (test_ohf_unweighted.cumsum(dim="time") / test_ohc_zeroed).groupby("time.year").mean().plot(label="Unweighted OHF / OHC")
    ax.set_ylim(0.8, 1.8)
    ax.hlines(1, ax.get_xlim()[0], ax.get_xlim()[1], colors="k", linestyles="dashed")
    plt.legend()


# %%
if testing:

    test_qflux_filepath = "/gdex/data/d651078/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/ocn/proc/tseries/month_1/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.pop.h.QFLUX.180001-184912.nc"
    test_ohc_filepath = "/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/CESM2_LME/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/ocn/proc/tseries/month_1/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.pop.h.OHC.180001-184912.nc"
    test_ohc = xr.open_dataset(test_ohc_filepath, chunks={"time": 1})
    test_ohf = compute_oceanflux_single_file(test_qflux_filepath, "/glade/work/jonahshaw/PRISM_data/spatial_oceanflux_data/")

    test_ohc_zeroed = test_ohc["OHC_global_mean"].sel(ohc_depth=-1)
    test_ohc_zeroed = (test_ohc_zeroed - test_ohc_zeroed.isel(time=0)).compute()

    # Convert units
    month_weights = get_weights_by_month2(test_ohf["time"], account_for_leap=False)
    test_ohf_weighted = test_ohf["OHF_global_mean"] * month_weights # [W/m²] * [s] = [J/m²]
    test_ohf_unweighted = test_ohf["OHF_global_mean"] * 365 * 24 * 60 * 60 / 12 # [W/m²] * [s] = [J/m²]

# %%
if testing:
    import matplotlib.pyplot as plt
    plt.figure()
    test_ohf_weighted.cumsum(dim="time").plot(label="Integrated OHF with monthly-weighting")
    test_ohf_unweighted.cumsum(dim="time").plot(label="Integrated OHF no monthly-weighting")
    test_ohc_zeroed.plot(label="OHC from CESM2")
    plt.legend()

    plt.figure()
    (test_ohf_weighted.cumsum(dim="time") - test_ohc_zeroed).plot(label="Weighted OHF - OHC")
    (test_ohf_unweighted.cumsum(dim="time") - test_ohc_zeroed).plot(label="Unweighted OHF - OHC")
    plt.legend()
    # (test_ohf_weighted.cumsum(dim="time") - test_ohc_zeroed).cumsum(dim="time").plot()
    # test_ohc_zeroed.plot()

    fig, ax = plt.subplots(1, 1)
    (test_ohf_weighted.cumsum(dim="time") / test_ohc_zeroed).plot(label="Weighted OHF / OHC")
    (test_ohf_unweighted.cumsum(dim="time") / test_ohc_zeroed).plot(label="Unweighted OHF / OHC")
    ax.set_ylim(0.5, 1.5)
    plt.legend()

# %%
if __name__ == "__main__":
    # If on glade
    CASE_CONFIG = {
        "CESM2_LME": {
            "data_dir": "/gdex/data/d651078",
            "file_patterns": [
                "b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/ocn/proc/tseries/month_1/",
                "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/ocn/proc/tseries/month_1/",
            ]
        },
        "CESM2_WACCM_HIST": {
            "data_dir": "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/",
            "file_patterns": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/ocn/proc/tseries/month_1/"]
        },
        "CESM2_WACCM_1850control": {
            "data_dir": "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.BW1850.f09_g17.CMIP6-piControl.001",
            "file_patterns": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1/"]
        },
        "CESM2_WACCM_SSP2-4.5": {
            "data_dir": "/gdex/data/d651045/CESM2-WACCM-SSP245",
            "file_patterns": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?/ocn/proc/tseries/month_1/"]
        },
        "ARISE_SAI": {
            "data_dir": "/gdex/data/d651059/ARISE-SAI-1.5",
            "file_patterns": [
                "b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI.00?/ocn/proc/tseries/month_1/",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?/ocn/proc/tseries/month_1/",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?/ocn/proc/tseries/month_1/",
            ]
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "data_dir": "/gdex/data/d314006",
            "file_patterns": [
                "CMIP6-MCB-???PCT/ocn/month_1/",
                "MCB-050PCT-ensm*/ocn/month_1/",
                "Baseline/ocn/month_1/",
            ]
        },
        # Add more cases as needed
    }
    # Normal units are kJ/cm2 (SI units are J/m2). Reasonable values are 0 - 140 kJ/cm2 (0 - 1.4e9 J/m2) (the conversion factor is 1 kJ/cm2 = 1e7 J/m2)

    if not os.path.exists(ancillary_dir):
        os.makedirs(ancillary_dir, exist_ok=True)
        if not os.path.exists(ancillary_file):
            test_file = "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.BW1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h.TEMP.000101-009912.nc"
            test_ds = xr.open_dataset(test_file)
            test_ds[["TAREA", "UAREA", "KMT", "z_w_bot", "dz"]].to_netcdf(ancillary_file)

    for case in CASE_CONFIG:
        rawdata_root = Path(CASE_CONFIG[case]["data_dir"])
        for pattern in CASE_CONFIG[case]["file_patterns"]:
            test_out = crawl_and_process(
                input_dir=rawdata_root,
                output_dir=f"/glade/work/jonahshaw/PRISM_data/spatial_oceanflux_data/{case}/",
                match_pattern=pattern,
                process_fn=submit_oceanflux_job,
                process_path_fn=lambda p: p.replace("QFLUX", "OHF")
            )

    # %%
