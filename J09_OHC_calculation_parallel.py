#!/usr/bin/env python3
"""
CESM2 Ocean Heat Content (OHC) Calculator

Computes global ocean heat content from CESM2/POP2 monthly output using full
equation of state (pre-calculated density from model).

Author: Drafted by OpenCode and corrected/refined by Jonah Shaw
Date: 2026-04-30
"""
# %%
import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
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
            if Path(src).match("**/" + match_pattern + "**.TEMP.*.nc"):
                dst = os.path.join(out_root, name)
                dst = process_path_fn(dst) if process_path_fn is not None else dst
                if os.path.exists(dst):
                    logging.info(f"{dst} already exists")
                    continue
                process_fn(src, dst, **fn_args)


def submit_OHC_job(filepath, savepath):
    """
    Call an existing bash script and supply filepath and savepath as arguments.

    Args:
        filepath (_type_): _description_
        savepath (_type_): _description_
    """

    # print(f"qsub -v INPUT_FILE={filepath},OUTPUT_FILE={savepath} /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_scripts/compute_ohc_single.sh")
    os.system(f"qsub -v INPUT_FILE={filepath},OUTPUT_FILE={savepath} /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_scripts/compute_ohc_single.sh")

def compute_ohc_single_file(filepath, savepath):
    
    rho_file = filepath.replace(".TEMP.", ".RHO.")
    if not os.path.exists(rho_file):
        logging.error(f"{rho_file} does not exist, cannot compute OHC")
        return

    logging.info(f"Processing {filepath} and {rho_file} to compute OHC, saving to {savepath}")
    ds_merged = xr.open_mfdataset([filepath, rho_file], chunks={"time":1}, decode_timedelta=True, combine="by_coords")
    
    # Check that the needed variables are a subset of the merged dataset variables
    required_vars = ["TEMP", "RHO", "dz", "TAREA", "KMT"]
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
    ohc_ds = compute_OHC(ds_merged)
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    ohc_ds.to_netcdf(savepath)


def compute_OHC(
    ds,
):
    """Compute the ocean heat content.
    OHC = integral of rho * c_p * dT over the ocean volume
    """

    # Physical constants
    CP_SEAWATER = 3850.0  # J/(kg·K) - specific heat capacity of seawater

    # Convert dz from cm to m
    dz_m = ds['dz'] / 100.0  # [m]
    
    # Convert TAREA from cm² to m²
    tarea_m2 = ds['TAREA'] / 1e4  # [m²]

    # Ocean mask (0=land, >0=ocean)
    kmt = ds['KMT']
    kmt_mask = kmt > 0  # True for ocean points, False for land points

    # Convert RHO from g/cm³ to kg/m³, 1kg = 1000g, 1m3 = 1e6 cm3
    rho_kg_m3 = ds['RHO'] * 1000.0  # [kg/m³]

    # Calculate OHC per unit volume: ρ × c_p × ΔT
    # Shape: (z_t, nlat, nlon) [J/(m³·K)]
    ohc_per_vol = rho_kg_m3 * CP_SEAWATER * ds['TEMP']  # [J/(m³)]
    # Integrate OHC per unit volume over depth to get OHC per unit area
    # Total OHC
    ohc_per_area = (ohc_per_vol * dz_m).sum(dim='z_t')  # [J/m²]
    # Get z_t where z_w_bot < 300m, 700m, 2000m
    depth_mask_300 = (ds['z_w_bot'] < 300e2).values
    depth_mask_700 = (ds['z_w_bot'] < 700e2).values
    depth_mask_2000 = (ds['z_w_bot'] < 2000e2).values
    # Down to 300m
    ohc_per_area_300m = (ohc_per_vol * dz_m).sel(z_t=depth_mask_300).sum(dim='z_t')  # [J/m²]
    # Down to 700m
    ohc_per_area_700m = (ohc_per_vol * dz_m).sel(z_t=depth_mask_700).sum(dim='z_t')  # [J/m²]
    # Down to 2000m
    ohc_per_area_2000m = (ohc_per_vol * dz_m).sel(z_t=depth_mask_2000).sum(dim='z_t')  # [J/m²]

    # Concatenate the OHC per area for different depth ranges into a single dataarray with a new "depth_range" dimension
    ohc_per_area = xr.concat(
        [ohc_per_area, ohc_per_area_300m, ohc_per_area_700m, ohc_per_area_2000m],
        pd.Index([-1, 300, 700, 2000], name="ohc_depth"),
    )

    # Compute global and ocean surface areas to include in metadata
    global_area = (ds['TAREA'] * 1e-4).sum(["nlat", "nlon"])
    ocean_area = (ds['TAREA'].where(ds["KMT"]>0) * 1e-4).sum(["nlat", "nlon"])

    # Apply ocean mask (KMT > 0 = ocean)
    ohc_masked = ohc_per_area.where(kmt_mask) # Set land points to NaN
    ohc_masked.attrs["long_name"] = "Ocean Heat Content per unit area"
    ohc_masked.attrs["units"] = "J/m²"
    ohc_masked.name = "OHC"
    # Compute the global mean OHC by averaging over lat and lon, weighting by the grid cell area
    # We can use the TAREA variable for area weighting
    global_mean_ohc = ohc_masked.weighted(tarea_m2).mean(dim=['nlat', 'nlon'], skipna=True)  # [J/m²]
    global_mean_ohc.attrs["long_name"] = "Global Ocean Heat Content"
    global_mean_ohc.attrs["units"] = "J/m²"
    global_mean_ohc.name = "OHC_global_mean"

    ohc_ds = xr.merge([ohc_masked, global_mean_ohc])
    ohc_ds.attrs["description"] = "Ocean Heat Content calculated from CESM2/POP2 output using the equation of state. OHC is calculated as the integral of rho * c_p * dT over the ocean volume, and then averaged over the ocean surface area to get OHC per unit area. The global mean OHC is also provided as a separate variable."
    ohc_ds.attrs["global_area_m2"] = global_area.values
    ohc_ds.attrs["ocean_area_m2"] = ocean_area.values

    if "time" not in ohc_ds.dims:
        return ohc_ds.compute()

    time_chunk = 24
    ntime = ohc_ds.sizes["time"]
    computed_chunks = []
    for start in range(0, ntime, time_chunk):
        stop = min(start + time_chunk, ntime)
        computed_chunks.append(ohc_ds.isel(time=slice(start, stop)).compute())
        logging.info(f"Computed OHC for time indices {start} to {stop-1} of {ntime}")

    logging.info("All time chunks computed, concatenating results")
    out = xr.concat(computed_chunks, dim="time")
    out.attrs = ohc_ds.attrs
    return out


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


# %%
if __name__ == "__main__":
    # If on CURC

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
        for pat in CASE_CONFIG[case]["file_patterns"]:
            test_out = crawl_and_process(
                input_dir=rawdata_root,
                output_dir=f"/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/{case}/",
                match_pattern=pat,
                process_fn=submit_OHC_job,
                process_path_fn=lambda p: p.replace("TEMP", "OHC")
            )

    # %%
