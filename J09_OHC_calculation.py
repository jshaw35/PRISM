#!/usr/bin/env python3
"""
CESM2 Ocean Heat Content (OHC) Calculator

Computes global ocean heat content from CESM2/POP2 monthly output using full
equation of state (pre-calculated density from model).

Author: OpenCode
Date: 2026-04-30
"""
# %%
import os
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# %%

def crawl_and_process2(
    input_dir,
    output_dir,
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
            if output_dir is not None:
                dst = os.path.join(out_root, name)
                if os.path.exists(dst):
                    logging.info(f"{dst} already exists")
                    continue
            data = process_fn(src, **fn_args)
            if isinstance(data, xr.Dataset):
                return data
                if output_dir is not None:
                    logging.info(f"Processed {src}")
                    if process_path_fn is not None:
                        dst = process_path_fn(dst)
                    logging.info(f"Writing {dst}")
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    data.to_netcdf(dst)
                    logging.info(f"done.")


def compute_OHC_wrapper(
    filepath: str,
    savepath: Path,
    match_pattern: str,
    match_var: str = "TEMP",
    var_detect_str: str = "h",
):

    # Parse the variable name from the test path
    # Test that the filepath matches the expected pattern and contains the variable we want to process (TEMP). If not, skip processing.
    # Check that filepath matches match_pattern wildcard pattern (e.g. "*/ocn/proc/tseries/month_1/*TEMP*"). If not, skip processing.
    if not Path(filepath).match("**/" + match_pattern + "**.nc"):
        # logging.info(f"{filepath} does not match pattern {match_pattern}, skipping")
        return 1
    filename = os.path.splitext(os.path.basename(filepath))[0]
    name_parts = filename.split(".")
    marker_idx = name_parts.index(var_detect_str)
    test_var = name_parts[marker_idx + 1]
    if test_var != match_var:
        return 1
    save_path = savepath / os.path.basename(filepath).replace(match_var, "OHC").replace(match_var, "OHC")
    if os.path.exists(save_path):
        logging.info(f"{save_path} already exists")
        return 1

    # Compute the precipitation proxy variable and add it to the list of variables to average
    rho_file = filepath.replace(match_var, "RHO")
    if not os.path.exists(rho_file):
        logging.error(f"{rho_file} does not exist, cannot compute OHC")
        return 1
    ds_merged = xr.open_mfdataset([filepath, rho_file], combine="by_coords", chunks={"time":1}, decode_timedelta=True, preprocess=lambda x: x.drop_vars(["time_written", "date_written"], errors="ignore"))[["TEMP", "RHO", "dz", "TAREA", "KMT"]]

    # Correct CESM time if necessary
    if ds_merged["time"][0]["time.month"] == 2:
        ds_merged = ds_merged.assign_coords(
            time=shift_noleap_time_back_one_month(ds_merged["time"].values)
        )

    return compute_OHC(ds_merged)


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
    ohc_per_area = (ohc_per_vol * dz_m).sum(dim='z_t')  # [J/m²]

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

    return computed_chunks, ohc_ds.attrs
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
    # save_path = Path("/home/josh2250/projects/PRISM/data/spatial_OHC_data/")

    CASE_CONFIG = {
        # "CESM2-LME": {
        #     "data_dir": "/gdex/data/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/ocn/proc/tseries/month_1/",
        #     "file_patterns": [
        #         "b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/ocn/proc/tseries/month_1/",
        #         "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/ocn/proc/tseries/month_1/",
        #     ]
        # },
        "CESM2_LME": {
            "data_dir": "/gdex/data/d651078",
            "file_patterns": [
                # "b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/ocn/proc/tseries/month_1/",
                "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/ocn/proc/tseries/month_1/",
            ]
        },
        # "CESM2_WACCM_HIST": {
        #     "data_dir": "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/",
        #     "file_patterns": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/ocn/proc/tseries/month_1/"]
        # },
        # "CESM2_WACCM_1850control": {
        #     "data_dir": "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.BW1850.f09_g17.CMIP6-piControl.001",
        #     "file_patterns": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1/"]
        # },
        # "CESM2_WACCM_SSP2-4.5": {
        #     "data_dir": "/gdex/data/d651045/CESM2-WACCM-SSP245",
        #     "file_patterns": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?/ocn/proc/tseries/month_1/"]
        # },
        # "ARISE_SAI": {
        #     "data_dir": "/gdex/data/d651059/ARISE-SAI-1.5",
        #     "file_patterns": [
        #         "1p5K-SAI.00?/ocn/proc/tseries/month_1/",
        #         "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?/ocn/proc/tseries/month_1/",
        #         "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?/ocn/proc/tseries/month_1/",
        #     ]
        # },
        # "CESM2_WACCM_SSP2-4.5_MCB": {
        #     "data_dir": "/gdex/data/d314006",
        #     "file_patterns": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?/ocn/proc/tseries/month_1/"]
        # },
        # Add more cases as needed
    }
    # Normal units are kJ/cm2 (SI units are J/m2). Reasonable values are 0 - 140 kJ/cm2 (0 - 1.4e9 J/m2) (the conversion factor is 1 kJ/cm2 = 1e7 J/m2)

    for case in CASE_CONFIG:
        rawdata_root = Path(CASE_CONFIG[case]["data_dir"])
        for pat in CASE_CONFIG[case]["file_patterns"]:
            test_out = crawl_and_process2(
                input_dir=rawdata_root,
                savepath=Path("/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/spatial_OHC_data/"),
                output_dir="/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/spatial_OHC_data/",
                match_pattern=pat,
                process_fn=compute_OHC_wrapper,
                process_path_fn=lambda p: case + "/" + p.replace("TEMP", "OHC")
            )
            break
        break

    # %%
