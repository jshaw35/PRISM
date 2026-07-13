"""
Compute the background radiation and precipitation proxy states from CESM control simulations.
Results will be maps of these variables.

"""
# %%
from pathlib import Path
import os
import xarray as xr
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

def compute_thermoprecip(
    ds,
):
    """Compute the thermodynamically-driven precipitation per O'Gorman 2012.
    L del P = del R_TOA - del R_SFC - del SHFLX
    P = (R_TOA - R_SFC - SHFLX) / L

    Where the convention is that positive value are upwards.

    Args:
        ds (_type_): _description_
    """
    L = 2.5e6  # J/kg, latent heat of vaporization or 2.257e3 J/kg or 2.45e6 J/kg
    # 2.5e6 per this lecture slide: https://ethz.ch/content/dam/ethz/special-interest/usys/iac/iac-dam/documents/edu/courses/climatological_and_hydrological_field_work/radiation_2025.pdf
    vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX"]
    assert set(vars).issubset(set(ds.data_vars)), "Not all variables in varlist are in the dataset."

    R_LW = ds["FLNT"] - ds["FLNS"] # Positive upwards convention for LW
    R_SW = -1 * (ds["FSNT"] - ds["FSNS"]) # The convention is positive downwards for SW, so invert
    R_ATM = R_LW + R_SW # Net radiation emitted/lost by the atmosphere
    SHFLX = ds["SHFLX"]
    # Compute in units of kg m^-2 s^-1
    P = (R_ATM - SHFLX) / L

    # R_TOA = ds["FLNT"] - ds["FSNT"] # Positive upwards convention for LW
    # R_SFC = (ds["FLNS"] - ds["FSNS"]) # The convention is positive downwards for SW, so invert
    # SHFLX = ds["SHFLX"]
    # # Compute in units of kg m^-2 s^-1
    # P = (R_TOA - R_SFC - SHFLX) / L

    # Convert to mm/day: 1000 mm / m, 86400 s / day, 1000 kg / m^3 for water density (last two cancel out)
    P = P * 86400
    P.attrs["long_name"] = "Thermodynamically-driven precipitation"
    P.attrs["units"] = "mm/day"
    P.name = "PRECIP_THERMO"
    return P


def crawl_and_process2(input_dir, output_dir, process_fn, **fn_args):
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        if output_dir is not None:
            out_root = output_dir if rel_root == "." else os.path.join(output_dir, rel_root)
            os.makedirs(out_root, exist_ok=True)
        for name in files:
            src = os.path.join(root, name)
            if output_dir is not None:
                dst = os.path.join(out_root, name)
                if os.path.exists(dst):
                    logging.info(f"{dst} already exists")
                    continue
            logging.info(f"Processing {src}")
            data = process_fn(src, **fn_args)
            if data is None:
                logging.error(f"Failed to process {src}")
                continue
            if output_dir is not None:
                logging.info(f"Writing {dst}")
                data.to_netcdf(dst)


def compute_thermoprecip_wrapper(
    filepath: str,
    match_var: str,
    var_detect_str: str = "h0",
):

    # Parse the variable name from the test path, assuming it is in the format of "case/atm/proc/tseries/month_1/case.cam.h0.VAR.nc"
    filename = os.path.splitext(os.path.basename(filepath))[0]
    name_parts = filename.split(".")
    marker_idx = name_parts.index(var_detect_str)
    test_var = name_parts[marker_idx + 1]
    if test_var != match_var:
        return 1
    save_path = filepath.replace(match_var, "PRECIP_THERMO")
    if os.path.exists(save_path):
        logging.info(f"{save_path} already exists")
        return 1

    # Compute the precipitation proxy variable and add it to the list of variables to average
    precip_vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX"]
    precip_vars_plus = precip_vars + ["gw"]
    precip_files = [filepath.replace(match_str, _var) for _var in precip_vars]
    for _file in precip_files:
        assert os.path.exists(_file), f"{_file} does not exist"
    try:
        ds_merged = xr.open_mfdataset(precip_files, combine="by_coords", preprocess=lambda x: x.drop_vars(["time_written", "date_written"], errors="ignore"))
    except:
        logging.info(precip_files)
    ds_merged = ds_merged[precip_vars_plus]
    precip_ds = compute_thermoprecip(ds_merged)
    precip_ds = xr.merge([precip_ds, ds_merged["gw"]])
    logging.info(f"Writing: {save_path}")
    os.path.dirname(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    precip_ds.to_netcdf(save_path)
    return 1

def compute_toaimbalance_wrapper(
    filepath: str,
    match_var: str,
    var_detect_str: str = "h0",
):

    # Parse the variable name from the test path, assuming it is in the format of "case/atm/proc/tseries/month_1/case.cam.h0.VAR.nc"
    filename = os.path.splitext(os.path.basename(filepath))[0]
    name_parts = filename.split(".")
    marker_idx = name_parts.index(var_detect_str)
    test_var = name_parts[marker_idx + 1]
    if test_var != match_var:
        return 1
    save_path = filepath.replace(match_var, "FNNT")
    if os.path.exists(save_path):
        logging.info(f"{save_path} already exists")
        return 1

    # Compute the precipitation proxy variable and add it to the list of variables to average
    toanet_vars = ["FLNT", "FSNT"]
    toanet_vars_plus = toanet_vars + ["gw"]
    toanet_files = [filepath.replace(match_str, _var) for _var in toanet_vars]
    for _file in toanet_files:
        assert os.path.exists(_file), f"{_file} does not exist"
    try:
        ds_merged = xr.open_mfdataset(toanet_files, combine="by_coords", preprocess=lambda x: x.drop_vars(["time_written", "date_written"], errors="ignore"))
    except:
        logging.info(toanet_files)
    ds_merged = ds_merged[toanet_vars_plus]
    toanet_ds = ds_merged["FSNT"] - ds_merged["FLNT"]
    toanet_ds.attrs["long_name"] = "Top-of-model net radiation (FSNT - FLNT)"
    toanet_ds.attrs["units"] = "W/m^-2"
    toanet_ds.name = "FNNT"
    toanet_ds = xr.merge([toanet_ds, ds_merged["gw"]])
    logging.info(f"Writing: {save_path}")
    os.path.dirname(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    toanet_ds.to_netcdf(save_path)
    return 1


# %%
if __name__ == "__main__":
    # If on CURC
    rawdata_root = Path("/home/josh2250/kaydata/jshaw/RadInt_rawdata/")
    match_str = "FLNT"
    # save_path = Path("/home/josh2250/projects/PRISM/data/control_baselines/")

    case_list = [
        "ARISE_SAI",
        "CESM2_1850control",
        "CESM2_LE",
        "CESM2_LME",
        "CESM2_SF",
        "CESM2_WACCM_SSP2-4.5",
        "CESM2_WACCM_SSP2-4.5_MCB",
        "CESM_LME",
        "CESM2_WACCM_1850control",
        "CESM2_WACCM_HIST",
    ]

    for case in case_list:
        crawl_and_process2(
            input_dir=rawdata_root / case,
            output_dir=None,
            process_fn=compute_thermoprecip_wrapper,
            match_var=match_str,
        )

        crawl_and_process2(
            input_dir=rawdata_root / case,
            output_dir=None,
            process_fn=compute_toaimbalance_wrapper,
            match_var=match_str,
        )

    # %%
    # Testing code
    # import matplotlib.pyplot as plt
    # precip_avg = precip_ds.mean(dim="time")
    # precip_avg.plot()
    # ds_avg = ds_merged.mean(dim="time")
    # %%
    # fig,axs = plt.subplots(2,3, figsize=(15,6))
    # fig.subplots_adjust(hspace=0.4)
    # ds_avg["FLNT"].plot(ax=axs[0,0])
    # ds_avg["FLNS"].plot(ax=axs[0,1])
    # (ds_avg["FLNT"] - ds_avg["FLNS"]).plot(ax=axs[0,2])
    # ds_avg["FSNT"].plot(ax=axs[1,0])
    # ds_avg["FSNS"].plot(ax=axs[1,1])
    # (ds_avg["FSNT"] - ds_avg["FSNS"]).plot(ax=axs[1,2])
    # %%