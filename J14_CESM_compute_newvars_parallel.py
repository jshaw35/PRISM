"""
Compute the background radiation and precipitation proxy states from CESM control simulations.
Results will be maps of these variables.

"""
# %%
from pathlib import Path
import os
import subprocess
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
        # Supply the output directory to the function arguments so it can exit early if the output file already exists.
        if output_dir is not None:
            out_root = output_dir if rel_root == "." else os.path.join(output_dir, rel_root)
            fn_args["out_root"] = out_root
        for name in files:
            # Only look for monthly files
            if "h0" not in name:
                continue
            src = os.path.join(root, name)
            data, filename = process_fn(src, **fn_args)
            # Fail gracefully
            if data is None:
                continue
            if output_dir is None:
                logging.info(f"Output directory not specified, skipping save for {filename}")
                continue
            dst = os.path.join(out_root, filename)
            if os.path.exists(dst):
                logging.info(f"{dst} already exists") # Should be duplicative now but keeping.
                continue
            os.makedirs(out_root, exist_ok=True)
            logging.info(f"Writing {dst}")
            data.to_netcdf(dst)


def compute_thermoprecip_wrapper(
    filepath: str,
    match_var: str,
    var_detect_str: str = "h0",
    out_root: str = None,
):

    # Parse the variable name from the test path, assuming it is in the format of "case/atm/proc/tseries/month_1/case.cam.h0.VAR.nc"
    filename = os.path.splitext(os.path.basename(filepath))[0]
    name_parts = filename.split(".")
    # Skip if not a h0 file
    try:
        marker_idx = name_parts.index(var_detect_str)
    except ValueError:
        return None, None
    test_var = name_parts[marker_idx + 1]
    if test_var != match_var:
        return None, None
    save_filename = filename.replace(match_var, "PRECIP_THERMO") + ".nc"
    if out_root is not None:
        save_filepath = os.path.join(out_root, save_filename)
        if os.path.exists(save_filepath):
            logging.info(f"{save_filepath} already exists")
            return None, None

    # Compute the precipitation proxy variable and add it to the list of variables to average
    precip_vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX"]
    precip_vars_plus = precip_vars + ["gw"]
    precip_files = [filepath.replace(match_var, _var) for _var in precip_vars]
    for _file in precip_files:
        assert os.path.exists(_file), f"{_file} does not exist"
    try:
        ds_merged = xr.open_mfdataset(precip_files, combine="by_coords", preprocess=lambda x: x.drop_vars(["time_written", "date_written"], errors="ignore"))
    except:
        logging.info(precip_files)
    ds_merged = ds_merged[precip_vars_plus]
    precip_ds = compute_thermoprecip(ds_merged)
    precip_ds = xr.merge([precip_ds, ds_merged["gw"]])

    return precip_ds, save_filename


def compute_toaimbalance_wrapper(
    filepath: str,
    match_var: str,
    var_detect_str: str = "h0",
    out_root: str = None,
):

    # Parse the variable name from the test path, assuming it is in the format of "case/atm/proc/tseries/month_1/case.cam.h0.VAR.nc"
    filename = os.path.splitext(os.path.basename(filepath))[0]
    name_parts = filename.split(".")
    # Skip if not a h0 file
    try:
        marker_idx = name_parts.index(var_detect_str)
    except ValueError:
        return None, None
    test_var = name_parts[marker_idx + 1]
    if test_var != match_var:
        return None, None

    save_filename = filename.replace(match_var, "FNNT") + ".nc"
    if out_root is not None:
        save_filepath = os.path.join(out_root, save_filename)
        if os.path.exists(save_filepath):
            logging.info(f"{save_filepath} already exists")
            return None, None

    # Compute the precipitation proxy variable and add it to the list of variables to average
    toanet_vars = ["FLNT", "FSNT"]
    toanet_vars_plus = toanet_vars + ["gw"]
    toanet_files = [filepath.replace(match_var, _var) for _var in toanet_vars]
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

    return toanet_ds, save_filename


def submit_new_vars_job(filepath, savepath):
    """
    Call an existing bash script and supply filepath and savepath as arguments.

    Args:
        filepath (_type_): _description_
        savepath (_type_): _description_
    """

    script_path = "/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_scripts/compute_newvars_single.sh"
    # print(f"qsub -v FILE_PATH={filepath},SAVE_PATH={savepath} {script_path}")
    subprocess.run(
        [
            "qsub",
            "-v",
            f"FILE_PATH={filepath},SAVE_PATH={savepath}",
            script_path,
        ],
        check=True,
    )


# %%
if __name__ == "__main__":
    machine = "glade"
    if machine == "glade":
        savepath_root = "/glade/work/jonahshaw/PRISM_data/derived_vars/"

    case_dict = {
        "ARISE-1.0": ["/glade/work/jonahshaw/PRISM_data/ARISE-1.0/"],
        "ARISE_SAI": ["/gdex/data/d651059/ARISE-SAI-1.5"],
        "CESM2_LME": ["/gdex/data/d651078"],
        "CESM2_WACCM_SSP2-4.5": ["/gdex/data/d651045/CESM2-WACCM-SSP245"],
        "CESM2_WACCM_SSP2-4.5_MCB": ["/gdex/data/d314006"],
        "CESM2_WACCM_1850control": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"]],
        "CESM2_WACCM_HIST": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.002", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.003"]],
    }

    for case in case_dict:
        load_paths = case_dict.get(case, [])
        if len(load_paths) == 1:
            load_path_list = [load_paths[0]]
            save_path_list = [case]
        elif len(load_paths) == 2:
            load_path_list = [load_paths[0] + subcase for subcase in load_paths[1]]
            save_path_list = [case + "/" + subcase for subcase in load_paths[1]]

        for load_path, save_path in zip(load_path_list, save_path_list):
            logging.info(f"Processing case: {load_path}")
            submit_new_vars_job(
                filepath=load_path,
                savepath=f"{savepath_root}/{save_path}",
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