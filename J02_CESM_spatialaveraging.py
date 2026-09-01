

from pathlib import Path
import os
import xarray as xr
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def average_spatially(
    datapath,
    average_vars = ["CLDTOT", "FLNS", "FLNSC", "FLNT", "FLNTC", "FLNR", "FLUT", "FLUTC", "FSNR", "FSNT", "FSNS", "FSNSC", "FSNTOA", "FSNTC", "FLNTCLR", "FSNTOAC", "LHFLX", "SHFLX", "TS", "PRECT", "PRECC", "PRECL", "FNNT", "PRECIP_THERMO"],
    var_detect_str: str = "h0",
    out_root: str = None,
):
    # Parse the variable name from the test path, assuming it is in the format of "case/atm/proc/tseries/month_1/case.cam.h0.VAR.nc"
    filename = os.path.splitext(os.path.basename(datapath))[0]
    filename = filename + ".nc"
    if out_root is not None:
        save_filepath = os.path.join(out_root, filename)
        if os.path.exists(save_filepath):
            logging.info(f"{save_filepath} already exists")
            return None, None

    name_parts = filename.split(".")
    # Skip if not a h0 file
    try:
        marker_idx = name_parts.index(var_detect_str)
    except ValueError:
        return None, None
    test_var = name_parts[marker_idx + 1]
    if not (test_var in average_vars):
        return None, None
    
    ds = xr.open_dataset(datapath)

    varlist = [i for i in list(ds.data_vars) if i in average_vars]

    try:
        lon_average = ds[varlist].mean(dim="lon")
    except ValueError as e:
        logging.error(f"Error processing {datapath}: {e}")
        logging.info(f"Available variables: {list(ds[varlist].data_vars)}")
        return None

    T_average = lon_average.weighted(ds["gw"]).mean(dim="lat")
    T_average_SH = lon_average.sel(lat=slice(-90,0)).weighted(ds["gw"]).mean(dim="lat")
    T_average_NH = lon_average.sel(lat=slice(0,90)).weighted(ds["gw"]).mean(dim="lat")
    
    T_average = T_average.assign_coords(spatial="G").expand_dims("spatial")
    T_average_SH = T_average_SH.assign_coords(spatial="SH").expand_dims("spatial")
    T_average_NH = T_average_NH.assign_coords(spatial="NH").expand_dims("spatial")

    out = xr.combine_by_coords([T_average, T_average_SH, T_average_NH])
    # Do not rename, just save to a separate directory.
    return out, filename


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
                logging.info(f"{dst} already exists")
                continue
            os.makedirs(out_root, exist_ok=True)
            logging.info(f"Writing {dst}")
            data.to_netcdf(dst)


# %%
if __name__ == "__main__":
    machine = "glade"
    if machine == "glade":
        savepath_root = "/glade/work/jonahshaw/PRISM_data/spatial_averages_data/"
        # Path to variables used only here
        derivedpath_root = "/glade/work/jonahshaw/PRISM_data/derived_vars/"

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
            load_path_derived_list = [derivedpath_root + case + "/"]
            save_path_list = [case]
        elif len(load_paths) == 2:
            load_path_list = [load_paths[0] + subcase for subcase in load_paths[1]]
            load_path_derived_list = [derivedpath_root + case + '/' + subcase for subcase in load_paths[1]]
            save_path_list = [case + "/" + subcase for subcase in load_paths[1]]

        for load_path, load_path_derived, save_path in zip(load_path_list, load_path_derived_list, save_path_list):
            logging.info(f"Processing case: {load_path}")
            crawl_and_process2(
                input_dir=load_path,
                output_dir=f"{savepath_root}/{save_path}",
                process_fn=average_spatially,
            )
            # Now process the derived variables, which are stored in a different directory.
            crawl_and_process2(
                input_dir=load_path_derived,
                output_dir=f"{savepath_root}/{save_path}",
                process_fn=average_spatially,
            )

# %%