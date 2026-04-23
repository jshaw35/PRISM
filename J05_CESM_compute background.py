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
varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS']

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
    L = 2.45e6  # J/kg, latent heat of vaporization or 2.257e3 J/kg or 2.5e6 J/kg
    vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX"]
    assert set(vars).issubset(set(ds.data_vars)), "Not all variables in varlist are in the dataset."

    R_TOA = ds["FLNT"] - ds["FSNT"]
    R_SFC = ds["FLNS"] - ds["FSNS"]
    SHFLX = ds["SHFLX"]

    # Compute in units of kg m^-2 s^-1
    P = (R_TOA - R_SFC - SHFLX) / L 
    # Convert to mm/day: 1000 mm / m, 86400 s / day, 1000 kg / m^3 for water density (last two cancel out)
    P = P * 86400
    P.attrs["long_name"] = "Thermodynamically-driven precipitation"
    P.attrs["units"] = "mm/day"
    P.name = "PRECIP_THERMO"
    return P


def crawl_and_process(input_dir, output_dir, process_fn):
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        out_root = output_dir if rel_root == "." else os.path.join(output_dir, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for name in files:
            src = os.path.join(root, name)
            dst = os.path.join(out_root, name)
            if os.path.exists(dst):
                logging.info(f"{dst} already exists")
                continue
            logging.info(f"Processing {src}")
            data = process_fn(src)
            if data is None:
                logging.error(f"Failed to process {src}")
                continue
            logging.info(f"Writing {dst}")
            data.to_netcdf(dst)


def compute_background_state(
    rawdata_root,
    save_path,
    case_name: str,
    pattern: list[str],
    mask= list[int] | None,
):
    label = pattern[0]
    glob_str = pattern[1]
    save_path = save_path / (case_name + ".nc")
    if save_path.exists():
        logging.info(f"{save_path} already exists, skipping computation for case {case_name}")
        return
    filepaths = list(rawdata_root.glob(glob_str))

    mean_var_list = []

    # Compute the precipitation proxy variable and add it to the list of variables to average
    precip_files = {}
    precip_vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX"]
    for _var in precip_vars:
        varfiles = [fp for fp in filepaths if f".{_var}." in fp.name]
        varfiles.sort()  # Ensure files are in a consistent order
        precip_files[_var] = varfiles
    for i in range(len(precip_files["FLNT"])):
        trying_files = [precip_files[_var][i] for _var in precip_vars]
        ds_merged = xr.open_mfdataset(trying_files, combine="by_coords")
        if not set(precip_vars).issubset(set(ds_merged.data_vars)):
            logging.warning(f"Not all variables in precip_vars are in the merged dataset for index {i}. Skipping this index.")
            continue

        precip_ds = compute_thermoprecip(ds_merged)
        precip_ds = precip_ds.assign_coords(index_t=i).expand_dims("index_t")
        mean_var_list.append(precip_ds)

    for _var in varlist:
        subset_filepaths = [fp for fp in filepaths if f".{_var}." in fp.name]
        if not subset_filepaths:
            logging.warning(f"No files found for variable {_var} in case {case_name} with pattern {glob_str}")
            continue
        tmean_ds_list = []
        for i,fp in enumerate(subset_filepaths):
            logging.info(f"Processing file: {fp}")
            ds = xr.open_dataset(fp)
            # Get the time length and add it to weight the average by the number of time steps in each file later
            time_len = ds.dims.get("time")
            ds_tmean = ds.mean(dim="time")
            ds_tmean = ds_tmean.assign_coords(index_t=i).expand_dims("index_t")
            # Add a variable indexed by the new index_t dimension that contains the time length of the original dataset for weighting later
            ds_tmean = ds_tmean.assign(time_len=("index_t", [time_len]))
            tmean_ds_list.append(ds_tmean)
        # Compute the weighted average of the time means from each file
        tmean_ds = xr.concat(tmean_ds_list, dim="index_t")
        weighted_tmean = tmean_ds.weighted(tmean_ds["time_len"]).mean(dim="index_t")
        mean_var_list.append(weighted_tmean)
    mean_ds = xr.merge(mean_var_list)
    mean_ds.to_netcdf(save_path)


# %%
if __name__ == "__main__":
    rawdata_root = Path("/home/josh2250/kaydata/jshaw/RadInt_rawdata/")
    save_path = Path("/home/josh2250/projects/PRISM/data/control_baselines/")

    case_dict = {
        "CESM2_LME_control": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008","CESM2_LME/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/atm/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.cam.h0.*"],
        # "CESM2_LME": ["b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002", "CESM2_LME/d651078/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/atm/proc/tseries/month_1/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.cam.h0.*"],
        "CESM2_1850control": ["b.e21.B1850.f09_g17.CMIP6-piControl.001", "CESM2_1850control/b.e21.B1850.f09_g17.CMIP6-piControl.001/atm/proc/tseries/month_1/b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.*"],
    }

    for case, pattern in case_dict.items():
        logging.info("Processing: %s, pattern: %s" % (case, pattern[0]))
        compute_background_state(
            rawdata_root,
            save_path,
            case,
            pattern,
        )
        break

    # cesm2_lme_control_str = "CESM2_LME/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/atm/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.cam.h0.*"
    # cesm2_lme_str = "CESM2_LME/d651078/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/atm/proc/tseries/month_1/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.cam.h0.*"

    # cesm2_pi_control_str = "CESM2_1850control/b.e21.B1850.f09_g17.CMIP6-piControl.001/atm/proc/tseries/month_1/b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.*"

    # %%

    # curc_cesm2_lme_datapath = "/home/josh2250/projects/PRISM/data/RadInt_procdata/CESM2_LME/"
    # curc_cesm2_lme_outpath = "/home/josh2250/projects/PRISM/data/control_baselines/CESM2_LME/"

    # curc_cesm2_le_datapath = "/home/josh2250/projects/PRISM/data/RadInt_procdata/CESM2_LE/"
    # curc_cesm2_le_outpath = "/home/josh2250/projects/PRISM/data/control_baselines/CESM2_LE/"

    # curc_cesm2_mcb_datapath = "/home/josh2250/projects/PRISM/data/RadInt_procdata/CESM2_WACCM_SSP2-4.5_MCB/"
    # curc_cesm2_mcb_outpath = "/home/josh2250/projects/PRISM/data/control_baselines/CESM2_WACCM_SSP2-4.5_MCB/"

    # crawl_and_process(curc_lme_datapath, curc_lme_outpath, average_spatially)
    # crawl_and_process(curc_cesm2_245_datapath, curc_cesm2_245_outpath, average_spatially)
    # crawl_and_process(curc_ariseSAI_datapath, curc_ariseSAI_outpath, average_spatially)
    # crawl_and_process(curc_cesm2_lme_datapath, curc_cesm2_lme_outpath, average_spatially)
    # crawl_and_process(curc_cesm2_le_datapath, curc_cesm2_le_outpath, average_spatially)
    # crawl_and_process(curc_cesm2_mcb_datapath, curc_cesm2_mcb_outpath, average_spatially)