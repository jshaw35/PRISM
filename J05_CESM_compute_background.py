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
varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FLUTC', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL"]

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


def compute_background_state(
    rawdata_root,
    save_path,
    case_name: str,
    pattern: list[str],
    mask=list[int] | None,
    tslice=None,
):
    label = pattern[0]
    glob_str = pattern[1]
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path / (case_name + ".nc")
    if save_path.exists():
        logging.info(f"{save_path} already exists, skipping computation for case {case_name}")
        return
    filepaths = list(rawdata_root.glob(glob_str))
    if len(filepaths) < 1:
        logging.info(f"No files found for root '{str(rawdata_root)}' and glob string '{glob_str}'")
        return

    mean_var_list = []

    # Compute the precipitation proxy variable and add it to the list of variables to average
    logging.info("Processing precipitation proxy variable")
    precip_files = {}
    precip_vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX"]
    for _var in precip_vars:
        varfiles = [fp for fp in filepaths if f".{_var}." in fp.name]
        varfiles.sort()  # Ensure files are in a consistent order
        precip_files[_var] = varfiles
    for i in range(len(precip_files["FLNT"])):
        print(i)
        trying_files = [precip_files[_var][i] for _var in precip_vars]
        ds_merged = xr.open_mfdataset(trying_files, combine="by_coords", preprocess=lambda x: x.drop_vars(["time_written", "date_written"]))[precip_vars]

        # Handle the CESM time coordinate issue and challenges with cftime.DatetimeNoLeap
        if ds_merged["time"][0]["time.month"] == 2:
            ds_merged = ds_merged.assign_coords(
                time=shift_noleap_time_back_one_month(ds_merged["time"].values)
            )
        # Get the time length and add it to weight the average by the number of time steps in each file later
        time_len = ds_merged.sizes["time"]
        if tslice is not None:
            ds_merged = ds_merged.sel(time=tslice)
        ds_tmean = ds_merged.mean(dim="time")
        if not set(precip_vars).issubset(set(ds_tmean.data_vars)):
            logging.warning(f"Not all variables in precip_vars are in the merged dataset for index {i}. Skipping this index.")
            continue

        precip_ds = compute_thermoprecip(ds_tmean)
        precip_ds = precip_ds.assign_coords(index_t=i).expand_dims("index_t")
        precip_ds = precip_ds.assign_coords(time_len=("index_t", [time_len]))
        mean_var_list.append(precip_ds)

    precip_ds_all = xr.concat(mean_var_list, dim="index_t")
    precip_ds_all = precip_ds_all.weighted(precip_ds_all["time_len"]).mean("index_t")

    mean_var_list = []
    for _var in varlist:
        subset_filepaths = [fp for fp in filepaths if f".{_var}." in fp.name]
        subset_filepaths.sort()  # Ensure files are in a consistent order
        if not subset_filepaths:
            logging.warning(f"No files found for variable {_var} in case {case_name} with pattern {glob_str}")
            continue
        logging.info(f"Processing variable: {_var} with {len(subset_filepaths)} files for case {case_name}")
        tmean_ds_list = []
        for i,fp in enumerate(subset_filepaths):
            ds = xr.open_dataset(fp)[_var]
            # Get the time length and add it to weight the average by the number of time steps in each file later
            time_len = ds.sizes["time"]
            ds_tmean = ds.mean(dim="time")
            ds_tmean = ds_tmean.assign_coords(index_t=i).expand_dims("index_t")
            # Add a variable indexed by the new index_t dimension that contains the time length of the original dataset for weighting later
            ds_tmean = ds_tmean.assign_coords(time_len=("index_t", [time_len]))
            tmean_ds_list.append(ds_tmean)
        # Compute the weighted average of the time means from each file
        tmean_ds = xr.concat(tmean_ds_list, dim="index_t")
        weighted_tmean = tmean_ds.weighted(tmean_ds["time_len"]).mean(dim="index_t")
        mean_var_list.append(weighted_tmean)

    if not mean_var_list:
        logging.error(f"No variables were processed for case {case_name} with pattern {glob_str}. No output will be saved.")
        return
    mean_ds = xr.merge(mean_var_list)
    final_ds = xr.merge([mean_ds, precip_ds_all])
    # return final_ds
    final_ds.to_netcdf(save_path)


def compute_background_state_ohc(
    rawdata_root,
    save_path,
    case_name: str,
    pattern: list[str],
    mask=list[int] | None,
    tslice=None,
):
    label = pattern[0]
    glob_str = pattern[1]
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path / (case_name + ".nc")
    if save_path.exists():
        logging.info(f"{save_path} already exists, skipping computation for case {case_name}")
        return
    filepaths = list(rawdata_root.glob(glob_str))
    if len(filepaths) < 1:
        logging.info(f"No files found for root '{str(rawdata_root)}' and glob string '{glob_str}'")
        return

    _var = "OHC"
    subset_filepaths = [fp for fp in filepaths if f".{_var}." in fp.name]
    subset_filepaths.sort()  # Ensure files are in a consistent order
    if not subset_filepaths:
        logging.warning(f"No files found for variable {_var} in case {case_name} with pattern {glob_str}")
        return
    logging.info(f"Processing variable: {_var} with {len(subset_filepaths)} files for case {case_name}")
    tmean_ds_list = []
    for i,fp in enumerate(subset_filepaths):
        ds = xr.open_dataset(fp)[_var]
        # Get the time length and add it to weight the average by the number of time steps in each file later
        time_len = ds.sizes["time"]
        ds_tmean = ds.mean(dim="time")
        ds_tmean = ds_tmean.assign_coords(index_t=i).expand_dims("index_t")
        # Add a variable indexed by the new index_t dimension that contains the time length of the original dataset for weighting later
        ds_tmean = ds_tmean.assign_coords(time_len=("index_t", [time_len]))
        tmean_ds_list.append(ds_tmean)
    # Compute the weighted average of the time means from each file
    tmean_ds = xr.concat(tmean_ds_list, dim="index_t")
    weighted_tmean = tmean_ds.weighted(tmean_ds["time_len"]).mean(dim="index_t")

    weighted_tmean.to_netcdf(save_path)


# %%
if __name__ == "__main__":
    # If on CURC
    rawdata_root = Path("/home/josh2250/kaydata/jshaw/RadInt_rawdata/")
    ohcdata_root = Path("/home/josh2250/kaydata/jshaw/RadInt_ohcdata/")
    save_path = Path("/home/josh2250/projects/PRISM/data/control_baselines/")
    ohc_save_path = Path("/home/josh2250/projects/PRISM/data/control_baselines_ohc/")

    case_dict = {
        "CESM2_LME_control": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008","CESM2_LME/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/atm/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.cam.h0.*", None],
        "CESM2_1850control": ["b.e21.B1850.f09_g17.CMIP6-piControl.001", "CESM2_1850control/b.e21.B1850.f09_g17.CMIP6-piControl.001/atm/proc/tseries/month_1/b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.*.1?????-??????.nc", None],
        "CESM2_LE_2000_2009_cmip6": ["b.e21.BHISTcmip6.f09_g17.LE2-1301.00?", "CESM2_LE/d651056/CESM2-LE/atm/proc/tseries/month_1/*/b.e21.BHISTcmip6.f09_g17.LE2-1301.00?.cam.h0.*.200001-200912.nc", None],
        "CESM2_WACCM_HIST_1850_1864": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?", "CESM2_WACCM_HIST/atm/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.cam.h0.*.nc", slice("1850", "1864")],
        "CESM2_WACCM_HIST_2000_2014": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?", "CESM2_WACCM_HIST/atm/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.cam.h0.*.nc", slice("2000", "2014")],
        "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0*", "CESM2_WACCM_1850control/atm/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0.*.nc", slice("0100", None)],
    }

    ohc_case_dict = {
        "CESM2_WACCM_HIST_1850_1864": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?", "CESM2_WACCM_HIST/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/ocn/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.pop.h.*.nc", slice("1850", "1864")],
        "CESM2_WACCM_HIST_2000_2014": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?", "CESM2_WACCM_HIST/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/ocn/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.pop.h.*.nc", slice("2000", "2014")],
        "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h*", "CESM2_WACCM_1850control/ocn/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h.*.nc", slice("0100", None)],
        "CESM2_WACCM_1850control_0050_0075": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h*", "CESM2_WACCM_1850control/ocn/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h.*.nc", slice("0050", "0075")],
    }

    # If on glade
    # rawdata_root = Path("/gdex/data/")
    # save_path = Path("/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/control_baselines/")

    # case_dict = {
    #     "CESM2_LME_control": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008","d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/atm/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.cam.h0.*"],
    #     # "CESM2_LME": ["b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002", "CESM2_LME/d651078/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/atm/proc/tseries/month_1/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.cam.h0.*"],
    #     "CESM2_1850control": ["b.e21.B1850.f09_g17.CMIP6-piControl.001", "b.e21.B1850.f09_g17.CMIP6-piControl.001/atm/proc/tseries/month_1/b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.*"],
    # }

    for case, pattern in case_dict.items():
        logging.info("Processing: %s, pattern: %s" % (case, pattern[0]))
        _ = compute_background_state(
            rawdata_root,
            save_path,
            case,
            pattern,
            tslice=pattern[2],
        )

    for case, pattern in ohc_case_dict.items():
        logging.info("Processing: %s, pattern: %s" % (case, pattern[0]))
        _ = compute_background_state_ohc(
            ohcdata_root,
            ohc_save_path,
            case,
            pattern,
            tslice=pattern[2],
        )
        # break

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