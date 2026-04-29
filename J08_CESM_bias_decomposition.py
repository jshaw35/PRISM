"""
Using the decomposition of Medeiros (2023) and Simpson et al. (2020).
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA002918
https://doi.org/10.1029/2020JD032835

Essentially, this breaks the error into a components from mean shifts and from spatial pattern errors (i.e., the Taylor diagram components of variance error and spatial correlation error). For my purpose, I think I may be able to combine the last two, but I'm not entirely sure.

"""
# %%
from pathlib import Path
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MultipleLocator

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

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


def get_weights_by_month(
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

    time_weights = []
    if account_for_leap == False:
        for _t in time_ds:
            time_weights.append(weights.sel(month=_t['time.month']))
    else:
        for _t in time_ds:
            if _t["time.year"] % 4 == 0:
                time_weights.append(weights_leap.sel(month=_t['time.month']))
            else:
                time_weights.append(weights.sel(month=_t['time.month']))

    # Duplicate the time dimension but with weights as values
    weights_ds = xr.DataArray(
        data=time_weights,
        dims="time",
        coords={
            "time":time_ds,
        },
    )
    return weights_ds


def crawl_and_list(input_dir, file_string):
    file_list = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if file_string in name:
                file_list.append(os.path.join(root, name))
    return file_list


def crawl_and_list_glob(input_dir, file_string):
    filelist = list(Path(input_dir).glob(f"**/{file_string}"))
    return [str(file) for file in filelist]


def compute_decadal(
    ds,
    center=True,
):
    ds_decadal = ds.resample(time='10YE', offset=pd.Timedelta(weeks=-52)).mean().groupby("time.year").mean()
    if center:
        ds_decadal["year"] = ds_decadal["year"] + 5
    return ds_decadal

    
# %%

if __name__ == "__main__":
    root_dir = "/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/"
    CASE_CONFIGS = {
        "CESM2-LME_control": {
            "path": root_dir + "data/error_relativetobaseline/CESM2_LME_control/",
            "subdirs": ["CESM2_LME"],
            "subdir_cases": {"CESM2_LME": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008", "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002"]},
            "append_cases": {
                "b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008": None,
                "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002": None,
            },
            "ufunc": None,
        },
        "CESM2_1850control": {
            "path": root_dir + "data/error_relativetobaseline/CESM2_1850control/",
            "subdirs": ["ARISE_SAI", "CESM2_1850control", "CESM2_LE", "CESM2_SF", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB"],
            "subdir_cases": {
                "CESM2_1850control": ["b.e21.B1850.f09_g17.CMIP6-piControl.001"],
                "CESM2_LE": ["b.e21.BHISTcmip6.f09_g17.LE2-1301.001"],
                "CESM2_SF": ["b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.101"],
                "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001"],
                "ARISE_SAI": ["1p5K-SAI.001", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001"],
                "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001"],
            },
            "append_cases": {
                "b.e21.B1850.f09_g17.CMIP6-piControl.001": None,
                "b.e21.BHISTcmip6.f09_g17.LE2-1301.001": None,
                "b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.101": None,
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001": None,
                "1p5K-SAI.001": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001": "1p5K-SAI.001",
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
            },
            "ufunc": None,
        },
        "CESM2-LE_CMIP6": {
            "path": root_dir + "data/error_relativetobaseline/CESM2_LE_2000_2009_cmip6/",
            "subdirs": ["ARISE_SAI", "CESM2_1850control", "CESM2_LE", "CESM2_SF", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB"],
            "subdir_cases": {
                "CESM2_1850control": ["b.e21.B1850.f09_g17.CMIP6-piControl.001"],
                "CESM2_LE": ["b.e21.BHISTcmip6.f09_g17.LE2-1301.001"],
                "CESM2_SF": ["b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.101"],
                "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001"],
                "ARISE_SAI": ["1p5K-SAI.001", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001"],
                "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001"],
            },
            "append_cases": {
                "b.e21.B1850.f09_g17.CMIP6-piControl.001": None,
                "b.e21.BHISTcmip6.f09_g17.LE2-1301.001": None,
                "b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.101": None,
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001": None,
                "1p5K-SAI.001": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001": "1p5K-SAI.001",
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
            },
            "ufunc": None,
        },
        "CESM2-LE_SMBB": {
            "path": root_dir + "data/error_relativetobaseline/CESM2_LE_2000_2009_smbb/",
            "subdirs": ["ARISE_SAI", "CESM2_1850control", "CESM2_LE", "CESM2_SF", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB"],
            "subdir_cases": {
                "CESM2_1850control": ["b.e21.B1850.f09_g17.CMIP6-piControl.001"],
                "CESM2_LE": ["b.e21.BHISTcmip6.f09_g17.LE2-1301.001"],
                "CESM2_SF": ["b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.101"],
                "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001"],
                "ARISE_SAI": ["1p5K-SAI.001", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001"],
                "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001"],
            },
            "append_cases": {
                "b.e21.B1850.f09_g17.CMIP6-piControl.001": None,
                "b.e21.BHISTcmip6.f09_g17.LE2-1301.001": None,
                "b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.101": None,
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001": None,
                "1p5K-SAI.001": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001": "1p5K-SAI.001",
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
            },
            "append_case": None,
            "ufunc": None,
        },
        # Add new cases here when ready
    }

    # %%
    # Load the data in a nested dictionary structure. The top level keys are the control case labels (e.g. "CESM2-LME", what is being used as the baseline for the error calculation). The second level keys are the simulations that are being tested against, and the third level keys are the specific case strings that are being used to identify the files for each simulation.
    data_dict = {}
    varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO"]
    year_dim = "year"
    for case_label in CASE_CONFIGS.keys():
        datapath = CASE_CONFIGS[case_label]["path"]
        case_dict = {}
        for subdir in CASE_CONFIGS[case_label]["subdir_cases"]:
            subcase_dict = {}
            datapath_subdir = os.path.join(datapath, subdir)
            for case_str in CASE_CONFIGS[case_label]["subdir_cases"][subdir]:

                all_files = []
                for var in varlist:
                    var_files = crawl_and_list_glob(datapath_subdir, f"**/*{case_str}*.{var}.*nc")
                    all_files.extend(var_files)
                if len(all_files) == 0:
                    logging.warning(f"No files found for case {case_label} with case string {case_str} in path {datapath_subdir}")
                    continue
                
                all_ds = xr.open_mfdataset(all_files)

                # Handle the CESM time coordinate issue and challenges with cftime.DatetimeNoLeap
                if "time" in all_ds.coords:
                    if all_ds["time"][0]["time.month"] == 2:
                        all_ds = all_ds.assign_coords(
                            time=shift_noleap_time_back_one_month(all_ds["time"].values)
                        )

                # If there is an append case specified, append the data from that case to the current dataset along the time dimension
                # e.g. for ARISE-SAI, we want to append the CESM2-SSP2-4.5 data it is branched from. We will assume that the append case has already been loaded and is available in data_dict.
                if CASE_CONFIGS[case_label]["append_cases"][case_str] is not None:
                    append_case_label = CASE_CONFIGS[case_label]["append_cases"][case_str]
                    # Get the subdir for the append case, which may be different from the current subdir.
                    append_subdir = None
                    for subdir_key, case_list in CASE_CONFIGS[case_label]["subdir_cases"].items():
                        if append_case_label in case_list:
                            append_subdir = subdir_key
                            break
                    if append_subdir is None:
                        logging.warning(f"Append case {append_case_label} not found in subdir_cases for case {case_label}. Skipping append.")
                    else:
                        if append_subdir == subdir:
                            append_ds = subcase_dict[append_case_label].sel({year_dim:slice(None, str(all_ds[year_dim][0].values - 1))})
                        else:
                            append_ds = case_dict[append_subdir][append_case_label].sel({year_dim:slice(None, str(all_ds[year_dim][0].values - 1))})
                        all_ds = xr.concat([append_ds, all_ds], dim=year_dim)

                if CASE_CONFIGS[case_label]["ufunc"] is not None:
                    all_ds = CASE_CONFIGS[case_label]["ufunc"](all_ds)
                subcase_dict[case_str] = all_ds
            case_dict[subdir] = subcase_dict
        data_dict[case_label] = case_dict
    
    # %%
    # Draft some plots to give opencode something to work with later.
    case_label = "CESM2_1850control"
    control_label = "CESM2_1850control"
    control_case = CASE_CONFIGS[case_label]["subdir_cases"][control_label][0]
    subdirs = {
        # "CESM2_1850control": "b.e21.B1850.f09_g17.CMIP6-piControl.001",
        "CESM2_LE": "b.e21.BHISTcmip6.f09_g17.LE2-1301.001",
        "CESM2_SF": "b.e21.B1850cmip6.f09_g17.CESM2-SF-EE.101",
        "CESM2_WACCM_SSP2-4.5": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
        "ARISE_SAI": "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001",
        "CESM2_WACCM_SSP2-4.5_MCB": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001",
    }
    test_var = "PRECIP_THERMO"
    test_var = "FSNS"
    error_components = ["NMSE"] #, "U"]
    xlims = (1850, 2100)
    component_plot_args = {
        "ARISE_SAI": {"color": "red", "linestyle": "--"},
        "CESM2_1850control": {"color": "blue", "linestyle": "-"},
        "CESM2_LE": {"color": "green", "linestyle": "-."},
        "CESM2_SF": {"color": "orange", "linestyle": "--"},
        "CESM2_WACCM_SSP2-4.5": {"color": "purple", "linestyle": "-"},
        "CESM2_WACCM_SSP2-4.5_MCB": {"color": "brown", "linestyle": "--"}
    }
    # component_plot_args = {
    #     "NMSE": {"label": "Variance error", "color": "red", "linestyle": "--"},
    #     "U": {"label": "Spatial correlation error", "color": "blue", "linestyle": "dotted"},
    # }
    time_dim = "year"

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    control = data_dict[case_label][control_label][control_case]
    for component in error_components:
        control_component_data = control[test_var].sel(error_component=component)
        control_mean = control_component_data.mean(dim=time_dim)
        control_stddev = control_component_data.std(dim=time_dim)
        ax.fill_between(
            np.arange(xlims[0], xlims[1]+1, 1),
            control_mean -2 *control_stddev,
            control_mean +2*control_stddev,
            label=f"{control_label} - {component}",
            color="black", linestyle="-", alpha=0.3,
        )
    for subdir in subdirs:
        case_str = subdirs[subdir]
        ds = data_dict[case_label][subdir][case_str]
        data = ds[test_var]
        for component in error_components:
            component_data = data.sel(error_component=component)
            ax.plot(
                component_data[time_dim], component_data,
                label=f"{subdir} - {component}",
                **component_plot_args[subdir],
            )
    plt.legend()
# %%
