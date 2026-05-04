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


def compute_IEEI(
    olr_ds,
    asr_ds,
    account_for_leap: bool = False,
):
    """
    Compute the integrated earth's energy imbalance (IEEI) from ASR and OLR fields.
    """
    assert (olr_ds["time"] == asr_ds["time"]).all(), "OLR and ASR time fields are not identical"
    time_ds = olr_ds["time"]

    weights = get_weights_by_month(time_ds, account_for_leap)
    eei_ds = asr_ds - olr_ds
    ieei_ds = np.cumsum(eei_ds * weights)

    # Convert to Watt by multipling by the Earth's surface area
    earth_radius = 6371e3 # meters
    earth_SA = 4 * np.pi * earth_radius**2

    return earth_SA * ieei_ds


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


def plot_eei_timeseries(
    asr_annual, olr_annual, eei_annual, ieei_annual,
    asr_decadal, olr_decadal, eei_decadal, ieei_decadal,
    ax=None, fontsize=14, case_name="",time_dim="year",
):
    """
    Plot timeseries of ASR, OLR, EEI, and IEEI on a subplot with twinned y-axes.
    
    Parameters
    ----------
    asr_annual, olr_annual, eei_annual, ieei_annual : xr.DataArray
        Annual-mean data
    asr_decadal, olr_decadal, eei_decadal, ieei_decadal : xr.DataArray
        Decadal-mean data
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    fontsize : int, default 14
        Font size for labels and titles
    case_name : str, default ""
        Name of the case to use as subplot title
    
    Returns
    -------
    ax1, ax2, ax3 : matplotlib.axes.Axes
        The three twinned axes (ASR/OLR, EEI, IEEI)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax1 = ax
    ax2 = ax1.twinx()
    if ieei_annual is not None:
        ax3 = ax1.twinx()
        
        # Offset the right spine of ax3 so it doesn't overlap with ax2
        ax3.spines["right"].set_position(("outward", 50))

    # Variable colors
    cmap = sns.color_palette("colorblind", n_colors=4)
    asr_color = cmap[0]
    olr_color = cmap[1]
    eei_color = cmap[2]
    ieei_color = cmap[3]
    # asr_color = "#1f77b4"  # Blue
    # olr_color = "#d62728"  # Red
    # eei_color = "#ff7f0e"  # Orange
    # ieei_color = "#2ca02c"  # Green
    
    # Plot annual means (thin, semi-transparent)
    ax1.plot(asr_annual[time_dim], asr_annual, color=asr_color, linestyle="-", 
             linewidth=0.7, alpha=0.5, label="ASR (annual)")
    ax1.plot(olr_annual[time_dim], olr_annual, color=olr_color, linestyle="-", 
             linewidth=0.7, alpha=0.5, label="OLR (annual)")
    
    ax2.plot(eei_annual[time_dim], eei_annual, color=eei_color, linestyle="-", 
             linewidth=0.7, alpha=0.5, label="EEI (annual)")
    if ieei_annual is not None:
        ax3.plot(ieei_annual[time_dim], ieei_annual, color=ieei_color, linestyle="-", 
                linewidth=0.7, alpha=0.5, label="IEEI (annual)")
    
    # Plot decadal means (thick, solid)
    ax1.plot(asr_decadal[time_dim], asr_decadal, color=asr_color, linestyle="-", 
             linewidth=2.5, label="ASR (decadal)")
    ax1.plot(olr_decadal[time_dim], olr_decadal, color=olr_color, linestyle="-", 
             linewidth=2.5, label="OLR (decadal)")
    
    ax2.plot(eei_decadal[time_dim], eei_decadal, color=eei_color, linestyle="-", 
             linewidth=2.5, label="EEI (decadal)")
    
    if ieei_annual is not None:
        ax3.plot(ieei_decadal[time_dim], ieei_decadal, color=ieei_color, linestyle="-", 
                linewidth=2.5, label="IEEI (decadal)")
    
    # Set axis labels and colors
    ax1.set_xlabel("Year", fontsize=fontsize)
    ax1.set_ylabel("ASR, OLR [W/m²]", fontsize=fontsize, color=asr_color)
    ax1.tick_params(axis="y", labelcolor=asr_color)
    
    ax2.set_ylabel("EEI [W/m²]", fontsize=fontsize, color=eei_color)
    ax2.tick_params(axis="y", labelcolor=eei_color)
    
    if ieei_annual is not None:
        ax3.set_ylabel("IEEI [W]", fontsize=fontsize, color=ieei_color)
        ax3.tick_params(axis="y", labelcolor=ieei_color)
    
    # Add title and grid
    ax1.set_title(case_name, fontsize=fontsize)
    ax1.grid(True, alpha=0.3)

    if ieei_annual is not None:
        return ax1, ax2, ax3
    else:
        return ax1, ax2, None


def plot_ieei_ts(
    ieei_annual, ieei_decadal, ts_annual, ts_decadal, ax=None,
    colors=["orange", "purple"], fontsize=14, time_dim="year",
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax1 = ax
    ax2 = ax.twinx()
    ax1.grid(True, alpha=0.3)
    # Plot annual means (thin, solid)
    ax1.plot(ieei_annual[time_dim], ieei_annual, color=colors[0], linestyle="-", 
                linewidth=0.7, alpha=0.5, label="IEEI (annual)")
    ax2.plot(ts_annual[time_dim], ts_annual, color=colors[1], linestyle="-", 
                linewidth=0.7, alpha=0.5, label="TS (annual)")
    
    # Plot decadal means (thick, solid)
    ax1.plot(ieei_decadal[time_dim], ieei_decadal, color=colors[0], linestyle="-", 
                linewidth=2.5, alpha=1, label="IEEI (decadal)")
    ax2.plot(ts_decadal[time_dim], ts_decadal, color=colors[1], linestyle="-", 
                linewidth=2.5, alpha=1, label="TS (decadal)")

    ax1.set_xlabel("Year", fontsize=14)
    ax1.set_ylabel("IEEI [J]", fontsize=14, color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2.tick_params(axis="y", labelcolor=colors[1])
    ax2.set_ylabel("Surface Temperature [K]", fontsize=14, color=colors[1])

    return ax1, ax2


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


def compute_ieei_with_start_year(
    asr_ds,
    olr_ds,
    start_year,
    account_for_leap: bool = False,
):
    """
    Compute the integrated earth's energy imbalance (IEEI) starting from a specified year.
    
    Parameters
    ----------
    asr_ds : xr.DataArray
        Absorbed shortwave radiation data
    olr_ds : xr.DataArray
        Outgoing longwave radiation data
    start_year : int
        Year to begin integration (IEEI will be zero at this year)
    account_for_leap : bool, default False
        Whether to account for leap years in the weighting
    
    Returns
    -------
    ieei_ds : xr.DataArray
        Integrated energy imbalance in Watts, with zero baseline at start_year
    """
    # Slice to start from the specified year
    asr_sliced = asr_ds.where(asr_ds["time.year"] >= start_year, drop=True)
    olr_sliced = olr_ds.where(olr_ds["time.year"] >= start_year, drop=True)
    
    # Compute IEEI using the existing function
    ieei_ds = compute_IEEI(olr_sliced, asr_sliced, account_for_leap=account_for_leap)
    
    return ieei_ds


def compute_decadal(
    ds,
    center=True,
):
    ds_decadal = ds.resample(time='10YE', offset=pd.Timedelta(weeks=-52)).mean().groupby("time.year").mean()
    if center:
        ds_decadal["year"] = ds_decadal["year"] + 5
    return ds_decadal


def extract_ensemble_numbers(filenames):
    """
    Extract ensemble numbers from CESM filenames.
    
    Ensemble numbers are 3-digit numeric strings bounded by periods (e.g., ".001.").
    This function parses each filename to identify the ensemble member.
    
    Args:
        filenames (List[str]): List of file paths
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping ensemble number strings to lists of files.
                             Keys are ensemble numbers (e.g., "001", "002", "101").
                             Values are lists of file paths containing that ensemble number.
    """
    ens_dict = {}
    
    for filepath in filenames:
        # Extract just the filename from the path
        filename = os.path.basename(filepath)
        
        # Split by period to find 3-digit numeric strings
        parts = filename.split(".")
        ens_number = None
        
        for part in parts:
            if len(part) == 3 and part.isdigit():
                ens_number = part
                break
        
        if ens_number is not None:
            if ens_number not in ens_dict:
                ens_dict[ens_number] = []
            ens_dict[ens_number].append(filepath)
        else:
            logging.warning(f"Could not extract ensemble number from filename: {filename}")
    
    return ens_dict


def get_ensemble_number_from_case_str(case_str):
    """
    Extract ensemble number from a case string (e.g., "b.e21.BHISTcmip6.f09_g17.LE2-1301.001" -> "001").
    
    Args:
        case_str (str): Case string identifier
    
    Returns:
        str: Ensemble number if found (3-digit numeric string), None otherwise
    """
    parts = case_str.split(".")
    for part in parts:
        if len(part) == 3 and part.isdigit():
            return part
    return None


def match_wildcard_case(pattern, case_list):
    """
    Find all cases in case_list that match the wildcard pattern.
    
    Simple wildcard matching: * matches any sequence of characters, ? matches single character.
    
    Args:
        pattern (str): Pattern string with optional * or ? wildcards (e.g., "case.name.*")
        case_list (List[str]): List of case strings to search
    
    Returns:
        List[str]: List of matching case strings from case_list
    """
    import fnmatch
    matches = [case for case in case_list if fnmatch.fnmatch(case, pattern)]
    return matches


def load_ensemble_cases(datapath_subdir, case_str, varlist):
    """
    Load case data with support for wildcard patterns matching multiple ensemble members.
    
    If case_str contains wildcards (* or ?):
    - Finds all matching files
    - Groups files by ensemble member (identified by 3-digit numeric strings in filenames)
    - Loads each ensemble separately to avoid conflicts
    - Adds 'ens' coordinate to track ensemble membership
    - Concatenates along new 'ens' dimension
    
    If case_str contains no wildcards:
    - Uses original behavior: finds all files matching the exact pattern
    - Returns single dataset as before
    
    Args:
        datapath_subdir (str): Path to subdirectory containing case files
        case_str (str): Case string, may contain wildcards (* or ?)
        varlist (List[str]): List of variable names to search for
    
    Returns:
        xarray.Dataset: Loaded dataset. If wildcards were used, includes new 'ens' dimension.
                       Returns None if no files are found.
    """
    has_wildcard = "*" in case_str or "?" in case_str
    
    if not has_wildcard:
        # Original behavior: no wildcards, use standard file finding
        all_files = []
        for var in varlist:
            var_files = crawl_and_list_glob(datapath_subdir, f"**/*{case_str}*.{var}.*nc")
            all_files.extend(var_files)
        all_files.sort()

        if len(all_files) == 0:
            return None
        
        all_ds = xr.open_mfdataset(all_files)
        return all_ds
    
    else:
        # Wildcard case: find matching files, group by ensemble, load separately
        all_files = []
        for var in varlist:
            # Use case_str directly in glob pattern (it contains wildcards)
            var_files = crawl_and_list_glob(datapath_subdir, f"**/*{case_str}*.{var}.*nc")
            all_files.extend(var_files)
        
        if len(all_files) == 0:
            logging.warning(f"No files found matching pattern: **/*{case_str}*.*.nc")
            return None
        
        # Extract ensemble numbers and group files
        all_files.sort()
        ens_dict = extract_ensemble_numbers(all_files)
        
        if len(ens_dict) == 0:
            logging.warning(f"No ensemble numbers could be extracted from matching files for pattern: {case_str}")
            return None
        
        # Sort ensemble numbers for consistent ordering
        sorted_ens_numbers = sorted(ens_dict.keys())
        
        # Load each ensemble member separately
        ensemble_datasets = []
        for ens_number in sorted_ens_numbers:
            ens_files = ens_dict[ens_number]
            
            try:
                # Load this ensemble's files with flexible coordinate handling
                ens_ds = xr.open_mfdataset(
                    ens_files,
                    combine='by_coords',
                    compat='no_conflicts'
                )
                
                # Add ensemble number as a data variable first, then expand the dimension
                ens_ds = ens_ds.expand_dims({'ens': [ens_number]})
                ensemble_datasets.append(ens_ds)
                
                logging.info(f"Loaded ensemble {ens_number} with {len(ens_files)} files")
            
            except Exception as e:
                logging.error(f"Error loading ensemble {ens_number}: {e}")
                continue
        
        if len(ensemble_datasets) == 0:
            logging.warning(f"No ensemble members could be loaded for pattern: {case_str}")
            return None
        
        # Concatenate all ensembles along the 'ens' dimension
        combined_ds = xr.concat(ensemble_datasets, dim='ens')

        return combined_ds


# %%

if __name__ == "__main__":
    root_dir = "/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/"
    CASE_CONFIGS = {
        "CESM_LME":{
            "path": root_dir + "data/RadInt_procdata/CESM_LME/",
            "subdir_cases": ["b.e11.BLMTRC5CN.f19_g16.00?"],
            "append_cases": {
                "b.e11.BLMTRC5CN.f19_g16.00?": None,
            },
            "ufunc": None,
        },
        "CESM2_LME": {
            "path": root_dir + "data/RadInt_procdata/CESM2_LME/",
            "subdir_cases": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008", "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002"],
            "append_cases": {
                "b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008": None,
                "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_1850control" :{
            "path": root_dir + "data/RadInt_procdata/CESM2_WACCM_1850control/",
            "subdir_cases": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_HIST": {
            "path": root_dir + "data/RadInt_procdata/CESM2_WACCM_HIST/",
            "subdir_cases": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001"],
            "append_cases": {
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?": None,
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_SSP2-4.5": {
            "path": root_dir + "data/RadInt_procdata/CESM2_WACCM_SSP2-4.5/",
            "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
            "append_cases": {
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001",
            },
            "ufunc": None,
        },
        "ARISE-SAI": {
            "path": root_dir + "data/RadInt_procdata/ARISE_SAI/",
            "subdir_cases": ["1p5K-SAI.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?"],
            "append_cases": {
                "1p5K-SAI.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?": "1p5K-SAI.00?",
            },
            "ufunc": None,
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "path": root_dir + "data/RadInt_procdata/CESM2_WACCM_SSP2-4.5_MCB/",
            "subdir_cases": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?", "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000", "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000", "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000", "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000", "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000"],
            "append_cases": {
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
            },
            "ufunc": None,
        },
    }
    # %%
    # Load the data in a nested dictionary structure. The top level keys are the control case labels (e.g. "CESM2-LME", what is being used as the baseline for the error calculation). The second level keys are the simulations that are being tested against, and the third level keys are the specific case strings that are being used to identify the files for each simulation.
    data_dict = {}
    varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO"]
    year_dim = "time"
    for case_label in CASE_CONFIGS.keys():
        logging.info(f"Loading data for case: {case_label}")
        datapath = CASE_CONFIGS[case_label]["path"]
        case_dict = {}
        for case_str in CASE_CONFIGS[case_label]["subdir_cases"]:
            logging.info(f"Loading data for subcase: {case_str}")

            # Load case data, supporting wildcards for ensemble members
            all_ds = load_ensemble_cases(datapath, case_str, varlist)
            if all_ds is None:
                logging.warning(f"No files found for case {case_label} with case string {case_str} in path {datapath}")
                continue

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
                # Handle both explicit case strings and wildcard patterns
                append_subdir = None
                append_case_to_use = None
                append_candidate = None

                has_wildcard = "*" in append_case_label or "?" in append_case_label
                # First check the current subdir since data_dict will not be updated with all cases until the end of the loop.
                if append_case_label in case_dict.keys():
                    append_candidate = case_dict.get(append_case_label)
                # Check if the append case label matches any cases in any of the subdirs for other cases.
                else:
                    for match_case in CASE_CONFIGS:
                        subdir_cases = CASE_CONFIGS[match_case]["subdir_cases"]
                        if has_wildcard:
                            # Try wildcard matching
                            matches = match_wildcard_case(append_case_label, subdir_cases)
                            if len(matches) > 0:
                                if len(matches) > 1:
                                    logging.warning(f"Append case pattern '{append_case_label}' matched multiple cases: {matches}. Using first match: {matches[0]}")
                                append_case_to_use = matches[0]
                                append_subdir = match_case
                                break
                        else:
                            # Exact match for non-wildcard case
                            if append_case_label in subdir_cases:
                                append_case_to_use = append_case_label
                                append_subdir = match_case
                                break
                if (append_subdir is None) and (append_candidate is None):
                    logging.warning(f"Append case {append_case_label} not found in subdir_cases for case {case_label}. Skipping append.")
                else:
                    if append_candidate is not None:
                        pass  # append_candidate was already found in the current case_dict, no need to search further
                    # Handle ensemble dimension in append case
                    elif append_subdir == case_str:
                        append_candidate = data_dict.get(append_subdir)
                    else:
                        append_candidate = data_dict.get(append_subdir, {}).get(append_case_to_use)
                    
                    if append_candidate is None:
                        logging.warning(f"Append case {append_case_label} not found in loaded data for case {case_label}. Skipping append.")
                        logging.warning(f"append_case_to_use: {append_case_to_use}")
                        logging.warning(f"append_subdir: {append_subdir}")
                        # Break the loop for testing purposes to avoid errors downstream
                    else:
                        # Extract ensemble number from current case if it has an ens dimension
                        if "ens" in all_ds.dims:
                            all_ds_ens_vals = all_ds["ens"].values
                            if "ens" in append_candidate.dims:
                                append_candidate_ens_vals = append_candidate["ens"].values
                                append_candidate_ens_vals_first = append_candidate_ens_vals[0]
                                match = [i in append_candidate_ens_vals for i in all_ds_ens_vals]
                                match_ens = [val if val in append_candidate_ens_vals else append_candidate_ens_vals_first for val in all_ds_ens_vals]
                                appended_list = []
                                for ens, match_bool in zip(all_ds_ens_vals, match):
                                    if "ens" not in append_candidate.indexes:
                                        append_candidate = append_candidate.set_index(ens="ens")
                                    if match_bool:
                                        append_ds_ens = append_candidate.sel(ens=ens, drop=False)
                                        logging.info(f"Appending ensemble {ens} from {append_case_label}")
                                    else:
                                        append_ds_ens = append_candidate.sel(ens=append_candidate_ens_vals_first, drop=False)
                                        logging.warning(f"Ensemble {ens} not found in append case {append_case_label}. Using first available ensemble {append_candidate_ens_vals_first}.")
                                    appended_list.append(append_ds_ens)
                                append_ds = xr.concat(appended_list, dim="ens")
                            else:
                                # No ens dimension in append candidate, use as-is
                                append_ds = append_candidate
                        else:
                            # Current case has no ensemble dimension
                            if "ens" in append_candidate.dims:
                                # Append candidate has ensembles, use first but keep as dimension
                                first_ens = append_candidate["ens"].values[0]
                                append_ds = append_candidate.isel(ens=0, drop=False)
                                logging.info(f"Current case has no ensemble dimension. Using first ensemble {first_ens} from append case.")
                            else:
                                # Neither has ensembles
                                append_ds = append_candidate
                        
                        # Perform the append operation with time dimension selection
                        # Check if cftime.DatetimeNoLeap is being used and select time accordingly
                        if isinstance(append_ds["time"][0].dtype, object):
                            # Likely cftime objects, select using cftime-compatible method
                            append_ds_subset = append_ds.sel({year_dim:slice(None, str(all_ds[year_dim][0].dt.year.values - 1))})
                        elif isinstance(all_ds["time"].values[0], np.datetime64) or isinstance(all_ds["time"].values[0], pd.Timestamp):
                            append_ds_subset = append_ds.sel({year_dim:slice(None, str(all_ds[year_dim][0].values - 1))})                        
                        else:
                            append_ds_subset = append_ds.sel({year_dim:slice(None, str(all_ds[year_dim][0].values - 1))})
                        
                        # Ensure ensemble dimension consistency before concatenation
                        # If one dataset has ens as an indexed dimension and the other doesn't, 
                        # reset the index to avoid xarray concat errors
                        if "ens" in all_ds.indexes and "ens" not in append_ds_subset.indexes:
                            # all_ds has indexed ens, append_ds_subset doesn't - reset all_ds ens index
                            all_ds = all_ds.reset_index("ens", drop=False)
                        elif "ens" not in all_ds.indexes and "ens" in append_ds_subset.indexes:
                            # append_ds_subset has indexed ens, all_ds doesn't - reset append_ds_subset ens index
                            append_ds_subset = append_ds_subset.reset_index("ens", drop=False)
                        
                        all_ds = xr.concat([append_ds_subset, all_ds], dim=year_dim)

            if CASE_CONFIGS[case_label]["ufunc"] is not None:
                all_ds = CASE_CONFIGS[case_label]["ufunc"](all_ds)
            case_dict[case_str] = all_ds
        data_dict[case_label] = case_dict
    

    # %%
    PLOT_CONFIGS1 = {
        "CESM_LME": {
            "selfunc": lambda ds: ds['b.e11.BLMTRC5CN.f19_g16.00?'].sel(ens="002"),
            # "ax1_lims": (228, 237),
            # "ax2_lims": (-3, 6),
            "ax1_lims": (230, 240),
            "ax2_lims": (-8, 2),
            "ax1_major_y": MultipleLocator(1),
            "axb1_lims": (-1.5e24, 1.5e24),
            "axb2_lims": (286.25, 287.75),
            "axb1_yticks": np.arange(-1.5e24, 1.5e24+0.01e24, 0.5e24),
            "axb2_yticks": np.arange(286.25, 287.75+0.01, 0.25),
            "xlims": (850, 1850),
            "keep_left_axes": True,
            "keep_right_axes": True,
        },
        "CESM2_LME": {
            "selfunc": lambda ds: ds['b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002'],
            # "ax1_lims": (235, 244),
            # "ax2_lims": (-3, 6),
            "ax1_lims": (237, 247),
            "ax2_lims": (-8, 2),
            "ax1_major_y": MultipleLocator(1),
            "axb1_lims": (-0.5e24, 1.0e24),
            "axb2_lims": (287.0, 289.5),
            "axb1_yticks": np.arange(-0.5e24, 1.0e24+0.01e24, 0.25e24),
            "axb2_yticks": np.arange(286.5, 289.5+0.01, 0.5),
            "xlims": (850, 1850),
            "keep_left_axes": True,
            "keep_right_axes": True,
        },
    }
    # %%
    # PLOT 1: Historical scenarios (CESM-LME, CESM2-LME)
    # Integration starts from year 850
    logging.info("Creating Plot 1: Historical scenarios (CESM-LME, CESM2-LME)")
    PLOT_CONFIGS = PLOT_CONFIGS1
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.40)
    
    case_list_1 = ["CESM_LME", "CESM2_LME"]
    start_year_1 = 850
    ts_var = "TS"
    olr_var = "FLNT"
    asr_var = "FSNT"

    for ax, axb, case_label in zip(axes[0], axes[1], case_list_1):
        logging.info(f"Plotting case: {case_label}")
        # Extract ASR and OLR data
        selfunc = PLOT_CONFIGS[case_label].get("selfunc")
        asr_ds = selfunc(data_dict[case_label])[asr_var].sel(spatial="G")
        olr_ds = selfunc(data_dict[case_label])[olr_var].sel(spatial="G")
        ts_ds = selfunc(data_dict[case_label])[ts_var].sel(spatial="G")
        
        # Compute annual means
        asr_annual = asr_ds.groupby("time.year").mean()
        olr_annual = olr_ds.groupby("time.year").mean()
        ts_annual = ts_ds.groupby("time.year").mean()
        eei_annual = asr_annual - olr_annual
        
        # Compute decadal means and center the decade labels by adding 5 years to the year coordinate (e.g. decade from 850-859 will be labeled as 855)        # Compute decadal means and center the decade labels by adding 5 years to the year coordinate (e.g. decade from 850-859 will be labeled as 855)
        asr_decadal = compute_decadal(asr_ds)
        olr_decadal = compute_decadal(olr_ds)
        ts_decadal = compute_decadal(ts_ds)
        eei_decadal = asr_decadal - olr_decadal
        
        # Compute IEEI starting from start_year_1
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, start_year_1)
        
        # Create annual and decadal means for IEEI by grouping years
        ieei_annual = ieei_ds.groupby("time.year").mean()
        ieei_decadal = ieei_ds.resample(time='10YE', offset=pd.Timedelta(weeks=-52)).mean().groupby("time.year").mean()
        
        # Plot ASR, OLR, and EEI
        ax1, ax2, ax3 = plot_eei_timeseries(
            asr_annual, olr_annual, eei_annual, None,
            asr_decadal, olr_decadal, eei_decadal, None,
            ax=ax, fontsize=14, case_name=case_label
        )
        # Plot IEEI and TS
        axb, axb2 = plot_ieei_ts(
            ieei_annual=ieei_annual, ieei_decadal=ieei_decadal,
            ts_annual=ts_annual, ts_decadal=ts_decadal,
            ax=axb, colors=["orange", "purple"], fontsize=14, time_dim="year",
        )

        ax1.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
        axb.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
        if "keep_left_axes" in PLOT_CONFIGS[case_label]:
            if not PLOT_CONFIGS[case_label]["keep_left_axes"]:
                ax1.set_ylabel('')
        if "keep_right_axes" in PLOT_CONFIGS[case_label]:
            if not PLOT_CONFIGS[case_label]["keep_right_axes"]:
                ax2.set_ylabel('')
                # ax3.set_ylabel('')
        # Set y-axis limits
        if "ax1_lims" in PLOT_CONFIGS[case_label]:
            ax1.set_ylim(*PLOT_CONFIGS[case_label]["ax1_lims"])
        if "ax2_lims" in PLOT_CONFIGS[case_label]:
            ax2.set_ylim(*PLOT_CONFIGS[case_label]["ax2_lims"])

        if "axb1_lims" in PLOT_CONFIGS[case_label]:
            axb.set_ylim(*PLOT_CONFIGS[case_label]["axb1_lims"])
        if "axb2_lims" in PLOT_CONFIGS[case_label]:
            axb2.set_ylim(*PLOT_CONFIGS[case_label]["axb2_lims"])

        if "axb1_yticks" in PLOT_CONFIGS[case_label]:
            axb.set_yticks(PLOT_CONFIGS[case_label]["axb1_yticks"])
        if "axb2_yticks" in PLOT_CONFIGS[case_label]:
            axb2.set_yticks(PLOT_CONFIGS[case_label]["axb2_yticks"])
        if "ax1_major_y" in PLOT_CONFIGS[case_label]:
            ax1.yaxis.set_major_locator(PLOT_CONFIGS[case_label]["ax1_major_y"])
            ax1.grid(True, alpha=0.3)

        # Add a horizontal line at y=0 for the EEI subplot
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)

    fig.savefig("figures/figure3b_toprow.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure3b_toprow.png")
    plt.close(fig)

    # %%
    PLOT_CONFIGS2 = {
        'CESM2_WACCM_SSP2-4.5': {
            "selfunc": lambda ds: ds['b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??'].mean("ens"),
            "ax1_lims": (236, 244),
            "ax2_lims": (-5, 3),
            # "axb1_lims": (0.0e24, 2.5e24),
            "axb1_lims": (-0.5e24, 2.5e24),
            "axb2_lims": (286.0, 292.0),
            "xlims": (1850, 2100),
            "keep_left_axes": True,
            "keep_right_axes": False,
        },
        "ARISE-SAI": {
            "selfunc": lambda ds: ds['b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?'].mean("ens"),
            # "selfunc": lambda ds: ds['b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?'].set_index(ens="ens").sel(ens="001"),
            "ax1_lims": (236, 244),
            "ax2_lims": (-5, 3),
            # "axb1_lims": (0.0e24, 2.5e24),
            "axb1_lims": (-0.5e24, 2.5e24),
            "axb2_lims": (286.0, 292.0),
            "xlims": (1850, 2100),
            "keep_left_axes": False,
            "keep_right_axes": False,
        },
        # "ARISE-SAI_extended": {
        #     "selfunc": lambda ds: ds['b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?'].set_index(ens="ens").sel(ens="001"),
        #     "ax1_lims": (236, 244),
        #     "ax2_lims": (-5, 3),
        #     # "axb1_lims": (0.0e24, 2.5e24),
        #     "axb1_lims": (-0.5e24, 2.5e24),
        #     "axb2_lims": (286.0, 292.0),
        #     "xlims": (1850, 2100),
        #     "keep_left_axes": False,
        #     "keep_right_axes": False,
        # },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "selfunc": lambda ds: ds['b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?'].mean("ens"),
            "ax1_lims": (236, 244),
            "ax2_lims": (-5, 3),
            # "axb1_lims": (0.0e24, 2.5e24),
            "axb1_lims": (-0.5e24, 2.5e24),
            "axb2_lims": (286.0, 292.0),
            "xlims": (1850, 2100),
            "keep_left_axes": False,
            "keep_right_axes": True,
        },
    }

    # %%
    # PLOT 2: Future scenarios (CESM2_WACCM_SSP2-4.5, ARISE-SAI, CESM2_WACCM_SSP2-4.5_MCB)
    # Integration starts from year 1850
    logging.info("Creating Plot 2: Future scenarios (CESM2_WACCM_SSP2-4.5, ARISE-SAI, CESM2_WACCM_SSP2-4.5_MCB)")
    PLOT_CONFIGS = PLOT_CONFIGS2

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(wspace=0.35)
    
    case_list_2 = ["CESM2_WACCM_SSP2-4.5", "ARISE-SAI", "CESM2_WACCM_SSP2-4.5_MCB"]
    start_year_2 = 1850
    ts_var = "TS"
    olr_var = "FLNT"
    asr_var = "FSNT"

    for ax, axb, case_label in zip(axes[0], axes[1], case_list_2):
        logging.info(f"Plotting case: {case_label}")

        # Extract ASR and OLR data
        selfunc = PLOT_CONFIGS[case_label].get("selfunc")
        asr_ds = selfunc(data_dict[case_label])[asr_var].sel(spatial="G")
        olr_ds = selfunc(data_dict[case_label])[olr_var].sel(spatial="G")
        ts_ds = selfunc(data_dict[case_label])[ts_var].sel(spatial="G")
        
        # Compute annual means
        asr_annual = asr_ds.groupby("time.year").mean()
        olr_annual = olr_ds.groupby("time.year").mean()
        ts_annual = ts_ds.groupby("time.year").mean()
        eei_annual = asr_annual - olr_annual
        
        # Compute decadal means
        asr_decadal = asr_ds.resample(time='10YE').mean().groupby("time.year").mean()
        olr_decadal = olr_ds.resample(time='10YE').mean().groupby("time.year").mean()
        ts_decadal = ts_ds.resample(time='10YE').mean().groupby("time.year").mean()
        eei_decadal = asr_decadal - olr_decadal
        
        # Compute IEEI starting from start_year_2
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, start_year_2)
        
        # Create annual and decadal means for IEEI by grouping years
        ieei_annual = ieei_ds.groupby("time.year").mean()
        ieei_decadal = ieei_ds.resample(time='10YE').mean().groupby("time.year").mean()

        # Plot
        ax1, ax2, ax3 = plot_eei_timeseries(
            asr_annual, olr_annual, eei_annual, None,
            asr_decadal, olr_decadal, eei_decadal, None,
            ax=ax, fontsize=14, case_name=case_label
        )

        # Plot IEEI and TS
        axb, axb2 = plot_ieei_ts(
            ieei_annual=ieei_annual, ieei_decadal=ieei_decadal,
            ts_annual=ts_annual, ts_decadal=ts_decadal,
            ax=axb, colors=["orange", "purple"], fontsize=14, time_dim="year",
        )

        ax1.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
        axb.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
        if "keep_left_axes" in PLOT_CONFIGS[case_label]:
            if not PLOT_CONFIGS[case_label]["keep_left_axes"]:
                ax1.set_ylabel('')
                axb.set_ylabel('')
        if "keep_right_axes" in PLOT_CONFIGS[case_label]:
            if not PLOT_CONFIGS[case_label]["keep_right_axes"]:
                ax2.set_ylabel('')
                axb2.set_ylabel('')
                # ax3.set_ylabel('')
        # Set y-axis limits
        if "ax1_lims" in PLOT_CONFIGS[case_label]:
            ax1.set_ylim(*PLOT_CONFIGS[case_label]["ax1_lims"])
        if "ax2_lims" in PLOT_CONFIGS[case_label]:
            ax2.set_ylim(*PLOT_CONFIGS[case_label]["ax2_lims"])

        if "axb1_lims" in PLOT_CONFIGS[case_label]:
            axb.set_ylim(*PLOT_CONFIGS[case_label]["axb1_lims"])
        if "axb2_lims" in PLOT_CONFIGS[case_label]:
            axb2.set_ylim(*PLOT_CONFIGS[case_label]["axb2_lims"])

        # Add a horizontal line at y=0 for the EEI subplot
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)

    fig.savefig("figures/figure3b_bottomrow.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure3b_bottomrow.png")
    plt.close(fig)
    # %%