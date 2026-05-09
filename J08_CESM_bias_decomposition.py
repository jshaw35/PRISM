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
        ds_decadal["year"] = ds_decadal["year"] - 5
    return ds_decadal


def compute_decadal2(
    ds,
    center=True,
):
    if "time" in ds.coords:
        ds_decadal = ds.rolling(time=120, min_periods=120, center=True).mean(dim="time").sel(time=ds["time"][::12])
    elif "year" in ds.coords:
        ds_decadal = ds.rolling(year=10, min_periods=10, center=True).mean(dim="year")
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
                
                # Add ensemble number as a data variable first, then expand to dimension
                ens_ds = ens_ds.assign_coords(ens=ens_number)
                # Expand the ens coordinate to a new dimension by wrapping in a new dimension
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


def plot_error_comparison(
    data_dict,
    case_label,
    control_label,
    control_case,
    subdirs,
    test_var,
    component_plot_args,
    error_components,
    case_plot_args,
    time_dim,
    xlims,
    ax=None,
    unc_gauss=True,
):
    """
    Plot error component comparison between control and simulations.

    Parameters
    ----------
    data_dict : dict
        Nested dictionary of loaded datasets
    case_label : str
        Label for the control case configuration
    control_label : str
        Label for the control simulation
    control_case : str
        Case string for the control dataset
    subdirs : dict
        Dictionary mapping simulation labels to case strings
    test_var : str
        Variable name to plot
    component_plot_args : dict
        Dictionary of plotting arguments for each error component
    error_components : list
        List of error components to plot
    case_plot_args : dict
        Dictionary of plotting arguments for each case/simulation
    time_dim : str
        Name of the time dimension
    xlims : tuple
        X-axis limits (min, max)

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()
    control = data_dict[case_label][control_label][control_case]

    for component in error_components:
        control_component_data = control[test_var].sel(error_component=component)
        
        if unc_gauss:
            control_mean = control_component_data.mean(dim=time_dim)
            control_stddev = control_component_data.std(dim=time_dim)
            low_bound = control_mean - 2 * control_stddev
            high_bound = control_mean + 2 * control_stddev
        else:
            low_bound = control_component_data.quantile(0.05, dim=time_dim)
            high_bound = control_component_data.quantile(0.95, dim=time_dim)
        ax.fill_between(
            np.arange(xlims[0], xlims[1] + 1, 1),
                low_bound,
                high_bound,
                label=f"{control_label} - {component}",
                color="black",
                linestyle="-",
                alpha=0.3,
            )

    for subdir in subdirs:
        case_str = subdirs[subdir]
        ds = data_dict[case_label][subdir][case_str]
        data = ds[test_var]
        for component in error_components:
            component_data = data.sel(error_component=component)
            if "ens" in component_data.dims:
                component_data = component_data.mean(dim="ens")
            if component == "NMSE":
                label = f"{subdir} - {component}"
                if "ens" in component_data.dims:
                    label += f" (N = {data.sizes['ens']})"
            else:
                label = None
            ax.plot(
                component_data[time_dim],
                component_data,
                label=label,
                **case_plot_args[subdir],
                **component_plot_args[component],
            )

    ax.legend()
    ax.set_xlim(xlims)

    return fig, ax


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
        "CESM2_WACCM_1850control": {
            "path": root_dir + "data/error_relativetobaseline/CESM2_WACCM_1850control/",
            "subdirs": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "CESM2_WACCM_SSP2-4.5", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5_MCB"],
            "subdir_cases": {
                "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
                "CESM2_WACCM_HIST": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?"],
                "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?"],
                "ARISE_SAI": ["1p5K-SAI.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?"],
                "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?"],
            },
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?": None,
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?": None,
                "1p5K-SAI.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?": "1p5K-SAI.00?",
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
            },
            "ufunc": None,
        },
        "CESM2(WACCM)_1850_1864": {
            "path": root_dir + "data/error_relativetobaseline/CESM2_WACCM_HIST_1850_1864/",
            "subdirs": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "CESM2_WACCM_SSP2-4.5", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5_MCB"],
            "subdir_cases": {
                "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
                "CESM2_WACCM_HIST": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?"],
                "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?"],
                "ARISE_SAI": ["1p5K-SAI.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?"],
                "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?"],
            },
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?": None,
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?": None,
                "1p5K-SAI.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?": "1p5K-SAI.00?",
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
            },
            "ufunc": None,
        },
        "CESM2(WACCM)_2000_2014": {
            "path": root_dir + "data/error_relativetobaseline/CESM2_WACCM_HIST_2000_2014/",
            "subdirs": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "CESM2_WACCM_SSP2-4.5", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5_MCB"],
            "subdir_cases": {
                "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
                "CESM2_WACCM_HIST": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?"],
                "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?"],
                "ARISE_SAI": ["1p5K-SAI.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?"],
                "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?"],
            },
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?": None,
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?": None,
                "1p5K-SAI.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?": "1p5K-SAI.00?",
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
            },
            "ufunc": None,
        },
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

                # Load case data, supporting wildcards for ensemble members
                all_ds = load_ensemble_cases(datapath_subdir, case_str, varlist)
                if all_ds is None:
                    logging.warning(f"No files found for case {case_label} with case string {case_str} in path {datapath_subdir}")
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
                    
                    has_wildcard = "*" in append_case_label or "?" in append_case_label
                    
                    for subdir_key, case_list in CASE_CONFIGS[case_label]["subdir_cases"].items():
                        if has_wildcard:
                            # Try wildcard matching
                            matches = match_wildcard_case(append_case_label, case_list)
                            if len(matches) > 0:
                                if len(matches) > 1:
                                    logging.warning(f"Append case pattern '{append_case_label}' matched multiple cases: {matches}. Using first match: {matches[0]}")
                                append_case_to_use = matches[0]
                                append_subdir = subdir_key
                                break
                        else:
                            # Exact match for non-wildcard case
                            if append_case_label in case_list:
                                append_case_to_use = append_case_label
                                append_subdir = subdir_key
                                break
                    
                    if append_subdir is None:
                        logging.warning(f"Append case {append_case_label} not found in subdir_cases for case {case_label}. Skipping append.")
                    else:
                        # Handle ensemble dimension in append case
                        if append_subdir == subdir:
                            append_candidate = subcase_dict.get(append_case_to_use)
                        else:
                            append_candidate = case_dict.get(append_subdir, {}).get(append_case_to_use)
                        
                        if append_candidate is None:
                            logging.warning(f"Append case {append_case_label} not found in loaded data for case {case_label}. Skipping append.")
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
                                    append_ds = append_candidate.isel(ens=0, drop=False)
                                    first_ens = append_ds["ens"].values[0]
                                    logging.info(f"Current case has no ensemble dimension. Using first ensemble {first_ens} from append case.")
                                else:
                                    # Neither has ensembles
                                    append_ds = append_candidate
                            
                            # Perform the append operation with time dimension selection
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
                subcase_dict[case_str] = all_ds
            case_dict[subdir] = subcase_dict
        data_dict[case_label] = case_dict
    
    # %%
    # Draft some plots to give opencode something to work with later.
    case_label = "CESM2_WACCM_1850control"
    control_label = "CESM2_WACCM_1850control"
    control_case = CASE_CONFIGS[case_label]["subdir_cases"][control_label][0]
    subdirs = {
        "CESM2_WACCM_HIST": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?",
        "CESM2_WACCM_SSP2-4.5": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
        "ARISE_SAI": 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?',
        "CESM2_WACCM_SSP2-4.5_MCB": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?",
    }
    test_var = "PRECIP_THERMO"
    test_var = "FSNS"
    component_plot_args = {
        "NMSE": {"linestyle": "solid"},
        "U": {"linestyle": "dotted"},
        "C": {"linestyle": "-."},
        "P": {"linestyle": ":"},
        # "NMSE": {"label": "Variance error", "linestyle": "--"},
        # "U": {"label": "Mean bias error", "linestyle": "dotted", },
        # "C": {"label": "Conditional bias error", "linestyle": "-."},
        # "P": {"label": "Phase error", "linestyle": ":"},
    }
    # error_components = ["NMSE", "P"]
    error_components = ["NMSE"]
    component_linestyles = ["-", "--", "-.", ":"]
    # xlims = (1850, 2100)
    # xlims = (2015, 2070)
    xlims = (1850, 2070)
    case_plot_args = {
        "ARISE_SAI": {"color": "red"},
        "CESM2_WACCM_HIST": {"color": "blue"},
        "CESM2_WACCM_SSP2-4.5": {"color": "purple"},
        "CESM2_WACCM_SSP2-4.5_MCB": {"color": "brown"},
    }
    time_dim = "year"

    plot_vars = varlist.copy()
    drop_vars = ["FLNTCLR", "PRECC", "PRECL"]
    for var in drop_vars:
        plot_vars.remove(var)
    plot_vars = [
        "FLNSC", "FLNTC", "FSNSC", "FSNTOAC",
        "FLNR", "FLNS", "FLUT", "FSNS", "FSNT", "FSNTOA",
        "LHFLX", "SHFLX",
        "CLDTOT", "PRECT", "PRECIP_THERMO",
    ]
    fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    fig.subplots_adjust(wspace=0.3)
    axs = axs.flat
    for ax, var in zip(axs, plot_vars):

        _, ax = plot_error_comparison(
            data_dict=data_dict,
            case_label=case_label,
            control_label=control_label,
            control_case=control_case,
            subdirs=subdirs,
            test_var=var,
            component_plot_args=component_plot_args,
            error_components=error_components,
            case_plot_args=case_plot_args,
            time_dim=time_dim,
            xlims=xlims,
            ax=ax,
        )
        ax.set_ylabel(f"{var} Error")
        ax.legend().remove()
    ax.legend(loc=[1.25, 0.35])
    # Remove the last subplot
    fig.delaxes(axs[-1])

    # fig.savefig("figures/figure3_draft.png", dpi=300, bbox_inches='tight')
    # logging.info("Saved figure3_draft.png")
    # plt.close(fig)

    # %%
    # Energetic change 1850 - 2070 for the most relevant variables.
    case_label = "CESM2_WACCM_1850control"
    control_label = "CESM2_WACCM_1850control"
    control_case = CASE_CONFIGS[case_label]["subdir_cases"][control_label][0]
    subdirs = {
        # "CESM2_WACCM_1850control": "b.e21.B1850.f09_g17.CMIP6-piControl.001",
        "CESM2_WACCM_HIST": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?",
        "CESM2_WACCM_SSP2-4.5": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00?",
        "ARISE_SAI": 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?',
        "CESM2_WACCM_SSP2-4.5_MCB": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?",
    }
    test_var = "PRECIP_THERMO"
    test_var = "FSNS"
    component_plot_args = {
        "NMSE": {"linestyle": "solid"},
        "U": {"linestyle": "dotted"},
        "C": {"linestyle": "-."},
        "P": {"linestyle": ":"},
        # "NMSE": {"label": "Variance error", "linestyle": "--"},
        # "U": {"label": "Mean bias error", "linestyle": "dotted", },
        # "C": {"label": "Conditional bias error", "linestyle": "-."},
        # "P": {"label": "Phase error", "linestyle": ":"},
    }
    # error_components = ["NMSE", "P"]
    error_components = ["NMSE"]
    component_linestyles = ["-", "--", "-.", ":"]
    # xlims = (1850, 2100)
    # xlims = (2015, 2070)
    xlims = (1850, 2070)
    ylims = (0, None)
    case_plot_args = {
        "ARISE_SAI": {"color": "red"},
        "CESM2_WACCM_HIST": {"color": "blue"},
        "CESM2_WACCM_SSP2-4.5": {"color": "purple"},
        "CESM2_WACCM_SSP2-4.5_MCB": {"color": "brown"},
    }
    time_dim = "year"

    plot_vars = [
        "FLNT", "FSNT", "FLNS", "FSNS",
        "SHFLX", "LHFLX", "PRECIP_THERMO",
        # "SHFLX", "LHFLX", "PRECT", "PRECIP_THERMO",
    ]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    fig.subplots_adjust(wspace=0.3)
    axs = axs.flat
    for ax, var in zip(axs, plot_vars):

        _, ax = plot_error_comparison(
            data_dict=data_dict,
            case_label=case_label,
            control_label=control_label,
            control_case=control_case,
            subdirs=subdirs,
            test_var=var,
            component_plot_args=component_plot_args,
            error_components=error_components,
            case_plot_args=case_plot_args,
            time_dim=time_dim,
            xlims=xlims,
            ax=ax,
            unc_gauss=False,
        )
        ax.set_ylim(ylims)
        ax.set_ylabel(f"{var} Error")
        ax.legend().remove()
    ax.legend(loc=[1.25, 0.35])
    # Remove the last subplot
    fig.delaxes(axs[-1])

    fig.savefig("figures/figure3b_draft.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure3b_draft.png")
    plt.close(fig)

    # %%
    # Energetic change in the last millenium
    case_label = "CESM2-LME_control"
    control_label = "CESM2_LME"
    control_case = CASE_CONFIGS[case_label]["subdir_cases"][control_label][0]
    subdirs = {
        "CESM2_LME": 'b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002',
    }
    test_var = "PRECIP_THERMO"
    test_var = "FSNS"
    component_plot_args = {
        "NMSE": {"linestyle": "solid"},
        "U": {"linestyle": "dotted"},
        "C": {"linestyle": "-."},
        "P": {"linestyle": ":"},
        # "NMSE": {"label": "Variance error", "linestyle": "--"},
        # "U": {"label": "Mean bias error", "linestyle": "dotted", },
        # "C": {"label": "Conditional bias error", "linestyle": "-."},
        # "P": {"label": "Phase error", "linestyle": ":"},
    }
    # error_components = ["NMSE", "P"]
    error_components = ["NMSE"]
    component_linestyles = ["-", "--", "-.", ":"]
    xlims = (850, 1850)
    ylims = (0, None)
    case_plot_args = {
        "CESM2_LME": {"color": "green"},
    }
    time_dim = "year"

    plot_vars = [
        "FLNT", "FSNT", "FLNS", "FSNS",
        "SHFLX", "LHFLX", "PRECIP_THERMO",
        # "SHFLX", "LHFLX", "PRECT", "PRECIP_THERMO",
    ]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    fig.subplots_adjust(wspace=0.3)
    axs = axs.flat
    for ax, var in zip(axs, plot_vars):

        _, ax = plot_error_comparison(
            data_dict=data_dict,
            case_label=case_label,
            control_label=control_label,
            control_case=control_case,
            subdirs=subdirs,
            test_var=var,
            component_plot_args=component_plot_args,
            error_components=error_components,
            case_plot_args=case_plot_args,
            time_dim=time_dim,
            xlims=xlims,
            ax=ax,
            unc_gauss=False,
        )
        ax.set_ylim(ylims)
        ax.set_ylabel(f"{var} Error")
        ax.legend().remove()
    # ax.legend(loc=[1.05, 0.35])
    # Remove the last subplot
    fig.delaxes(axs[-1])

    fig.savefig("figures/figure3paleo_draft.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure3paleo_draft.png")
    plt.close(fig)

# %%