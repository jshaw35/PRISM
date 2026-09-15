"""
Script with shared functions for PRISM processing and plotting functions.


"""

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path
import logging
import fnmatch
import dask


# %%
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def weighted_annualmean(
    ds,
    account_for_leap: bool = False,
):
    """
    Compute an appropriately weighted annual mean accounting for the days in each month.
    """
    assert("time" in ds.coords), "Dataset must have a time coordinate"

    ds_weightedannual = ds.groupby("time.year").map(lambda x: x.weighted(get_weights_by_month2(x["time"], account_for_leap)).mean(dim="time"))

    return ds_weightedannual


def weighted_periodmean(ds, time_slice, account_for_leap=False):
    """
    Compute mean over a time period with proper monthly weighting.
    
    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset with monthly time coordinate
    time_slice : slice
        Time slice (e.g., slice("2060", "2069"))
    account_for_leap : bool, default False
        Whether to account for leap years
    
    Returns
    -------
    xr.Dataset or xr.DataArray
        Weighted mean over the specified period
    """
    ds_subset = ds.sel(time=time_slice)
    return ds_subset.weighted(get_weights_by_month2(ds_subset["time"], account_for_leap)).mean(dim="time")


def get_weights_by_month2(
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

    month_values = time_ds.dt.month
    year_values = time_ds.dt.year

    # Vectorized selection of the appropriate weights for each time point
    if account_for_leap:
        weights = xr.where((year_values % 4 == 0), weights_leap.sel(month=month_values), weights.sel(month=month_values))
    else:
        weights = weights.sel(month=month_values)
    return weights


def compute_decadal2(
    ds,
    center=True,
):
    if "time" in ds.coords:
        ds_annual = weighted_annualmean(ds)
        ds_decadal = ds_annual.rolling(year=10, min_periods=10, center=center).mean()
    elif "year" in ds.coords:
        ds_decadal = ds.rolling(year=10, min_periods=10, center=center).mean()
    return ds_decadal


###########################################################################
# Function to crawl through a directory and list files matching a pattern
###########################################################################

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


# ============================================================================
# MULTI-ENSEMBLE LOADING FUNCTIONS (originally from J11)
# ============================================================================

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
                ens_number = int(part)
                break
        
        if ens_number is not None:
            if ens_number not in ens_dict:
                ens_dict[ens_number] = []
            ens_dict[ens_number].append(filepath)
        else:
            logging.warning(f"Could not extract ensemble number from filename: {filename}")
    
    return ens_dict


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
    matches = [case for case in case_list if fnmatch.fnmatch(case, pattern)]
    return matches


def load_ensemble_cases(
    datapath_subdir,
    case_str,
    varlist,
    identifier: str=None,
    time_dim: str="time",
):
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
            if identifier is not None:
                var_files = crawl_and_list_glob(datapath_subdir, f"**/*{case_str}*{identifier}.{var}.*nc")
            else:
                var_files = crawl_and_list_glob(datapath_subdir, f"**/*{case_str}*.{var}.*nc")
            all_files.extend(var_files)
        all_files.sort()

        if len(all_files) == 0:
            return None
        try:
            all_ds = xr.open_mfdataset(
                all_files,
                preprocess=lambda ds: ds[varlist],
                combine="nested",
                concat_dim=time_dim,
            )
        except:
            all_ds = xr.open_mfdataset(all_files)
        return all_ds
    
    else:
        # Wildcard case: find matching files, group by ensemble, load separately
        all_files = []
        for var in varlist:
            # Use case_str directly in glob pattern (it contains wildcards)
            if identifier is not None:
                var_files = crawl_and_list_glob(datapath_subdir, f"**/*{case_str}*{identifier}.{var}.*nc")
            else:
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
                    compat='no_conflicts',
                    preprocess=lambda ds: ds[varlist],
                )
                
                # Add ensemble number as a data variable first, then expand the dimension
                ens_ds = ens_ds.expand_dims({'ens': [ens_number]})
                # Handle time duplicates in ARISE-1.0 data
                if "time" in ens_ds.indexes:
                    if ens_ds.indexes['time'].has_duplicates:
                        ens_ds = ens_ds.drop_duplicates(dim='time')
                ensemble_datasets.append(ens_ds)
                
                logging.info(f"Loaded ensemble {ens_number} with {len(ens_files)} files")
            
            except Exception as e:
                logging.error(f"Error loading ensemble {ens_number}: {e}")
                logging.info(f"Files were: {ens_files}")
                continue
        
        if len(ensemble_datasets) == 0:
            logging.warning(f"No ensemble members could be loaded for pattern: {case_str}")
            return None
        
        # Concatenate all ensembles along the 'ens' dimension
        try:
            combined_ds = xr.concat(ensemble_datasets, dim='ens')
        except Exception as e:
            logging.error(f"Error concatenating ensemble datasets: {e}")
            logging.info(f"List of ensemble datasets: {ensemble_datasets}")
            return None

        return combined_ds


def load_data_with_configs(CASE_CONFIGS, varlist, year_dim="time", load_into_memory=False, **kwargs):
    """
    Load ensemble case data according to CASE_CONFIGS dictionary.
    
    Handles complex data loading workflows including:
    - Loading data from multiple cases and subcases
    - Supporting wildcard patterns for ensemble members
    - Appending data from branched simulations (e.g., ARISE-SAI from SSP2-4.5)
    - Managing ensemble dimensions across datasets
    - Applying user-defined transformation functions
    
    Args:
        CASE_CONFIGS (dict): Configuration dictionary with structure:
            {
                "case_label": {
                    "path": str,                          # Root path for case data
                    "subdir_cases": List[str],           # Case string patterns (may contain * or ?)
                    "append_cases": {                    # Mapping of case_str to append case label
                        "case_str": "append_case_label"  # or None
                    },
                    "ufunc": callable or None            # Optional transformation function
                },
                ...
            }
        varlist (List[str]): List of variable names to load (e.g., ["OHC", "TS", "FLNT"])
        year_dim (str, default "time"): Name of time dimension in datasets
    
    Returns:
        dict: Nested dictionary structure:
            data_dict[case_label][case_str] = xarray.Dataset
            Where case_label is a configuration key and case_str is a specific case pattern.
    """
    data_dict = {}
    
    for case_label in CASE_CONFIGS.keys():
        logging.info(f"Loading data for case: {case_label}")
        datapath = CASE_CONFIGS[case_label]["path"]
        case_dict = {}
        
        for case_str in CASE_CONFIGS[case_label]["subdir_cases"]:
            logging.info(f"Loading data for subcase: {case_str}")

            # Load case data, supporting wildcards for ensemble members
            all_ds = load_ensemble_cases(datapath, case_str, varlist, time_dim=year_dim, **kwargs)
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
                                    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                                        if match_bool:
                                            append_ds_ens = append_candidate.sel(ens=ens, drop=False)
                                            logging.info(f"Appending ensemble {ens} from {append_case_label}")
                                        else:
                                            append_ds_ens = append_candidate.sel(ens=append_candidate_ens_vals_first, drop=False)
                                            # Rename the ensemble value to match the current case
                                            append_ds_ens = append_ds_ens.assign_coords(ens=ens)
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
                                with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                                    append_ds = append_candidate.isel(ens=0, drop=False)
                                logging.info(f"Current case has no ensemble dimension. Using first ensemble {first_ens} from append case.")
                            else:
                                # Neither has ensembles
                                append_ds = append_candidate
                        
                        # Perform the append operation with time dimension selection
                        # Check if cftime.DatetimeNoLeap is being used and select time accordingly
                        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                            if np.issubdtype(all_ds[year_dim].dtype, np.integer):
                                # Integer year coordinates (e.g. int64)
                                append_ds_subset = append_ds.sel({year_dim: slice(None, str(all_ds[year_dim][0].values - 1))})
                            elif isinstance(append_ds[year_dim][0].dtype, object):
                                # Likely cftime objects, select using cftime-compatible method
                                append_ds_subset = append_ds.sel({year_dim:slice(None, str(all_ds[year_dim][0].dt.year.values - 1))})
                            elif isinstance(all_ds[year_dim].values[0], np.datetime64) or isinstance(all_ds[year_dim].values[0], pd.Timestamp):
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
            if load_into_memory:
                all_ds = all_ds.load()
            case_dict[case_str] = all_ds
        data_dict[case_label] = case_dict

    return data_dict


######
# Non-function shared code

title_dict = {
    "CESM2-LME": {
        "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.0??": "CESM2 LME",
    },
    "CESM2_WACCM_1850control" :{
        "b.e21.BW1850.f09_g17.CMIP6-piControl.001": "CESM2 WACCM 1850 Control",
    },
    "CESM2-WACCM-HIST": {
        "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??": "CESM2 WACCM HIST",
    },
    "CESM2_WACCM_SSP2-4.5": {
        "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": "CESM2 WACCM SSP2-4.5",
    },
    "ARISE-SAI": {
        "1p5K-SAI.0??": "ARISE-SAI 1p5K",
        "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": "ARISE-SAI-1.5",
        "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": "ARISE-SAI-1.5 EXTENDED",
    },
    "ARISE-1.0": {
        "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??": "ARISE-SAI-1.0",
        "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??": "ARISE-SAI-1.37-2045",
    },
    "CESM2_WACCM_SSP2-4.5_MCB": {
        "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": "MCB SMBB-050PCT",
        "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": "MCB CMIP6 baseline",
        "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": "MCB CMIP6-025PCT",
        "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": "MCB CMIP6-050PCT",
        "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": "MCB CMIP6-075PCT",
        "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": "MCB CMIP6-125PCT",
    },
}

# %%