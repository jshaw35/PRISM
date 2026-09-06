"""
Note: The CESM2(WACCM6) control simulation is 500 years long, but the historical members are taken from years 56, 61, and 71. The CESM2(WACCM6) ocean is not stable, however. So we cannot sample it to construct a distribution of variability characteristics.
This would also explain why the OHC is greater than the TOA EEI in the last millenium and historical simulations.

From Danabasoglu (2020):
The three members of the CESM2(WACCM6) historical simulations are initialized from Years 56, 61, and 71 of the corresponding PI control integration. The late and early start dates for the CESM2(CAM6) and CESM2(WACCM6) historical simulations simply reflect their respective PI control integration lengths and are not intended to avoid or sample any particular variability characteristics.

"""

# %%
from pathlib import Path
import os
import xarray as xr
import numpy as np
import pandas as pd
import dask

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


def compute_weighted_annual_mean(ds, account_for_leap=False):
    """
    Compute annual means weighted by days in each month.
    
    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset with monthly time coordinate
    account_for_leap : bool, default False
        Whether to account for leap years
    
    Returns
    -------
    xr.Dataset or xr.DataArray
        Annual means with proper time weighting
    """
    return ds.groupby("time.year").map(
        lambda x: x.weighted(get_weights_by_month2(x["time"], account_for_leap)).mean(dim="time")
    )


def compute_weighted_period_mean(ds, time_slice, account_for_leap=False):
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


def crawl_and_list_glob(input_dir, file_string):
    filelist = list(Path(input_dir).glob(f"**/{file_string}"))
    return [str(file) for file in filelist]


def compute_decadal2(
    ds,
    center=True,
):
    """
    Compute decadal means from annual data using a 10-year rolling window.
    
    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input data. Expected to have either 'time' (monthly) or 'year' (annual) coordinate.
        If monthly data is provided via 'time', expects 120-month (10-year) windows.
        If annual data is provided via 'year', applies 10-year rolling mean.
        Note: Input data should already be properly weighted if temporal averaging was performed.
    center : bool, default True
        If True, center the rolling window (not currently used in the function body).
    
    Returns
    -------
    xr.Dataset or xr.DataArray
        Decadal means with 'year' coordinate
    """
    if "time" in ds.coords:
        ds_decadal = ds.rolling(time=120, min_periods=120, center=True).mean(dim="time").sel(time=ds["time"][::12])
        ds_decadal["time"] = ds_decadal["time.year"]
        ds_decadal = ds_decadal.rename({"time": "year"})
    elif "year" in ds.coords:
        ds_decadal = ds.rolling(year=10, min_periods=10, center=True).mean(dim="year")
    return ds_decadal


def compute_picontrol_uncertainty(
    pi_annual,
    variable_names=None,
    branch_period=(50, 75),
    decadal_selection=None,
    detrend=True,
):
    """
    Compute annual and decadal uncertainty statistics from piControl annual data.
    
    Computes standard deviation and quantiles (0.025, 0.975) for both annual and decadal
    averages after detrending. Returns merged dataset with uncertainty values.
    
    Parameters
    ----------
    pi_annual : xr.Dataset or xr.DataArray
        Annual piControl data with 'year' dimension
    variable_names : list, optional
        List of variable names to rename with "_uncertainty" suffix. 
        If None, applies to all variables.
    branch_period : tuple, default (50, 75)
        Year range to extract as representative of branch period
    decadal_selection : slice or array-like, optional
        Selection to apply to decadal years. If None, uses all decadal years.
        Example: decadal_selection=slice(5, None, 10) selects every 10th year starting from 5
    
    Returns
    -------
    pi_all : xr.Dataset
        Merged dataset containing:
        - Uncertainty data for annual and decadal periods (quantiles and std)
        - Mean state over the branch period
    
    Notes
    -----
    Detrends the annual data before computing statistics to remove model drift.
    If decadal_selection is provided as a slice, applies it directly to year indices.
    """
    # Compute decadal from annual
    if variable_names is not None:
        pi_annual = pi_annual[variable_names]
    pi_decadal = compute_decadal2(pi_annual)
    
    # Apply decadal selection if specified
    if decadal_selection is not None:
        if isinstance(decadal_selection, slice):
            # If it's a slice, apply to indices
            pi_decadal = pi_decadal.isel(year=decadal_selection)
        else:
            # If it's array-like, use it as index selection
            pi_decadal = pi_decadal.sel(year=decadal_selection)
    
    # Compute annual and decadal after detrending
    if detrend:
        pi_annual_detrended = detrend_ds(pi_annual, dim="year", deg=1)
        pi_decadal_detrended = detrend_ds(pi_decadal, dim="year", deg=1)
    else:
        pi_annual_detrended = pi_annual
        pi_decadal_detrended = pi_decadal
    
    # Compute annual uncertainty
    quantile_vars = ["year"]
    if "ens" in pi_annual_detrended.dims:
        quantile_vars.append("ens")
    pi_annual_std = pi_annual_detrended.std(dim=quantile_vars).assign_coords(quantile=-1).expand_dims("quantile")
    pi_annual_detrended = pi_annual_detrended.chunk({"year": -1})
    pi_annual_quantiles = pi_annual_detrended.quantile([0.025, 0.975], dim=quantile_vars)
    pi_annual_unc = xr.concat([pi_annual_quantiles, pi_annual_std], dim="quantile")
    
    # Compute decadal uncertainty
    pi_decadal_std = pi_decadal_detrended.std(dim=quantile_vars).assign_coords(quantile=-1).expand_dims("quantile")
    pi_decadal_detrended = pi_decadal_detrended.chunk({"year": -1})
    pi_decadal_quantiles = pi_decadal_detrended.quantile([0.025, 0.975], dim=quantile_vars)
    pi_decadal_unc = xr.concat([pi_decadal_quantiles, pi_decadal_std], dim="quantile")
    
    # Concatenate annual and decadal along period dimension
    pi_unc_all = xr.concat([pi_annual_unc, pi_decadal_unc], dim=xr.DataArray([1, 10], dims=["period"]))
    
    # Rename variables to indicate they are uncertainty measures
    if variable_names is not None:
        rename_dict = {var: f"{var}_uncertainty" for var in variable_names}
    else:
        # If no variable names provided, rename all data variables
        rename_dict = {var: f"{var}_uncertainty" for var in pi_unc_all.data_vars}
    pi_unc_all = pi_unc_all.rename(rename_dict)
    
    # Extract branch period mean state
    pi_branch_period = pi_annual.sel(year=slice(*branch_period))
    
    # Merge uncertainty with the mean state
    pi_all = xr.merge([pi_unc_all, pi_branch_period], compat='override')
    
    return pi_all


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


def load_data_with_configs(CASE_CONFIGS, varlist, year_dim="time", **kwargs):
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
            all_ds = load_ensemble_cases(datapath, case_str, varlist, **kwargs)
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
                                            logging.warning(f"Ensemble {ens} not found in append case {append_case_label}. Using first available ensemble {append_candidate_ens_vals_first}.")
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

    return data_dict


def load_ensemble_cases(
    datapath_subdir,
    case_str,
    varlist,
    identifier: str=None,
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
                concat_dim="time",
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
            # return ensemble_datasets

        return combined_ds


def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    detrended = da - fit
    return detrended, p.polyfit_coefficients


def detrend_ds(ds, dim, deg=1):
    # Detrend all variables in a dataset
    detrended_list = []
    for _var in ds.data_vars:
        detrended_da, polyfit_coefficients = detrend_dim(ds[_var], dim=dim, deg=deg)
        detrended_da.name = _var
        detrended_list.append(detrended_da)
        polyfit_coefficients.name = _var + "_polyfit_coefficients"
        detrended_list.append(polyfit_coefficients)
    detrended_ds = xr.merge(detrended_list)

    return detrended_ds


# %%

if __name__ == "__main__":
    # Where processed OHC data is stored
    ohc_data_root = "/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/"

    # Where the output data will be stored
    output_data_root = "/glade/work/jonahshaw/PRISM_data/"
    ohc_spatial_save_root = f"{output_data_root}/spatial_maps_ohc/"
    atm_spatial_save_root = f"{output_data_root}/spatial_maps_atm/"

    # What atmospheric variables to process
    atm_varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FNNT', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL"]
    atm_derived_varlist = ['FNNT', 'PRECIP_THERMO']

    # Set config dictionaries.
    # This tells the script which cases to load and how to process them.
    # Configuration for CAM variables from the control cases (1850 control and 2015-2034 period)
    CASE_CONFIGS_CONTROL_ATM = {
        "CESM2_WACCM_1850control" :{
            "path": "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/",
            "subdir_cases": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_SSP2-4.5": {
             "path": "/gdex/data/d651045/CESM2-WACCM-SSP245/",
             "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
             "append_cases": {
                 "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
             },
             "ufunc": None,
        },
    }

    # Configuration for CAM variables from the test cases (everything using SSP2-4.5)
    CASE_CONFIGS1 = {
        "CESM2_WACCM_SSP2-4.5": {
             "path": "/gdex/data/d651045/CESM2-WACCM-SSP245/",
             "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
             "append_cases": {
                 "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
             },
             "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "ARISE-SAI": {
            "path": "/gdex/data/d651059/ARISE-SAI-1.5/",
            "subdir_cases": [
                "1p5K-SAI.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??",
            ],
            "append_cases": {
                "1p5K-SAI.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "ARISE-1.0": {
            "path": "/glade/work/jonahshaw/PRISM_data/ARISE-1.0/",
            "subdir_cases": [
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??",
            ],
            "append_cases": {
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "path": "/gdex/data/d314006/",
            "subdir_cases": [
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000",
            ],
            "append_cases": {
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
    }

    # Configs for loading OHC data
    # Control cases
    CASE_CONFIGS_CONTROL_OCN = {
        "CESM2_WACCM_1850control" :{
            "path": ohc_data_root + "CESM2_WACCM_1850control/",
            "subdir_cases": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_SSP2-4.5": {
            "path": ohc_data_root + "CESM2_WACCM_SSP2-4.5/",
            "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
            "append_cases": {
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
            },
            "ufunc": lambda ds: ds.sel(time=slice("2015", "2034")),
        },
    }
    # Test cases
    CASE_CONFIGS2 = {
        "CESM2_WACCM_SSP2-4.5": {
            "path": ohc_data_root + "CESM2_WACCM_SSP2-4.5/",
            "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
            "append_cases": {
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "ARISE-SAI": {
            "path": ohc_data_root + "ARISE_SAI/",
            "subdir_cases": [
                "1p5K-SAI.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??",
            ],
            "append_cases": {
                "1p5K-SAI.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "ARISE-1.0": {
            "path": ohc_data_root + "ARISE-1.0/",
            "subdir_cases": [
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??",
            ],
            "append_cases": {
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "path": ohc_data_root + "CESM2_WACCM_SSP2-4.5_MCB/",
            "subdir_cases": [
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000"
            ],
            "append_cases": {
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
    }
    # Load configs must be repeated for the derived variables compute from CAM.
    CASE_CONFIGS_CONTROL_ATM_DERIVED = {
        "CESM2_WACCM_1850control" :{
            "path": "/glade/work/jonahshaw/PRISM_data/derived_vars/CESM2_WACCM_1850control/",
            "subdir_cases": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_SSP2-4.5": {
            "path": "/glade/work/jonahshaw/PRISM_data/derived_vars/CESM2_WACCM_SSP2-4.5/",
            "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
            "append_cases": {
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
            },
            "ufunc": None,
        },
    }

    # Configuration for CAM variables from the test cases (everything using SSP2-4.5)
    CASE_CONFIGS1_DERIVED = {
        "CESM2_WACCM_SSP2-4.5": {
            "path": "/glade/work/jonahshaw/PRISM_data/derived_vars/CESM2_WACCM_SSP2-4.5/",
            "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
            "append_cases": {
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "ARISE-SAI": {
            "path": "/glade/work/jonahshaw/PRISM_data/derived_vars/ARISE_SAI/",
            "subdir_cases": [
                "1p5K-SAI.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??",
            ],
            "append_cases": {
                "1p5K-SAI.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "ARISE-1.0": {
            "path": "/glade/work/jonahshaw/PRISM_data/derived_vars/ARISE-1.0/",
            "subdir_cases": [
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??",
            ],
            "append_cases": {
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??": None,
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "path": "/glade/work/jonahshaw/PRISM_data/derived_vars/CESM2_WACCM_SSP2-4.5_MCB/",
            "subdir_cases": [
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000",
            ],
            "append_cases": {
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": None,
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": None,
            },
            "ufunc": lambda ds: compute_weighted_period_mean(ds, slice("2060", "2069"), account_for_leap=False),
        },
    }

    # %%
    # This config dict tells the function how to compute uncertainty for each control case.
    # e.g. From what period to sample, if detrending should be done, etc.
    # Configure uncertainty settings for controls
    CONFIG_CONTROL_UNCERTAINTY = {
        "CESM2_WACCM_1850control": {
            "case": "b.e21.BW1850.f09_g17.CMIP6-piControl.001",
            "branch_period": (50, 75),
            "decadal_selection": slice(5, None, 10),
            "**args": {},
        },
        "CESM2_WACCM_SSP2-4.5": {
            "case": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
            "branch_period": (None, None),
            "decadal_selection": None,
            "**args": {"detrend": False},
        },
    }
    # Only different here is branch period, keeping in case.
    # CONFIG_CONTROL_UNCERTAINTY_ATM = {
    #     "CESM2_WACCM_1850control": {
    #         "case": "b.e21.BW1850.f09_g17.CMIP6-piControl.001",
    #         "branch_period": (None, None),
    #         "decadal_selection": slice(5, None, 10),
    #         "**args": {"detrend": True},
    #     },
    #     "CESM2_WACCM_SSP2-4.5": {
    #         "case": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #         "branch_period": (None, None),
    #         "decadal_selection": None,
    #         "**args": {"detrend": False},
    #     },
    # }

    # %%
    # Compute uncertainty for the control members using the generalized function
    var = "OHC"
    ohc_varlist = [var]
    load_config_dict = CASE_CONFIGS_CONTROL_OCN
    uncertainty_config_dict = CONFIG_CONTROL_UNCERTAINTY
    assert load_config_dict.keys() == uncertainty_config_dict.keys(), "Keys in load_config_dict and uncertainty_config_dict do not match."
    year_dim = "time"
    for case_label, config in uncertainty_config_dict.items():
        case = config["case"]
        save_dir = Path(ohc_spatial_save_root) / case_label
        save_file = f"{case}.{var}.spatial_uncertainty.nc"
        save_path = save_dir / save_file
        if os.path.exists(save_path):
            logging.info(f"{save_path} exists. Skipping")
            continue
        ohc_control_dict = load_data_with_configs(
            {case_label: load_config_dict[case_label]},
            ohc_varlist,
            year_dim=year_dim,
        )
        ohc_ds = ohc_control_dict[case_label][case]
        # Detrend the OHC time series by removing the linear trend, which is a common practice to account for model drift in control simulations. This will help ensure that the confidence intervals reflect internal variability rather than long-term trends.
        ohc_annual = compute_weighted_annual_mean(ohc_ds, account_for_leap=False)
        del ohc_ds # Free memory
        ohc_annual = ohc_annual.compute()
        
        # Compute uncertainty using the generalized function
        ohc_all = compute_picontrol_uncertainty(
            ohc_annual,
            variable_names=["OHC", "OHC_global_mean"],
            branch_period=config["branch_period"],
            decadal_selection=config["decadal_selection"],
            **config["**args"],
        )
        logging.info(f"Starting compute")
        ohc_all = ohc_all.compute()
        os.makedirs(save_dir, exist_ok=True)
        ohc_all.to_netcdf(save_path)

    # %%
    # For the future scenarios, compute the average fields over the 2060-2069 period.
    ohc_varlist = ["OHC"]
    year_dim = "time"
    load_config_dict = CASE_CONFIGS2
    for scenario in load_config_dict.keys():
        # Only load the data for the current scenario.
        ohc_dict = load_data_with_configs({scenario: load_config_dict[scenario]}, ohc_varlist, year_dim=year_dim)
        save_dir = Path(ohc_spatial_save_root) / scenario
        os.makedirs(save_dir, exist_ok=True)
        for case_str, ds in ohc_dict[scenario].items():
            save_file = f"{case_str}.{var}.spatial_mean2060_2069.nc"
            save_path = save_dir / save_file
            if os.path.exists(save_path):
                logging.info(f"{save_path} exists. Skipping")
            else:
                logging.info(f"Saving to {save_path}")
                ds.to_netcdf(save_path) # Implicitly calls compute() here, but perhaps not as efficient?

    # Clean up
    del ohc_dict
    del ds

    # %%
    # Apply to the ATM variables
    # Process each variable separately for memory reasons.

    # Compute the uncertainty for each ATM variable from the control period.
    load_config_dict = CASE_CONFIGS_CONTROL_ATM
    uncertainty_config_dict = CONFIG_CONTROL_UNCERTAINTY
    year_dim = "time"
    for var in atm_varlist:
        logging.info(f"Processing {var}")
        for case_label, config in uncertainty_config_dict.items():
            save_dir = Path(atm_spatial_save_root) / case_label
            case = config["case"]
            save_file = f"{case}.{var}.spatial_uncertainty.nc"
            save_path = save_dir / save_file
            if os.path.exists(save_path):
                logging.info(f"{save_path} exists. Skipping")
                continue
            # Load the control data for only the current variable and case.
            var_data_dict = load_data_with_configs(
                {case_label: load_config_dict[case_label]},
                [var],
                year_dim=year_dim,
                identifier="h0",
            )
            if len(var_data_dict[case_label]) == 0:
                logging.warning(f"No data found for case {case_label} and variable {var}")
                continue
            atm_ds = var_data_dict[case_label][case]

            # Detrend the ATM time series by removing the linear trend, which is a common practice to account for model drift in control simulations. This will help ensure that the confidence intervals reflect internal variability rather than long-term trends.
            atm_annual = compute_weighted_annual_mean(atm_ds, account_for_leap=False)
            del atm_ds # Free memory
            atm_annual = atm_annual[[var]].compute()

            # Compute uncertainty using the generalized function
            atm_all = compute_picontrol_uncertainty(
                atm_annual,
                variable_names=[var],
                branch_period=config["branch_period"],
                decadal_selection=config["decadal_selection"],
                **config["**args"],
            )
            logging.info(f"Starting compute")
            atm_all = atm_all.compute()
            os.makedirs(save_dir, exist_ok=True)
            atm_all.to_netcdf(save_path)

    # %%
    # Compute the 2060 - 2069 spatial means
    load_config_dict = CASE_CONFIGS1
    for var in atm_varlist:
        logging.info(f"Processing {var}")
        # for scenario in future_scenarios:
        for scenario in load_config_dict.keys():
            var_data_dict = load_data_with_configs(
                {scenario: load_config_dict[scenario]},
                [var],
                year_dim=year_dim,
                identifier="h0",
            )
            save_dir = Path(atm_spatial_save_root) / scenario
            os.makedirs(save_dir, exist_ok=True)
            for case_str, ds in var_data_dict[scenario].items():
                save_file = f"{case_str}.{var}.spatial_mean2060_2069.nc"
                save_path = save_dir / save_file
                if os.path.exists(save_path):
                    logging.info(f"{save_path} exists. Skipping")
                else:
                    logging.info(f"Saving to {save_path}")
                    ds.to_netcdf(save_path)

    # %%
    # Now repeat the process for the derived ATM variables.
    # Compute the uncertainty for each ATM variable from the control period.
    load_config_dict = CASE_CONFIGS_CONTROL_ATM_DERIVED
    uncertainty_config_dict = CONFIG_CONTROL_UNCERTAINTY
    year_dim = "time"
    for var in atm_derived_varlist:
        logging.info(f"Processing {var}")
        for case_label, config in uncertainty_config_dict.items():
            save_dir = Path(atm_spatial_save_root) / case_label
            case = config["case"]
            save_file = f"{case}.{var}.spatial_uncertainty.nc"
            save_path = save_dir / save_file
            if os.path.exists(save_path):
                logging.info(f"{save_path} exists. Skipping")
                continue
            # Load the control data for only the current variable and case.
            var_data_dict = load_data_with_configs(
                {case_label: load_config_dict[case_label]},
                [var],
                year_dim=year_dim,
                identifier="h0",
            )
            if len(var_data_dict[case_label]) == 0:
                logging.warning(f"No data found for case {case_label} and variable {var}")
                continue
            atm_ds = var_data_dict[case_label][case]

            # Detrend the ATM time series by removing the linear trend, which is a common practice to account for model drift in control simulations. This will help ensure that the confidence intervals reflect internal variability rather than long-term trends.
            atm_annual = compute_weighted_annual_mean(atm_ds, account_for_leap=False)
            del atm_ds # Free memory
            atm_annual = atm_annual[[var]].compute()

            # Compute uncertainty using the generalized function
            atm_all = compute_picontrol_uncertainty(
                atm_annual,
                variable_names=[var],
                branch_period=config["branch_period"],
                decadal_selection=config["decadal_selection"],
                **config["**args"],
            )
            logging.info(f"Starting compute")
            atm_all = atm_all.compute()
            os.makedirs(save_dir, exist_ok=True)
            atm_all.to_netcdf(save_path)

    # %%
    # Compute the 2060 - 2069 spatial means
    load_config_dict = CASE_CONFIGS1_DERIVED
    for var in atm_derived_varlist:
        logging.info(f"Processing {var}")
        # for scenario in future_scenarios:
        for scenario in load_config_dict.keys():
            var_data_dict = load_data_with_configs(
                {scenario: load_config_dict[scenario]},
                [var],
                year_dim=year_dim,
                identifier="h0",
            )
            save_dir = Path(atm_spatial_save_root) / scenario
            os.makedirs(save_dir, exist_ok=True)
            for case_str, ds in var_data_dict[scenario].items():
                save_file = f"{case_str}.{var}.spatial_mean2060_2069.nc"
                save_path = save_dir / save_file
                if os.path.exists(save_path):
                    logging.info(f"{save_path} exists. Skipping")
                else:
                    logging.info(f"Saving to {save_path}")
                    ds.to_netcdf(save_path)
    # %%