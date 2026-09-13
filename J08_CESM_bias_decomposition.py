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
import pandas as pd
import logging

from J15_shared_functions import (
    shift_noleap_time_back_one_month,
    load_ensemble_cases,
    match_wildcard_case,
    load_data_with_configs,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

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
    data_root = "/glade/work/jonahshaw/PRISM_data/error_relativetobaseline_atm/CESM2_WACCM_1850control_0100_0499/"
    CASE_CONFIGS_ATM = {
        "CESM2_WACCM_1850control" :{
            "path": f"{data_root}/CESM2_WACCM_1850control/",
            "subdir_cases": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
            "append_cases": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
            },
            "ufunc": None,
        },
        "CESM2-WACCM-HIST": {
            "path": f"{data_root}/CESM2_WACCM_HIST/",
            "subdir_cases": [
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??",
            ],
            "append_cases": {
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_SSP2-4.5": {
             "path": f"{data_root}/CESM2_WACCM_SSP2-4.5/",
             "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
             "append_cases": {
                 "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
             },
             "ufunc": None,
        },
        "ARISE-SAI": {
            "path": f"{data_root}/ARISE_SAI/",
            "subdir_cases": [
                "1p5K-SAI.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??",
            ],
            "append_cases": {
                "1p5K-SAI.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??",
                # "1p5K-SAI.0??": None,
                # "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": None,
                # "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": None,
            },
            "ufunc": None,
        },
        "ARISE-1.0": {
            "path": f"{data_root}/ARISE-1.0/",
            "subdir_cases": [
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??",
            ],
            "append_cases": {
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                # "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??": None,
                # "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??": None,
            },
            "ufunc": None,
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "path": f"{data_root}/CESM2_WACCM_SSP2-4.5_MCB/",
            "subdir_cases": [
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000",
            ],
            "append_cases": {
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
                # "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": None,
                # "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": None,
                # "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": None,
                # "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": None,
                # "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": None,
                # "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": None,
            },
            "ufunc": None,
        },
    }

    # %%
    # Load the data using the generalized loading function
    # data_varlist = ['FLNT', 'FSNT', 'TS']
    data_varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO", "FNNT"]
    year_dim = "year"
    ohc_varlist = ["OHC", "OHC_global_mean"]

    # Load data lazily using the generalized function
    data_dict = {}
    for var in data_varlist:
        data_dict[var] = load_data_with_configs(CASE_CONFIGS_ATM, [var], year_dim=year_dim, load_into_memory=False)

    # %%
    # root_dir = "/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/"
    # CASE_CONFIGS = {
    #     "CESM2-LME_control": {
    #         "path": root_dir / "CESM2_LME_control/",
    #         "subdirs": ["CESM2_LME"],
    #         "subdir_cases": {"CESM2_LME": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008", "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002"]},
    #         "append_cases": {
    #             "b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008": None,
    #             "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002": None,
    #         },
    #         "ufunc": None,
    #     },
    #     "CESM2_WACCM_1850control": {
    #         "path": root_dir / "CESM2_WACCM_1850control/",
    #         "subdirs": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "CESM2_WACCM_SSP2-4.5", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5_MCB"],
    #         "subdir_cases": {
    #             "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
    #             "CESM2_WACCM_HIST": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??"],
    #             "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
    #             "ARISE_SAI": ["1p5K-SAI.0??", "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??"],
    #             "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??"],
    #         },
    #         "append_cases": {
    #             "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
    #             "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??": None,
    #             "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
    #             "1p5K-SAI.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": "1p5K-SAI.0??",
    #             "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #         },
    #         "ufunc": None,
    #     },
    #     "CESM2(WACCM)_1850_1864": {
    #         "path": root_dir / "CESM2_WACCM_HIST_1850_1864/",
    #         "subdirs": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "CESM2_WACCM_SSP2-4.5", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5_MCB"],
    #         "subdir_cases": {
    #             "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
    #             "CESM2_WACCM_HIST": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??"],
    #             "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
    #             "ARISE_SAI": ["1p5K-SAI.0??", "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??"],
    #             "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??"],
    #         },
    #         "append_cases": {
    #             "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
    #             "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??": None,
    #             "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
    #             "1p5K-SAI.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": "1p5K-SAI.0??",
    #             "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #         },
    #         "ufunc": None,
    #     },
    #     "CESM2(WACCM)_2000_2014": {
    #         "path": root_dir / "CESM2_WACCM_HIST_2000_2014/",
    #         "subdirs": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "CESM2_WACCM_SSP2-4.5", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5_MCB"],
    #         "subdir_cases": {
    #             "CESM2_WACCM_1850control": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
    #             "CESM2_WACCM_HIST": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??"],
    #             "CESM2_WACCM_SSP2-4.5": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
    #             "ARISE_SAI": ["1p5K-SAI.0??", "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??", "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??"],
    #             "CESM2_WACCM_SSP2-4.5_MCB": ["b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??"],
    #         },
    #         "append_cases": {
    #             "b.e21.BW1850.f09_g17.CMIP6-piControl.001": None,
    #             "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??": None,
    #             "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": None,
    #             "1p5K-SAI.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": "1p5K-SAI.0??",
    #             "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
    #         },
    #         "ufunc": None,
    #     },
    # }

    # %%
    # Load the data in a nested dictionary structure. The top level keys are the control case labels (e.g. "CESM2-LME", what is being used as the baseline for the error calculation). The second level keys are the simulations that are being tested against, and the third level keys are the specific case strings that are being used to identify the files for each simulation.
    data_dict = {}
    varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO", "FNNT"]
    LOAD_CONFIGS = CASE_CONFIGS_ATM
    year_dim = "year"
    for case_label in LOAD_CONFIGS.keys():
        datapath = LOAD_CONFIGS[case_label]["path"]
        case_dict = {}
        for subdir in LOAD_CONFIGS[case_label]["subdir_cases"]:
            subcase_dict = {}
            datapath_subdir = os.path.join(datapath, subdir)
            for case_str in LOAD_CONFIGS[case_label]["subdir_cases"][subdir]:

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
                if LOAD_CONFIGS[case_label]["append_cases"][case_str] is not None:
                    append_case_label = LOAD_CONFIGS[case_label]["append_cases"][case_str]
                    
                    # Get the subdir for the append case, which may be different from the current subdir.
                    # Handle both explicit case strings and wildcard patterns
                    append_subdir = None
                    append_case_to_use = None
                    
                    has_wildcard = "*" in append_case_label or "?" in append_case_label
                    
                    for subdir_key, case_list in LOAD_CONFIGS[case_label]["subdir_cases"].items():
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

                if LOAD_CONFIGS[case_label]["ufunc"] is not None:
                    all_ds = LOAD_CONFIGS[case_label]["ufunc"](all_ds)
                subcase_dict[case_str] = all_ds
            case_dict[subdir] = subcase_dict
        data_dict[case_label] = case_dict
    
    # %%
    # Draft some plots to give opencode something to work with later.
    case_label = "CESM2_WACCM_1850control"
    control_label = "CESM2_WACCM_1850control"
    control_case = CASE_CONFIGS[case_label]["subdir_cases"][control_label][0]
    subdirs = {
        "CESM2_WACCM_HIST": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??",
        "CESM2_WACCM_SSP2-4.5": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
        "ARISE_SAI": 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??',
        "CESM2_WACCM_SSP2-4.5_MCB": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
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
        "CESM2_WACCM_HIST": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??",
        "CESM2_WACCM_SSP2-4.5": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
        "ARISE_SAI": 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??',
        "CESM2_WACCM_SSP2-4.5_MCB": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
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