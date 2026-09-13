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

import logging
from J15_shared_functions import (
    weighted_annualmean,
    weighted_periodmean,
    compute_decadal2,
    load_data_with_configs,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

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
             "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
            "ufunc": lambda ds: weighted_periodmean(ds, slice("2060", "2069"), account_for_leap=False),
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
        ohc_annual = weighted_annualmean(ohc_ds, account_for_leap=False)
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
            atm_annual = weighted_annualmean(atm_ds, account_for_leap=False)
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
            atm_annual = weighted_annualmean(atm_ds, account_for_leap=False)
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