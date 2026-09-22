"""
Plot spatial maps of change relative to the reference period for the relevant energetic variables.

To-do's:
- Report the NMSE (and components in addition to the significant fraction)
- Include integrated EEI maps and ocean heat content change maps to show where the energy is going.

"""
# %%
from pathlib import Path
import glob
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
import seaborn as sns
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import logging
from copy import deepcopy

from J15_shared_functions import (
    load_data_with_configs,
    compute_decadal2,
    title_dict,
    CASE_CONFIGS_TEMPLATE,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)


def plot_difference_map(
    data_dict: dict,
    unc_data_dict: dict,
    var_name: str,
    control_name: str,
    control_case: str,
    test_name: str,
    test_case: str,
    control_ufunc: callable = None,
    test_ufunc: callable = None,
    uncertainty: bool = False,
    uncertainty_detrended: bool = True,
    uncertainty_tsel: slice = None,
    detrend_PIC: bool = False,
    ax = None,
    cmap = "viridis",
    vlims = None,
    colorbar: bool = False,
    cax = None,
    plt_kwargs = {},
    cbar_kwargs = {},
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    control_data = unc_data_dict[control_name][control_case][var_name]
    if control_ufunc is not None:
        control_data = control_ufunc(control_data)

    test_data = data_dict[test_name][test_case][var_name]
    if test_ufunc is not None:
        test_data = test_ufunc(test_data)
    difference_data = test_data - control_data
    assert difference_data.shape == control_data.shape, f"Difference data shape does not match control data shape. test shape: {test_data.shape}, control shape: {control_data.shape}"

    # If cmap and vlims are provided, create a diverging colormap centered on zero.
    norm = None
    if cmap is not None and vlims is not None:
        norm = TwoSlopeNorm(vmin=vlims[0], vcenter=0, vmax=vlims[1])

        im = ax.pcolormesh(
            difference_data.lon,
            difference_data.lat,
            difference_data,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            **plt_kwargs,
        )
    else:
        im = ax.pcolormesh(
            difference_data.lon,
            difference_data.lat,
            difference_data,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            vmin=vlims[0] if vlims is not None else None,
            vmax=vlims[1] if vlims is not None else None,
            **plt_kwargs,
        )

    # Create a mask indicating where the difference is not statistically significant, if uncertainty is True.
    if uncertainty:
        if uncertainty_detrended:
            uncertainty_data = unc_data_dict[control_name][control_case][var_name + "_uncertainty"].sel(period=10)
            uncertainty_low = control_data + uncertainty_data.sel(quantile=0.025)
            uncertainty_high = control_data + uncertainty_data.sel(quantile=0.975)
        else:
            uncertainty_data = unc_data_dict[control_name][control_case][var_name].sel(year=uncertainty_tsel)
            quantile_vars = ["year"]
            if "ens" in uncertainty_data.dims:
                quantile_vars.append("ens")
            # Compute the decadal mean
            uncertainty_data = compute_decadal2(uncertainty_data)
            # Take non-overlapping timesteps
            uncertainty_data = uncertainty_data.sel(year=uncertainty_data.year[5::10])
            uncertainty_low = uncertainty_data.quantile(0.025, dim=quantile_vars)
            uncertainty_high = uncertainty_data.quantile(0.975, dim=quantile_vars)
        significance_mask = (test_data < uncertainty_low) | (test_data > uncertainty_high)
        ax.contourf(
            difference_data.lon,
            difference_data.lat,
            ~significance_mask,
            levels=[0, 0.5, 1.5],
            colors='none',
            hatches=['', '///'],
            transform=ccrs.PlateCarree(),
        )
        # Compute the weighted fraction of the globe that is significant:
        significance_mask_binary = significance_mask.astype(int)
        significant_fraction = significance_mask_binary.weighted(np.cos(np.deg2rad(difference_data.lat))).mean()
        # ax.set_title(f"Significant fraction: {significant_fraction.values:.2%}")
        ax.set_title(f"{significant_fraction.values:.2%}", y=0.96, loc="center")

    if colorbar:
        if cax is None:
            cbar = plt.colorbar(im, ax=ax, **cbar_kwargs)
        else:
            cbar = plt.colorbar(im, cax=cax, **cbar_kwargs)
        # if cbar_label is not None:
            # cbar.set_label(cbar_label)

    return ax


# %%

if __name__ == "__main__":

    data_root = (
        f"/glade/work/jonahshaw/PRISM_data/spatial_maps_atm/"
    )
    CASE_CONFIGS_ATM = deepcopy(CASE_CONFIGS_TEMPLATE)
    # Remove cases that are not used here
    CASE_CONFIGS_ATM.pop("CESM2-LM", None)
    # CASE_CONFIGS_ATM.pop("CESM2_WACCM_1850control", None)
    CASE_CONFIGS_ATM.pop('CESM2-WACCM-HIST', None)

    for config in CASE_CONFIGS_ATM.values():
        config["path"] = str(Path(data_root) / config.pop("path"))

    # %%
    # Load the data using the generalized loading function
    data_varlist = ["FNNT", "FLNT", "FSNT", "FNNT", "FLNS", "FSNS", "TS", "PRECIP_THERMO", "LHFLX", "PRECT"]
    # data_varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO", "FNNT"]
    year_dim = "year"

    # Load data lazily using the generalized function
    # data_dict = {}
    # for var in data_varlist:
    #     data_dict[var] = load_data_with_configs(CASE_CONFIGS_ATM, [var], year_dim=year_dim, load_into_memory=False)

    # %%
    # spatial_proc_dir = "/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/spatial_maps/"
    # CASE_CONFIGS1 = {
    #     "CESM2_WACCM_1850control" :{
    #         "path": spatial_proc_dir + "CESM2_WACCM_1850control/",
    #         "subdir_cases": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
    #     },
    #     "CESM2_WACCM_SSP2-4.5": {
    #         "path": spatial_proc_dir + "CESM2_WACCM_SSP2-4.5/",
    #         "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
    #     },
    #     "ARISE-SAI": {
    #         "path": spatial_proc_dir + "ARISE-SAI/",
    #         "subdir_cases": [
    #             "1p5K-SAI.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??",
    #             "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??",
    #         ],
    #     },
    #     "CESM2_WACCM_SSP2-4.5_MCB": {
    #         "path": spatial_proc_dir + "CESM2_WACCM_SSP2-4.5_MCB/",
    #         "subdir_cases": [
    #             "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
    #             "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000",
    #             "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000",
    #             "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000",
    #             "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000",
    #             "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000",
    #         ],
    #     },
    # }
    # %%
    CASE_CONFIGS = CASE_CONFIGS_ATM
    # CASE_CONFIGS = CASE_CONFIGS1
    mean_data_dict = {}
    uncertainty_data_dict = {}
    
    for case_label in CASE_CONFIGS.keys():
        logging.info(f"Loading data for case: {case_label}")
        datapath = CASE_CONFIGS[case_label]["path"]
        mean_case_dict = {}
        uncertainty_case_dict = {}
        
        for case_str in CASE_CONFIGS[case_label]["subdir_cases"]:
            logging.info(f"Loading data for subcase: {case_str}")
            spatial_files = glob.glob(os.path.join(datapath, case_str + "*spatial_uncertainty.nc"))
            logging.info(f"Found {len(spatial_files)} files for subcase {case_str}")
            if len(spatial_files) == 0:
                logging.warning(f"No files found for subcase {case_str} in path {datapath}")
                # continue
            else:
                spatial_subds = xr.open_mfdataset(spatial_files, combine="by_coords")
                uncertainty_case_dict[case_str] = spatial_subds
            spatial_files = glob.glob(os.path.join(datapath, case_str + "*spatial_mean2060_2069.nc"))
            logging.info(f"Found {len(spatial_files)} files for subcase {case_str}")
            if len(spatial_files) == 0:
                logging.warning(f"No files found for subcase {case_str} in path {datapath}")
                # continue
            else:
                spatial_subds = xr.open_mfdataset(spatial_files, combine="by_coords")
                mean_case_dict[case_str] = spatial_subds
        if mean_case_dict: mean_data_dict[case_label] = mean_case_dict
        if uncertainty_case_dict: uncertainty_data_dict[case_label] = uncertainty_case_dict

# %%
# Draft plotting code.

PLOT_CONFIGS1 = {
    "CESM2_WACCM_SSP2-4.5": {
        "case": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
        "ufunc": lambda ds: ds.mean("ens"),
        "args": {"colorbar": True},
    },
    "ARISE-SAI": {
        "case": "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??",
        "ufunc": lambda ds: ds.mean("ens"),
        "args": {"colorbar": False},
    },
    "CESM2_WACCM_SSP2-4.5_MCB": {
        "case": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
        "ufunc": lambda ds: ds.mean("ens"),
        "args": {"colorbar": False},
    },
}

PLOT_VAR_CONFIGS = {
    "FNNT": {
        "cmap": "bwr",
        "vlims": [-40, 40],
        "cbar_kwargs": {"extend": "both", "label": "Wm$^2$"},
    },
    "FLNT": {
        "cmap": "bwr",
        "vlims": [-45, 45],
        "cbar_kwargs": {"extend": "both", "label": "Wm$^2$"},
    },
    "FSNT": {
        "cmap": "bwr",
        "vlims": [-60, 60],
        "cbar_kwargs": {"extend": "both", "label": "Wm$^2$"}
    },
    "FLNS": {
        "cmap": "bwr",
        "vlims": [-35, 35],
        "cbar_kwargs": {"extend": "both", "label": "Wm$^2$"}
    },
    "FSNS": {
        "cmap": "bwr",
        "vlims": [-100, 50],
        "cbar_kwargs": {"extend": "both", "label": "Wm$^2$"}
    },
    "SHFLX": {
        "cmap": "bwr",
        "vlims": [-20, 20],
        "cbar_kwargs": {"extend": "both", "label": "Wm$^2$"}
    },
    "LHFLX": {
        "cmap": "bwr",
        "vlims": [-50, 50],
        "cbar_kwargs": {"extend": "both", "label": "Wm$^2$"}
    },
    "PRECIP_THERMO": {
        "cmap": "BrBG",
        "vlims": [-2, 2],
        "cbar_kwargs": {"extend": "both", "label": "mm/day"}
    }
}

# %%
energy_vars = ["FNNT", "FLNT", "FSNT", "FLNS", "FSNS", "SHFLX", "LHFLX", "PRECIP_THERMO"]

# Plot relative to the 1850 control case
control_name = "CESM2_WACCM_1850control"
control_case = "b.e21.BW1850.f09_g17.CMIP6-piControl.001"
control_ufunc = lambda ds: ds.mean("year")
control_detrended = True
control_tsel = None

fig, axs = sp_map(len(energy_vars), 3,  projection=ccrs.Robinson(), figsize=(12, 19))
axes = axs.T

# Add a cbar axis for each row
cbar_axes = []
for i in range(len(energy_vars)):
    cbar_ax = fig.add_axes([0.92, 0.11 + i*0.098, 0.02, 0.09])
    cbar_axes.append(cbar_ax)

for label, axs in zip(PLOT_CONFIGS1, axes):
    case_config = PLOT_CONFIGS1[label]
    test_case = case_config["case"]
    test_ufunc = case_config["ufunc"]

    for i, (var_name, ax) in enumerate(zip(energy_vars, axs)):
        print(f"Plotting {var_name} for case {label}")

        ax_out = plot_difference_map(
            data_dict=mean_data_dict,
            unc_data_dict=uncertainty_data_dict,
            var_name=var_name,
            control_name=control_name,
            control_case=control_case,
            control_ufunc=control_ufunc,
            test_name=label,
            test_case=case_config["case"],
            test_ufunc=case_config["ufunc"],
            uncertainty=True,
            uncertainty_detrended=control_detrended,
            uncertainty_tsel=control_tsel,
            detrend_PIC=False,
            ax=ax,
            cmap=PLOT_VAR_CONFIGS[var_name]["cmap"],
            vlims=PLOT_VAR_CONFIGS[var_name]["vlims"],
            cax=cbar_axes[-1 - i],
            cbar_kwargs=PLOT_VAR_CONFIGS[var_name]["cbar_kwargs"],
            **case_config["args"],
        )

        ax_out.coastlines()
        ax_out.set_global()

# Label the top of each column with the case name
for ax, label in zip(axes[:, 0], PLOT_CONFIGS1.keys()):
    plt.text(0.5, 1.15, label, transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

# Label the left of each row with the variable name
for ax, var_name in zip(axes[0, :], energy_vars):
    plt.text(-0.1, 0.5, var_name, transform=ax.transAxes, ha='right', va='center', fontsize=10, rotation=90)

# %%
fig.savefig("figures/figure4_ensmean_1850.png", dpi=300, bbox_inches='tight')
logging.info("Saved figure4_ensmean_1850.png")
plt.close(fig)
# %%

energy_vars = ["FNNT", "FLNT", "FSNT", "FLNS", "FSNS", "SHFLX", "LHFLX", "PRECIP_THERMO"]

# Plot relative to the 2015-2034 period of the SSP2-4.5 case
control_name = 'CESM2_WACCM_SSP2-4.5'
control_case = 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??'
control_ufunc = lambda ds: ds.sel(year=slice(2015, 2034)).mean(dim=["year", "ens"])
control_detrended = False
control_tsel = slice(2015, 2034)

fig, axs = sp_map(len(energy_vars), 3,  projection=ccrs.Robinson(), figsize=(12, 19))
axes = axs.T

# Add a cbar axis for each row
cbar_axes = []
for i in range(len(energy_vars)):
    cbar_ax = fig.add_axes([0.92, 0.11 + i*0.098, 0.02, 0.09])
    cbar_axes.append(cbar_ax)

for label, axs in zip(PLOT_CONFIGS1, axes):
    case_config = PLOT_CONFIGS1[label]
    test_case = case_config["case"]
    test_ufunc = case_config["ufunc"]

    for i, (var_name, ax) in enumerate(zip(energy_vars, axs)):
        print(f"Plotting {var_name} for case {label}")

        ax_out = plot_difference_map(
            data_dict=mean_data_dict,
            unc_data_dict=uncertainty_data_dict,
            var_name=var_name,
            control_name=control_name,
            control_case=control_case,
            control_ufunc=control_ufunc,
            test_name=label,
            test_case=case_config["case"],
            test_ufunc=case_config["ufunc"],
            uncertainty=True,
            uncertainty_detrended=control_detrended,
            uncertainty_tsel=control_tsel,
            detrend_PIC=False,
            ax=ax,
            cmap=PLOT_VAR_CONFIGS[var_name]["cmap"],
            vlims=PLOT_VAR_CONFIGS[var_name]["vlims"],
            cax=cbar_axes[-1 - i],
            cbar_kwargs=PLOT_VAR_CONFIGS[var_name]["cbar_kwargs"],
            **case_config["args"],
        )

        ax_out.coastlines()
        ax_out.set_global()

# Label the top of each column with the case name
for ax, label in zip(axes[:, 0], PLOT_CONFIGS1.keys()):
    plt.text(0.5, 1.15, label, transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

# Label the left of each row with the variable name
for ax, var_name in zip(axes[0, :], energy_vars):
    plt.text(-0.1, 0.5, var_name, transform=ax.transAxes, ha='right', va='center', fontsize=10, rotation=90)

# %%

fig.savefig("figures/figure4_ensmean_2015_2034.png", dpi=300, bbox_inches='tight')
logging.info("Saved figure4_ensmean_2015_2034.png")
plt.close(fig)

# %%
energy_vars = ["FNNT", "FLNT", "FSNT", "FLNS", "FSNS", "SHFLX", "LHFLX", "PRECIP_THERMO"]
test_cases = ['CESM2_WACCM_SSP2-4.5', 'ARISE-SAI', 'ARISE-1.0', 'CESM2_WACCM_SSP2-4.5_MCB']
PLOT_CONFIGS_ALL = {}
for case in test_cases:
    case_dict = mean_data_dict[case]
    for subdir in case_dict:
        if "ens" in case_dict[subdir].dims:
            ufunc = lambda ds: ds.mean("ens")
        else:
            ufunc = None
        PLOT_CONFIGS_ALL[f"{case}_{subdir}"] = {
            "case": case,
            "subcase": subdir,
            "ufunc": ufunc,
            "args": {"colorbar": False},
        }

# Remove the MCB control and the ARISE extended case from the plot configs, since they are not relevant for this figure.
PLOT_CONFIGS_ALL.pop('ARISE-SAI_b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??')
PLOT_CONFIGS_ALL.pop('CESM2_WACCM_SSP2-4.5_MCB_b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000')

# Make sure there is a colorbar
PLOT_CONFIGS_ALL['CESM2_WACCM_SSP2-4.5_b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??']["args"]["colorbar"] = True

# Plot relative to the 2015-2034 period of the SSP2-4.5 case
control_name = 'CESM2_WACCM_SSP2-4.5'
control_case = 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??'
control_ufunc = lambda ds: ds.sel(year=slice(2015, 2034)).mean(dim=["year", "ens"])
control_detrended = False
control_tsel = slice(2015, 2034)

num_rows = len(energy_vars)
num_cols = len(PLOT_CONFIGS_ALL)
config = PLOT_CONFIGS_ALL
fig, axs = sp_map(num_rows, num_cols,  projection=ccrs.Robinson(), figsize=(3*num_cols, 2*num_rows))
fig.subplots_adjust(wspace=0.12, hspace=0.15)
axes = axs.T

# Add a cbar axis for each row
cbar_axes = []
for i in range(len(energy_vars)):
    cbar_ax = fig.add_axes([0.92, 0.11 + i*0.097, 0.005, 0.09])
    cbar_axes.append(cbar_ax)

for label, axs in zip(config, axes):
    case_config = config[label]
    test_case = case_config["case"]
    test_ufunc = case_config["ufunc"]

    for i, (var_name, ax) in enumerate(zip(energy_vars, axs)):
        print(f"Plotting {var_name} for case {label}")

        ax_out = plot_difference_map(
            data_dict=mean_data_dict,
            unc_data_dict=uncertainty_data_dict,
            var_name=var_name,
            control_name=control_name,
            control_case=control_case,
            control_ufunc=control_ufunc,
            test_name=case_config["case"],
            test_case=case_config["subcase"],
            test_ufunc=case_config["ufunc"],
            uncertainty=True,
            uncertainty_detrended=control_detrended,
            uncertainty_tsel=control_tsel,
            detrend_PIC=False,
            ax=ax,
            cmap=PLOT_VAR_CONFIGS[var_name]["cmap"],
            vlims=PLOT_VAR_CONFIGS[var_name]["vlims"],
            cax=cbar_axes[-1 - i],
            cbar_kwargs=PLOT_VAR_CONFIGS[var_name]["cbar_kwargs"],
            **case_config["args"],
        )

        ax_out.coastlines()
        ax_out.set_global()

# Label the top of each column with the case name
for ax, label in zip(axes[:, 0], config.keys()):
    title = title_dict[config[label]["case"]][config[label]["subcase"]]
    N = mean_data_dict[config[label]["case"]][config[label]["subcase"]].dims.get("ens", 1)
    ax.text(0.5, 1.2, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=14)
    ax.text(0.82, 1.0, f"N={N}", transform=ax.transAxes, ha='left', va='bottom', fontsize=12, fontweight='bold')

# Label the left of each row with the variable name
for ax, var_name in zip(axes[0, :], energy_vars):
    ax.text(-0.1, 0.5, var_name, transform=ax.transAxes, ha='right', va='center', fontsize=14, rotation=90)

# %%
fig.savefig("figures/figure4_allcases_2015_2034.png", dpi=300, bbox_inches='tight')
logging.info("Saved figure4_allcases_2015_2034.png")
plt.close(fig)
# %%
