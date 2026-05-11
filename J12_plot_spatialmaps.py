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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)


def plot_difference_map(
    data_dict: dict,
    var_name: str,
    control_name: str,
    control_case: str,
    test_name: str,
    test_case: str,
    control_ufunc: callable = None,
    test_ufunc: callable = None,
    uncertainty: bool = False,
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

    control_data = data_dict[control_name][control_case][var_name]
    if control_ufunc is not None:
        control_data = control_ufunc(control_data)

    test_data = data_dict[test_name][test_case][var_name]
    if test_ufunc is not None:
        test_data = test_ufunc(test_data)
    difference_data = test_data - control_data

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
        uncertainty_data = data_dict[control_name][control_case][var_name + "_uncertainty"].sel(period=10)
        uncertainty_low = control_data + uncertainty_data.sel(quantile=0.025)
        uncertainty_high = control_data + uncertainty_data.sel(quantile=0.975)
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
        ax.set_title(f"Significant fraction: {significant_fraction.values:.2%}")

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


    spatial_proc_dir = "/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/spatial_maps/"
    CASE_CONFIGS1 = {
        "CESM2_WACCM_1850control" :{
            "path": spatial_proc_dir + "CESM2_WACCM_1850control/",
            "subdir_cases": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
        },
        "CESM2_WACCM_SSP2-4.5": {
            "path": spatial_proc_dir + "CESM2_WACCM_SSP2-4.5/",
            "subdir_cases": ["b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"],
        },
        "ARISE-SAI": {
            "path": spatial_proc_dir + "ARISE-SAI/",
            "subdir_cases": [
                "1p5K-SAI.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.00?",
            ],
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "path": spatial_proc_dir + "CESM2_WACCM_SSP2-4.5_MCB/",
            "subdir_cases": [
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000",
            ],
        },
    }
    CASE_CONFIGS = CASE_CONFIGS1
    data_dict = {}
    
    for case_label in CASE_CONFIGS.keys():
        logging.info(f"Loading data for case: {case_label}")
        datapath = CASE_CONFIGS[case_label]["path"]
        case_dict = {}
        
        for case_str in CASE_CONFIGS[case_label]["subdir_cases"]:
            logging.info(f"Loading data for subcase: {case_str}")
            spatial_files = glob.glob(os.path.join(datapath, case_str + "*.nc"))
            logging.info(f"Found {len(spatial_files)} files for subcase {case_str}")
            if len(spatial_files) == 0:
                logging.warning(f"No files found for subcase {case_str} in path {datapath}")
                continue
            spatial_subds = xr.open_mfdataset(spatial_files, combine="by_coords")
            case_dict[case_str] = spatial_subds
        data_dict[case_label] = case_dict

# %%
# Draft plotting code.

PLOT_CONFIGS1 = {
    "CESM2_WACCM_SSP2-4.5": {
        "case": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
        # "ufunc": lambda ds: ds.isel(ens=0),
        "ufunc": lambda ds: ds.mean("ens"),
        "args": {"colorbar": True},
    },
    "ARISE-SAI": {
        "case": "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00?",
        # "ufunc": lambda ds: ds.isel(ens=0),
        "ufunc": lambda ds: ds.mean("ens"),
        "args": {"colorbar": False},
    },
    "CESM2_WACCM_SSP2-4.5_MCB": {
        "case": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.00?",
        # "ufunc": lambda ds: ds.isel(ens=0),
        "ufunc": lambda ds: ds.mean("ens"),
        "args": {"colorbar": False},
    },
}

PLOT_VAR_CONFIGS = {
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

energy_vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX", "LHFLX", "PRECIP_THERMO"]

control_name = "CESM2_WACCM_1850control"
control_case = "b.e21.BW1850.f09_g17.CMIP6-piControl.001"
control_ufunc = lambda ds: ds.mean("year")

fig, axs = sp_map(7, 3,  projection=ccrs.Robinson(), figsize=(12, 19))
axes = axs.T

# Add a cbar axis for each row
cbar_axes = []
for i in range(len(energy_vars)):
    cbar_ax = fig.add_axes([0.92, 0.11 + i*0.1125, 0.02, 0.1])
    cbar_axes.append(cbar_ax)

for label, axs in zip(PLOT_CONFIGS1, axes):
    case_config = PLOT_CONFIGS1[label]
    test_case = case_config["case"]
    test_ufunc = case_config["ufunc"]

    for i, (var_name, ax) in enumerate(zip(energy_vars, axs)):
        print(f"Plotting {var_name} for case {label}")

        ax_out = plot_difference_map(
            data_dict=data_dict,
            var_name=var_name,
            control_name=control_name,
            control_case=control_case,
            control_ufunc=control_ufunc,
            test_name=label,
            test_case=case_config["case"],
            test_ufunc=case_config["ufunc"],
            uncertainty=True,
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

fig.savefig("figures/figure4_ensmean.png", dpi=300, bbox_inches='tight')
logging.info("Saved figure4_ensmean.png")
plt.close(fig)
# %%