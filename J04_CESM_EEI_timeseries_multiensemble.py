# %%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from copy import deepcopy

from J15_shared_functions import (
    weighted_annualmean,
    get_weights_by_month2,
    compute_decadal2,
    load_data_with_configs,
    title_dict,
    CASE_CONFIGS_TEMPLATE,
)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

def compute_iEEI(
    olr_ds,
    asr_ds,
    account_for_leap: bool = False,
):
    """
    Compute the integrated earth's energy imbalance (iEEI) from ASR and OLR fields.
    """
    assert (olr_ds["time"] == asr_ds["time"]).all(), "OLR and ASR time fields are not identical"
    time_ds = olr_ds["time"]

    weights = get_weights_by_month2(time_ds, account_for_leap)
    eei_ds = asr_ds - olr_ds
    # Use the named time dimension so the cumulative sum is independent of
    # the variable's underlying dimension order.
    ieei_ds = (eei_ds * weights).cumsum(dim="time") # J/m^2

    return ieei_ds


def plot_eei_timeseries(
    asr_annual, olr_annual, eei_annual, ieei_annual,
    asr_decadal, olr_decadal, eei_decadal, ieei_decadal,
    ax=None, fontsize=14, case_name="",time_dim="year",
    colors=sns.color_palette("colorblind", n_colors=3),
):
    """
    Plot timeseries of ASR, OLR, EEI, and iEEI on a subplot with twinned y-axes.
    
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
        The three twinned axes (ASR/OLR, EEI, iEEI)
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
    asr_color = colors[0]
    olr_color = colors[1]
    eei_color = colors[2]
    if ieei_annual is not None:
        ieei_color = colors[3]

    # Plot annual means from individual ensemble mebers (thin, semi-transparent)
    if "ens" in asr_annual.dims:
        for ens_member in asr_annual["ens"].values:
            ax1.plot(asr_annual[time_dim], asr_annual.sel(ens=ens_member), color=asr_color, linestyle="-", 
                     linewidth=0.7, alpha=0.5)
            ax1.plot(olr_annual[time_dim], olr_annual.sel(ens=ens_member), color=olr_color, linestyle="-", 
                     linewidth=0.7, alpha=0.5)
            ax2.plot(eei_annual[time_dim], eei_annual.sel(ens=ens_member), color=eei_color, linestyle="-", 
                     linewidth=0.7, alpha=0.5)
        # Now convert to the ensemble mean for the next step
        asr_decadal = asr_decadal.mean(dim="ens")
        olr_decadal = olr_decadal.mean(dim="ens")
        eei_decadal = eei_decadal.mean(dim="ens")
    else:
        ax1.plot(asr_annual[time_dim], asr_annual, color=asr_color, linestyle="-", 
                    linewidth=0.7, alpha=0.5)
        ax1.plot(olr_annual[time_dim], olr_annual, color=olr_color, linestyle="-", 
                    linewidth=0.7, alpha=0.5)
        ax2.plot(eei_annual[time_dim], eei_annual, color=eei_color, linestyle="-", 
                    linewidth=0.7, alpha=0.5)
    
    # Plot decadal means (thick, solid)
    ax1.plot(asr_decadal[time_dim], asr_decadal, color=asr_color, linestyle="-", 
             linewidth=2.5, label="ASR (decadal)")
    ax1.plot(olr_decadal[time_dim], olr_decadal, color=olr_color, linestyle="-", 
             linewidth=2.5, label="OLR (decadal)")
    
    ax2.plot(eei_decadal[time_dim], eei_decadal, color=eei_color, linestyle="-", 
             linewidth=2.5, label="EEI (decadal)")
    
    if ieei_annual is not None:
        ax3.plot(ieei_decadal[time_dim], ieei_decadal, color=ieei_color, linestyle="-", 
                linewidth=2.5, label="iEEI (decadal)")
    
    # Set axis labels and colors
    ax1.set_xlabel("Year", fontsize=fontsize)
    ax.text(
        -0.15, 0.42, "ASR,",
        transform=ax.transAxes, rotation=90,
        ha="center", va="top", fontsize=fontsize, color=asr_color,
    )
    ax.text(
        -0.15, 0.43, "OLR [Wm$^{-2}$]",
        transform=ax.transAxes, rotation=90,
        ha="center", va="bottom", fontsize=fontsize, color=olr_color,
    )
    ax1.tick_params(axis="y")
    ax2.set_ylabel("EEI [Wm$^{-2}$]", fontsize=fontsize, color=eei_color)
    ax2.tick_params(axis="y", labelcolor=eei_color)
    
    if ieei_annual is not None:
        ax3.set_ylabel("iEEI [W]", fontsize=fontsize, color=ieei_color)
        ax3.tick_params(axis="y", labelcolor=ieei_color)
    
    # Add title and grid
    ax1.set_title(case_name, fontsize=fontsize)
    ax1.grid(True, alpha=0.3)

    if ieei_annual is not None:
        return ax1, ax2, ax3
    else:
        return ax1, ax2, None


def plot_ieei_ts_ohc(
    ieei_annual, ieei_decadal, ts_annual, ts_decadal, ohc_annual, ohc_decadal, ax=None,
    fontsize=14, time_dim="year",
    colors=sns.color_palette("colorblind", n_colors=6)[3:],
    
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax1 = ax
    ax2 = ax.twinx()
    ax1.grid(True, alpha=0.3)
    ieei_color = colors[0]
    ts_color = colors[1]
    ohc_color = colors[2]

    # Plot annual means from individual ensemble mebers (thin, semi-transparent)
    if "ens" in ieei_annual.dims:
        for ens_member in ieei_annual["ens"].values:
            ax1.plot(ieei_annual[time_dim], ieei_annual.sel(ens=ens_member), color=ieei_color,          
                     linestyle="-", linewidth=0.7, alpha=0.5)
            ax2.plot(ts_annual[time_dim], ts_annual.sel(ens=ens_member), color=ts_color, linestyle="-", 
                     linewidth=0.7, alpha=0.5)
            ax1.plot(ohc_annual[time_dim], ohc_annual.sel(ens=ens_member), color=ohc_color, linestyle="-", 
                     linewidth=0.7, alpha=0.5)
        # Now convert to the ensemble mean for the next step
        ieei_decadal = ieei_decadal.mean(dim="ens")
        ts_decadal = ts_decadal.mean(dim="ens")
        ohc_decadal = ohc_decadal.mean(dim="ens")
    else:
        ax1.plot(ieei_annual[time_dim], ieei_annual, color=ieei_color, linestyle="-", 
                    linewidth=0.7, alpha=0.5)
        ax2.plot(ts_decadal[time_dim], ts_decadal, color=ts_color, linestyle="-", 
                    linewidth=0.7, alpha=0.5)
        ax1.plot(ohc_annual[time_dim], ohc_annual, color=ohc_color, linestyle="-", 
                    linewidth=0.7, alpha=0.5)

    # Plot decadal means (thick, solid)
    ax1.plot(ieei_decadal[time_dim], ieei_decadal, color=ieei_color, linestyle="-", 
                linewidth=2.5, alpha=1, label="iEEI (decadal)")
    ax2.plot(ts_decadal[time_dim], ts_decadal, color=ts_color, linestyle="-", 
                linewidth=2.5, alpha=1, label="Surface Temperature (decadal)")
    ax1.plot(ohc_decadal[time_dim], ohc_decadal, color=ohc_color, linestyle="-", 
                linewidth=2.5, alpha=1, label="OHC (decadal)")

    ax1.set_xlabel("Year", fontsize=fontsize)
    ax1.text(
        -0.15, 0.42, "iEEI,",
        transform=ax.transAxes, rotation=90,
        ha="center", va="top", fontsize=fontsize, color=ieei_color,
    )
    ax1.text(
        -0.15, 0.43, "OHC [Jm$^{-2}$]",
        transform=ax.transAxes, rotation=90,
        ha="center", va="bottom", fontsize=fontsize, color=ohc_color,
    )
    # ax1.set_ylabel("iEEI, OHC [Jm$^{-2}$]", fontsize=14)
    # ax1.tick_params(axis="y")

    ax2.tick_params(axis="y", labelcolor=ts_color)
    ax2.set_ylabel("Surface Temperature [K]", fontsize=fontsize, color=ts_color)

    return ax1, ax2


def compute_ieei_with_start_year(
    asr_ds,
    olr_ds,
    start_year,
    account_for_leap: bool = False,
):
    """
    Compute the integrated earth's energy imbalance (iEEI) starting from a specified year.

    Parameters
    ----------
    asr_ds : xr.DataArray
        Absorbed shortwave radiation data
    olr_ds : xr.DataArray
        Outgoing longwave radiation data
    start_year : int
        Year to begin integration (iEEI will be zero at this year)
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

    # Compute iEEI using the existing function
    ieei_ds = compute_iEEI(olr_sliced, asr_sliced, account_for_leap=account_for_leap)

    # Mask where asr and olr are nans
    ieei_ds = ieei_ds.where(~np.isnan(asr_sliced) & ~np.isnan(olr_sliced))

    return ieei_ds


def remove_axis_text_objects(ax, remove_strings: list = None):
    """Remove all Text artists explicitly added to an Axes."""
    for text in ax.texts:
        for _str in remove_strings:
            if _str in text.get_text():
                text.remove()


# %%

if __name__ == "__main__":
    data_root = "/glade/work/jonahshaw/PRISM_data/spatial_averages_data/"
    data_root_ohc = "/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/"    

    CASE_CONFIGS_ATM = deepcopy(CASE_CONFIGS_TEMPLATE)
    CASE_CONFIGS_OCN = deepcopy(CASE_CONFIGS_TEMPLATE)

    for config in CASE_CONFIGS_ATM.values():
        config["path"] = str(Path(data_root) / config.pop("path"))
    for config in CASE_CONFIGS_OCN.values():
        config["path"] = str(Path(data_root_ohc) / config.pop("path"))

    # %%
    # Load the data using the generalized loading function
    data_varlist = ['FLNT', 'FSNT', 'TS']
    year_dim = "time"
    ohc_varlist = ["OHC", "OHC_global_mean"]

    # Load data using the generalized function (requires 4 nodes on CURC)
    data_dict = {}
    for var in data_varlist:
        data_dict[var] = load_data_with_configs(CASE_CONFIGS_ATM, [var], year_dim=year_dim, load_into_memory=False)

    ohc_dict = load_data_with_configs(CASE_CONFIGS_OCN, ohc_varlist, year_dim=year_dim, load_into_memory=False)

    # %%
    # CAM global surface area: 
    earth_radius_cam = 6.37122e6
    earth_SA_cam = 4 * np.pi * earth_radius_cam**2
    PLOT_CONFIGS1 = {
        "CESM2-LM": {
            "case_str": "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.0??",
            "ax1_lims": (237, 247),
            "ax2_lims": (-8, 2),
            "ax1_major_y": MultipleLocator(1),
            "axb2_lims": (287.0, 289.5),
            "axb1_lims": (-1.0e9, 2.25e9),
            "axb1_yticks": np.arange(-1.0e9, 2.25e9, 0.5e9),
            "axb2_yticks": np.arange(286.5, 290.0+0.01, 0.5),
            "xlims": (850, 1850),
            "keep_left_axes": True,
            "keep_right_axes": True,
        },
        "CESM2_WACCM_1850control": {
            "case_str": "b.e21.BW1850.f09_g17.CMIP6-piControl.001",
            "ax1_lims": (237, 247),
            "ax2_lims": (-8, 2),
            "ax1_major_y": MultipleLocator(1),
            "axb2_lims": (287.0, 289.5),
            "axb1_lims": (-0.1e9, 1.9e9),
            "axb1_yticks": np.arange(0, 1.9e9, 0.25e9),
            "axb2_yticks": np.arange(286.5, 290.0+0.01, 0.5),
            "xlims": (1, 500),
            "keep_left_axes": True,
            "keep_right_axes": True,
        },
    }
    # %%
    # PLOT 1: Historical scenarios (CESM2-LM, CESM2_WACCM_1850control)
    # Integration starts from year 850
    logging.info("Creating Plot 1: Historical scenarios (CESM2_WACCM-LM, CESM2_WACCM_1850control)")
    PLOT_CONFIGS = PLOT_CONFIGS1
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.40)
    
    case_list_1 = PLOT_CONFIGS1.keys()
    startyears_dict = {
        "CESM_LME": 850,
        "CESM2-LM": 850,
        "CESM2_WACCM_1850control": 1,
    }
    ts_var = "TS"
    olr_var = "FLNT"
    asr_var = "FSNT"
    
    colors = sns.color_palette("colorblind", n_colors=6)

    top_handles = []
    bottom_handles = []
    top_labels = []
    bottom_labels = []

    for idx, (ax, axb, case_label) in enumerate(zip(axes[0], axes[1], case_list_1)):
        logging.info(f"Plotting case: {case_label}")
        
        # Get case_str from config
        case_str = PLOT_CONFIGS[case_label]["case_str"]
        
        # Extract ASR and OLR data
        asr_ds = data_dict[asr_var][case_label][case_str][asr_var].sel(spatial="G").squeeze().compute()
        olr_ds = data_dict[olr_var][case_label][case_str][olr_var].sel(spatial="G").squeeze().compute()
        ts_ds = data_dict[ts_var][case_label][case_str][ts_var].sel(spatial="G").squeeze().compute()
        ohc_ds = ohc_dict[case_label][case_str].squeeze()
        ohc_ds = ohc_ds["OHC_global_mean"].sel(ohc_depth=-1) * ohc_ds.attrs["ocean_area_m2"] / earth_SA_cam
        ohc_ds = ohc_ds - ohc_ds.isel(time=0)  # Convert OHC to anomaly

        # Compute annual means
        asr_annual = weighted_annualmean(asr_ds)
        olr_annual = weighted_annualmean(olr_ds)
        ts_annual = weighted_annualmean(ts_ds)

        eei_annual = asr_annual - olr_annual
        ohc_annual = weighted_annualmean(ohc_ds) if ohc_ds is not None else None
        
        # Compute decadal means
        asr_decadal = compute_decadal2(asr_ds)
        olr_decadal = compute_decadal2(olr_ds)
        ts_decadal = compute_decadal2(ts_ds)
        eei_decadal = asr_decadal - olr_decadal
        ohc_decadal = compute_decadal2(ohc_ds) if ohc_ds is not None else None
        
        # Compute iEEI starting from appropriate year for each case
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, startyears_dict[case_label])
        
        # Create annual and decadal means for iEEI by grouping years
        ieei_annual = weighted_annualmean(ieei_ds)
        ieei_decadal = compute_decadal2(ieei_ds)

        # Plot ASR, OLR, and EEI
        ax1, ax2, ax3 = plot_eei_timeseries(
            asr_annual, olr_annual, eei_annual, None,
            asr_decadal, olr_decadal, eei_decadal, None,
            ax=ax, fontsize=14, case_name=case_label,
            colors=colors,
        )
        
        # Plot iEEI, TS, and OHC
        axb, axb2 = plot_ieei_ts_ohc(
            ieei_annual=ieei_annual, ieei_decadal=ieei_decadal,
            ts_annual=ts_annual, ts_decadal=ts_decadal,
            ohc_annual=ohc_annual, ohc_decadal=ohc_decadal,
            ax=axb, colors=colors[3:], fontsize=14, time_dim="year",
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

        # Add ensemble count annotation
        n_ens = 1
        if "ens" in asr_annual.dims:
            n_ens = len(asr_annual["ens"])
        ax1.text(0.98, 0.98, f"N={n_ens}", fontsize=12, fontweight="bold", 
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right')

        # Add a horizontal line at y=0 for the EEI subplot
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax.set_facecolor("whitesmoke")
        axb.set_facecolor("whitesmoke")

        # Collect handles and labels from left column for combined legend
        if idx == 0:
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            top_handles.extend(handles1[2:] + handles2[1:])
            top_labels.extend(["ASR", "OLR", "EEI"])
            handles_b, labels_b = axb.get_legend_handles_labels()
            handles_b2, labels_b2 = axb2.get_legend_handles_labels()
            bottom_handles.extend(handles_b[2:] + handles_b2[1:])
            bottom_labels.extend(["iEEI", "OHC", "Surface Temperature"])

    # Create a single legend for the left column
    # if top_handles and top_labels:
    #     axes[0,0].legend(
    #         top_handles, top_labels, loc='lower right', 
    #         fontsize=12, framealpha=0.95,
    #     )
    # if bottom_handles and bottom_labels:
    #     axes[1,0].legend(
    #         bottom_handles, bottom_labels, loc='lower right', 
    #         fontsize=12, framealpha=0.95,
    #     )
    # Create panel labels
    panel_labels = [f"{chr(97 + i)}." for i in range(len(axes.flat))]
    for i, (ax, label) in enumerate(zip(axes.flat, panel_labels)):
        ax.text(0.02, 0.98, label, fontsize=14, fontweight="bold", 
                transform=ax.transAxes, verticalalignment='top')


    # %%
    fig.savefig("figures/figure_timeseries_past.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure_timeseries_past.png")
    plt.close(fig)

    # %%
    earth_radius_cam = 6.37122e6
    earth_SA_cam = 4 * np.pi * earth_radius_cam**2

    # Shared y-axis and time limits for easier updating across future-scenario plots
    ALL_SHARED_PLOT_LIMITS = {
        "ax1_lims": (236, 245),
        "ax2_lims": (-5, 4),
        "axb1_lims": (-0.1e9, 6e9),
        "axb2_lims": (286.0, 293.0),
    }
    SOME_SHARED_PLOT_LIMITS = {
        "xlims": (2015, 2100),
        "keep_left_axes": False,
        "keep_right_axes": True,
        "preceding_case": ["CESM2-WACCM-HIST", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??"],
    }
    PLOT_CONFIGS2 = {
        'CESM2-WACCM-HIST': {
            "case_str": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??",
            **ALL_SHARED_PLOT_LIMITS,
            "xlims": (1850, 2015),
            "keep_left_axes": True,
            "keep_right_axes": False,
            "preceding_case": None,
        },
        'CESM2_WACCM_SSP2-4.5': {
            "case_str": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??",
            **ALL_SHARED_PLOT_LIMITS,
            **SOME_SHARED_PLOT_LIMITS,
        },
        "ARISE-SAI": {
            "case_str": "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??",
            **ALL_SHARED_PLOT_LIMITS,
            **SOME_SHARED_PLOT_LIMITS,
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "case_str": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??",
            **ALL_SHARED_PLOT_LIMITS,
            "xlims": (2015, 2100),
            "keep_left_axes": False,
            "keep_right_axes": True,
            "preceding_case": ["CESM2-WACCM-HIST", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??"],
        },
    }

    # %%
    # PLOT 2: Future scenarios (CESM2_WACCM_SSP2-4.5, ARISE-SAI, CESM2_WACCM_SSP2-4.5_MCB)
    # Integration starts from year 1850
    logging.info("Creating Plot 2: Future scenarios (CESM2_WACCM_SSP2-4.5, ARISE-SAI, CESM2_WACCM_SSP2-4.5_MCB)")
    PLOT_CONFIGS = PLOT_CONFIGS2

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.subplots_adjust(wspace=0.35)
    
    case_list_2 = ["CESM2-WACCM-HIST", "CESM2_WACCM_SSP2-4.5", "ARISE-SAI", "CESM2_WACCM_SSP2-4.5_MCB"]
    start_year_2 = 1850
    ts_var = "TS"
    olr_var = "FLNT"
    asr_var = "FSNT"
    colors = sns.color_palette("colorblind", n_colors=6)

    top_handles = []
    bottom_handles = []
    top_labels = []
    bottom_labels = []
    case_end_ohc_ieei_offsets = {}  # Dictionary to store the ending OHC and iEEI values for each case

    for idx, (ax, axb, case_label) in enumerate(zip(axes[0], axes[1], case_list_2)):
        logging.info(f"Plotting case: {case_label}")

        # Get case_str from config
        case_str = PLOT_CONFIGS[case_label]["case_str"]

        # Extract ASR and OLR data
        asr_ds = data_dict[asr_var][case_label][case_str][asr_var].sel(spatial="G").squeeze().compute()
        olr_ds = data_dict[olr_var][case_label][case_str][olr_var].sel(spatial="G").squeeze().compute()
        ts_ds = data_dict[ts_var][case_label][case_str][ts_var].sel(spatial="G").squeeze().compute()
        ohc_ds = ohc_dict[case_label][case_str].squeeze()
        ohc_ds = ohc_ds["OHC_global_mean"].sel(ohc_depth=-1) * ohc_ds.attrs["ocean_area_m2"] / earth_SA_cam
        ohc_ds = ohc_ds - ohc_ds.isel(time=0)  # Convert OHC to anomaly

        # Because OHC and iEEI normalized to a time period, add the ending value of the preceding case to the current case for continuity
        if "preceding_case" in PLOT_CONFIGS[case_label] and PLOT_CONFIGS[case_label]["preceding_case"] is not None:
            ohc_offset = case_end_ohc_ieei_offsets[PLOT_CONFIGS[case_label]["preceding_case"][0]][PLOT_CONFIGS[case_label]["preceding_case"][1]]["ohc_end"]
            ieei_offset = case_end_ohc_ieei_offsets[PLOT_CONFIGS[case_label]["preceding_case"][0]][PLOT_CONFIGS[case_label]["preceding_case"][1]]["ieei_end"]
            ohc_ds = ohc_ds + ohc_offset
        
        # Compute annual means
        asr_annual = weighted_annualmean(asr_ds)
        olr_annual = weighted_annualmean(olr_ds)
        ts_annual = weighted_annualmean(ts_ds)
        eei_annual = asr_annual - olr_annual
        ohc_annual = weighted_annualmean(ohc_ds) if ohc_ds is not None else None

        # Compute decadal means
        asr_decadal = compute_decadal2(asr_ds)
        olr_decadal = compute_decadal2(olr_ds)
        ts_decadal = compute_decadal2(ts_ds)
        eei_decadal = asr_decadal - olr_decadal
        ohc_decadal = compute_decadal2(ohc_ds) if ohc_ds is not None else None
        
        # Compute iEEI starting from start_year_2
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, start_year_2)
        if "preceding_case" in PLOT_CONFIGS[case_label] and PLOT_CONFIGS[case_label]["preceding_case"] is not None:
            ieei_ds = ieei_ds + ieei_offset
        
        # Create annual and decadal means for iEEI by grouping years
        ieei_annual = weighted_annualmean(ieei_ds)
        ieei_decadal = compute_decadal2(ieei_ds)

        # Store the ending OHC and iEEI values for the current case
        case_end_ohc_ieei_offsets[case_label] = {case_str: {
            "ohc_end": ohc_annual.isel(year=-1).mean(dim="ens").values,
            "ieei_end": ieei_annual.isel(year=-1).mean(dim="ens").values},
                                                 }

        # Plot
        ax1, ax2, ax3 = plot_eei_timeseries(
            asr_annual, olr_annual, eei_annual, None,
            asr_decadal, olr_decadal, eei_decadal, None,
            ax=ax, fontsize=14, case_name=case_label,
            colors=colors,
        )

        # Plot iEEI, TS, and OHC
        axb, axb2 = plot_ieei_ts_ohc(
            ieei_annual=ieei_annual, ieei_decadal=ieei_decadal,
            ts_annual=ts_annual, ts_decadal=ts_decadal,
            ohc_annual=ohc_annual, ohc_decadal=ohc_decadal,
            ax=axb, colors=colors[3:], fontsize=14, time_dim="year",
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
        # Set y-axis limits
        if "ax1_lims" in PLOT_CONFIGS[case_label]:
            ax1.set_ylim(*PLOT_CONFIGS[case_label]["ax1_lims"])
        if "ax2_lims" in PLOT_CONFIGS[case_label]:
            ax2.set_ylim(*PLOT_CONFIGS[case_label]["ax2_lims"])

        if "axb1_lims" in PLOT_CONFIGS[case_label]:
            axb.set_ylim(*PLOT_CONFIGS[case_label]["axb1_lims"])
        if "axb2_lims" in PLOT_CONFIGS[case_label]:
            axb2.set_ylim(*PLOT_CONFIGS[case_label]["axb2_lims"])

        # Add ensemble count annotation
        n_ens = 1
        if "ens" in asr_annual.dims:
            n_ens = len(asr_annual["ens"])
        ax1.text(0.98, 0.98, f"N={n_ens}", fontsize=12, fontweight="bold", 
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right')

        # Add a horizontal line at y=0 for the EEI subplot
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax.set_facecolor("whitesmoke")
        axb.set_facecolor("whitesmoke")

        # Collect handles and labels from left column for combined legend
        if idx == 0:
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            top_handles.extend(handles1 + handles2)
            top_labels.extend(["ASR", "OLR", "EEI"])
            handles_b, labels_b = axb.get_legend_handles_labels()
            handles_b2, labels_b2 = axb2.get_legend_handles_labels()
            bottom_handles.extend(handles_b + handles_b2)
            bottom_labels.extend(["iEEI", "OHC", "Surface Temperature"])

    # Create panel labels
    panel_labels = [f"{chr(97 + i)}." for i in range(len(axes.flat))]
    for i, (ax, label) in enumerate(zip(axes.flat, panel_labels)):
        ax.text(0.02, 0.98, label, fontsize=14, fontweight="bold", 
                transform=ax.transAxes, verticalalignment='top')

    # %%
    fig.savefig("figures/figure_timeseries_future.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure_timeseries_future.png")
    plt.close(fig)
    # %%

    # Create a configs dictionary for a mega plot
    ALL_SHARED_PLOT_LIMITS = {
        "ax1_lims": (236, 245),
        "ax2_lims": (-5, 4),
        "axb1_lims": (-0.1e9, 6e9),
        "axb2_lims": (286.0, 293.0),
    }
    SOME_SHARED_PLOT_LIMITS = {
        "xlims": (2015, 2100),
        "keep_left_axes": False,
        "keep_right_axes": True,
        "preceding_case": ["CESM2 WACCM HIST", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??"],
    }

    # Build a plot configs dictionary for a mega plot
    megaplot_cases = [
        "CESM2-WACCM-HIST",
        "CESM2_WACCM_SSP2-4.5",
        "ARISE-SAI",
        "ARISE-1.0",
        "CESM2_WACCM_SSP2-4.5_MCB",
    ]
    PLOT_CONFIGS_ALLFUTURE = {}
    for case in megaplot_cases:
        for experiment in title_dict[case]:
            PLOT_CONFIGS_ALLFUTURE[title_dict[case][experiment]] = {
                "case_str": case,
                "exp_str": experiment,
                **ALL_SHARED_PLOT_LIMITS,
                **SOME_SHARED_PLOT_LIMITS,
            }
    # Apply corrections.
    PLOT_CONFIGS_ALLFUTURE["CESM2 WACCM HIST"]["xlims"] = (1850, 2015)
    PLOT_CONFIGS_ALLFUTURE["CESM2 WACCM HIST"]["keep_left_axes"] = True
    PLOT_CONFIGS_ALLFUTURE["CESM2 WACCM HIST"]["keep_right_axes"] = False
    PLOT_CONFIGS_ALLFUTURE["CESM2 WACCM HIST"]["preceding_case"] = None

    # %%
    logging.info("Creating Plot 3: All Future scenarios")
    PLOT_CONFIGS = PLOT_CONFIGS_ALLFUTURE
    PLOT_CONFIGS.pop('MCB CMIP6 baseline')  # Remove the MCB control from the mega plot

    nrows = 6
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 26))
    fig.subplots_adjust(wspace=0.3)
    top_axes = axes[::2, :]
    bottom_axes = axes[1::2, :]
    
    start_year_2 = 1850
    ts_var = "TS"
    olr_var = "FLNT"
    asr_var = "FSNT"
    colors = sns.color_palette("colorblind", n_colors=6)

    top_handles = []
    bottom_handles = []
    top_labels = []
    bottom_labels = []
    case_end_ohc_ieei_offsets = {}  # Dictionary to store the ending OHC and iEEI values for each case

    for idx, (ax, axb, case_title) in enumerate(zip(top_axes.flatten(), bottom_axes.flatten(), PLOT_CONFIGS.keys())):
        logging.info(f"Plotting case: {case_title}")

        # Get case_str from config
        case_label = PLOT_CONFIGS[case_title]["case_str"]
        case_str = PLOT_CONFIGS[case_title]["exp_str"]

        # Extract ASR and OLR data
        asr_ds = data_dict[asr_var][case_label][case_str][asr_var].sel(spatial="G").squeeze().compute()
        olr_ds = data_dict[olr_var][case_label][case_str][olr_var].sel(spatial="G").squeeze().compute()
        ts_ds = data_dict[ts_var][case_label][case_str][ts_var].sel(spatial="G").squeeze().compute()
        ohc_ds = ohc_dict[case_label][case_str].squeeze()
        ohc_ds = ohc_ds["OHC_global_mean"].sel(ohc_depth=-1) * ohc_ds.attrs["ocean_area_m2"] / earth_SA_cam
        ohc_ds = ohc_ds - ohc_ds.isel(time=0)  # Convert OHC to anomaly

        # Because OHC and iEEI normalized to a time period, add the ending value of the preceding case to the current case for continuity
        if "preceding_case" in PLOT_CONFIGS[case_title] and PLOT_CONFIGS[case_title]["preceding_case"] is not None:
            ohc_offset = case_end_ohc_ieei_offsets[PLOT_CONFIGS[case_title]["preceding_case"][0]][PLOT_CONFIGS[case_title]["preceding_case"][1]]["ohc_end"]
            ieei_offset = case_end_ohc_ieei_offsets[PLOT_CONFIGS[case_title]["preceding_case"][0]][PLOT_CONFIGS[case_title]["preceding_case"][1]]["ieei_end"]
            ohc_ds = ohc_ds + ohc_offset
        
        # Compute annual means
        asr_annual = weighted_annualmean(asr_ds)
        olr_annual = weighted_annualmean(olr_ds)
        ts_annual = weighted_annualmean(ts_ds)
        eei_annual = asr_annual - olr_annual
        ohc_annual = weighted_annualmean(ohc_ds) if ohc_ds is not None else None

        # Compute decadal means
        asr_decadal = compute_decadal2(asr_ds)
        olr_decadal = compute_decadal2(olr_ds)
        ts_decadal = compute_decadal2(ts_ds)
        eei_decadal = asr_decadal - olr_decadal
        ohc_decadal = compute_decadal2(ohc_ds) if ohc_ds is not None else None
        
        # Compute iEEI starting from start_year_2
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, start_year_2)
        if "preceding_case" in PLOT_CONFIGS[case_title] and PLOT_CONFIGS[case_title]["preceding_case"] is not None:
            ieei_ds = ieei_ds + ieei_offset
        
        # Create annual and decadal means for iEEI by grouping years
        ieei_annual = weighted_annualmean(ieei_ds)
        ieei_decadal = compute_decadal2(ieei_ds)

        # Store the ending OHC and iEEI values for the current case
        if "ens" in ohc_annual.dims:
            case_end_ohc_ieei_offsets[case_title] = {case_str: {
                "ohc_end": ohc_annual.isel(year=-1).mean(dim="ens").values,
                "ieei_end": ieei_annual.isel(year=-1).mean(dim="ens").values},
                                                    }
        else:
            case_end_ohc_ieei_offsets[case_title] = {case_str: {
                "ohc_end": ohc_annual.isel(year=-1).values,
                "ieei_end": ieei_annual.isel(year=-1).values},
                                                    }

        # Plot
        ax1, ax2, ax3 = plot_eei_timeseries(
            asr_annual, olr_annual, eei_annual, None,
            asr_decadal, olr_decadal, eei_decadal, None,
            ax=ax, fontsize=14, case_name=case_title,
            colors=colors,
        )

        # Plot iEEI, TS, and OHC
        axb, axb2 = plot_ieei_ts_ohc(
            ieei_annual=ieei_annual, ieei_decadal=ieei_decadal,
            ts_annual=ts_annual, ts_decadal=ts_decadal,
            ohc_annual=ohc_annual, ohc_decadal=ohc_decadal,
            ax=axb, colors=colors[3:], fontsize=14, time_dim="year",
        )

        ax1.set_xlim(PLOT_CONFIGS[case_title]["xlims"])
        axb.set_xlim(PLOT_CONFIGS[case_title]["xlims"])
        if "keep_left_axes" in PLOT_CONFIGS[case_title]:
            if not PLOT_CONFIGS[case_title]["keep_left_axes"]:
                ax1.set_ylabel('')
                axb.set_ylabel('')
        if "keep_right_axes" in PLOT_CONFIGS[case_title]:
            if not PLOT_CONFIGS[case_title]["keep_right_axes"]:
                ax2.set_ylabel('')
                axb2.set_ylabel('')
        # If not the last column, remove righthand side y-axis labels 
        if (idx+1) % ncols != 0:
            ax2.set_ylabel('')
            axb2.set_ylabel('')
        # If not the first column, remove lefthand side y-axis labels
        if idx % ncols != 0:
            ax1.set_ylabel('')
            axb.set_ylabel('')
        # Set y-axis limits
        if "ax1_lims" in PLOT_CONFIGS[case_title]:
            ax1.set_ylim(*PLOT_CONFIGS[case_title]["ax1_lims"])
        if "ax2_lims" in PLOT_CONFIGS[case_title]:
            ax2.set_ylim(*PLOT_CONFIGS[case_title]["ax2_lims"])

        if "axb1_lims" in PLOT_CONFIGS[case_title]:
            axb.set_ylim(*PLOT_CONFIGS[case_title]["axb1_lims"])
        if "axb2_lims" in PLOT_CONFIGS[case_title]:
            axb2.set_ylim(*PLOT_CONFIGS[case_title]["axb2_lims"])

        # Add ensemble count annotation
        n_ens = 1
        if "ens" in asr_annual.dims:
            n_ens = len(asr_annual["ens"])
        ax1.text(0.98, 0.98, f"N={n_ens}", fontsize=12, fontweight="bold", 
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right')

        # Add a horizontal line at y=0 for the EEI subplot
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax.set_facecolor("whitesmoke")
        axb.set_facecolor("whitesmoke")

        # Collect handles and labels from left column for combined legend
        if idx == 0:
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            top_handles.extend(handles1 + handles2)
            top_labels.extend(["ASR", "OLR", "EEI"])
            handles_b, labels_b = axb.get_legend_handles_labels()
            handles_b2, labels_b2 = axb2.get_legend_handles_labels()
            bottom_handles.extend(handles_b + handles_b2)
            bottom_labels.extend(["iEEI", "OHC", "Surface Temperature"])

    # Create panel labels
    panel_labels = [f"{chr(97 + i)}." for i in range(len(axes.flat))]
    for i, (ax, label) in enumerate(zip(axes.flat, panel_labels)):
        ax.text(0.02, 0.98, label, fontsize=14, fontweight="bold", 
                transform=ax.transAxes, verticalalignment='top')
    # Only keep x-axis labels on the bottom row of subplots
    for ax in axes[:-1,:].flat:
        ax.set_xlabel('')
    # Remove text objects replacing y-axis labels:
    remove_strings = ["ASR", "OLR", "iEEI", "OHC"]
    for ax in axes[:,1:].flat:
        remove_axis_text_objects(ax, remove_strings)

    # %%
    fig.savefig("figures/figure_timeseries_futureALL.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure_timeseries_futureALL.png")
    plt.close(fig)

# %%
