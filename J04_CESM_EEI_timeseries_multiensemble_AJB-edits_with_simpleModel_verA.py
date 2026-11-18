"""
J04 with Will Kamp's three-layer energy balance model overlaid on figure_timeseries_future.

CESM2 stays as in J04 (solid). The simple model (simple_model_kamp.py, from WorkingModel.ipynb) is
drawn dotted in the same color as the matching CESM2 variable, on the columns in SIMPLE_MODEL_COLUMNS.
The ARISE-SAI column uses one of the model's two SAI controllers, chosen by SAI_CONTROLLER.
Figures are saved with a "_with_simpleModel_verA" suffix so J04's own figures are not overwritten.

The future figure is drawn in several layouts (FUTURE_FIGURE_VERSIONS), each saved with its own suffix:
    verA  Historical | SSP2-4.5 | ARISE-SAI | MCB          model overlaid on the first three columns
    verB  Historical | SSP2-4.5 | ARISE-SAI                model overlaid on all three columns
    verC  Historical | SSP2-4.5 | ARISE-SAI | model        no overlays; model in its own column
    verD  Historical | SSP2-4.5 | ARISE-SAI | MCB | model  no overlays; model in its own column
"""
# %%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
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
from prism_paths import FIGURES_DIR, PLACEHOLDER_OHC_FILE, SPATIAL_AVERAGES_DIR, SPATIAL_OHC_DIR
from simple_model_kamp import simple_model_annual_fields, RUN_START_YEAR, RUN_END_YEAR
# figures/ is gitignored, so it is missing in a fresh clone; create it before any savefig
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ------------------------------------------------------------------
# Simple-model settings: change the scenario here
# ------------------------------------------------------------------
# CO2 scenario for the simple model: "ssp245", "ssp126", "ssp585", "revert1850", or "none".
# "none" holds CO2 at 280 ppm, so every simple-model quantity stays constant.
# All SSPs share the same CO2 through 2023. This changes the simple model only; the CESM2 runs are fixed.
SIMPLE_MODEL_SCENARIO = "ssp245"
# SAI controller for the ARISE-SAI column:
#   "constant_Ta"    : holds T_a at its value when the ramp begins (2035)
#   "zero_imbalance" : holds the net TOA imbalance Q at zero; T_a is free to drift
SAI_CONTROLLER = "constant_Ta"
# Columns of the future figure that show the simple model, with the SAI controller each uses (None = no SAI).
# The MCB column has none: the simple model does not simulate MCB.
SIMPLE_MODEL_COLUMNS = {
    "CESM2-WACCM-HIST": None,
    "CESM2_WACCM_SSP2-4.5": None,
    "ARISE-SAI": SAI_CONTROLLER,
}
# Opacity of the simple-model dotted curves (1 = opaque, 0 = invisible). The white outline behind each
# dotted curve uses the same alpha, so lowering it also fades the outline.
SIMPLE_MODEL_ALPHA = 1.0
# Column titles that differ from the case label
COLUMN_TITLES = {"CESM2-WACCM-HIST": "Historical"}
FIGURE_SUFFIX = "_with_simpleModel_verA"

# ------------------------------------------------------------------
# Versions of the future time-series figure (figure_timeseries_future_with_simpleModel_<ver>_...)
# ------------------------------------------------------------------
# Name used in a column list for a column showing the three-layer model alone (no CESM2)
SIMPLE_MODEL_COLUMN = "THREE_LAYER_MODEL"
# columns            : left to right; CESM2 case labels from PLOT_CONFIGS2, or SIMPLE_MODEL_COLUMN
# overlay_simple_model: draw the model dotted over the CESM2 columns listed in SIMPLE_MODEL_COLUMNS
FUTURE_FIGURE_VERSIONS = {
    # The original: model overlaid on Historical, SSP2-4.5 and ARISE-SAI; MCB column without the model
    "verA": dict(
        columns=["CESM2-WACCM-HIST", "CESM2_WACCM_SSP2-4.5", "ARISE-SAI", "CESM2_WACCM_SSP2-4.5_MCB"],
        overlay_simple_model=True,
    ),
    # verA without the MCB column
    "verB": dict(
        columns=["CESM2-WACCM-HIST", "CESM2_WACCM_SSP2-4.5", "ARISE-SAI"],
        overlay_simple_model=True,
    ),
    # No overlays; the MCB column is replaced by a model-only column
    "verC": dict(
        columns=["CESM2-WACCM-HIST", "CESM2_WACCM_SSP2-4.5", "ARISE-SAI", SIMPLE_MODEL_COLUMN],
        overlay_simple_model=False,
    ),
    # No overlays; MCB column kept and a fifth, model-only column added
    "verD": dict(
        columns=["CESM2-WACCM-HIST", "CESM2_WACCM_SSP2-4.5", "ARISE-SAI", "CESM2_WACCM_SSP2-4.5_MCB", SIMPLE_MODEL_COLUMN],
        overlay_simple_model=False,
    ),
}
# Which versions to draw (each is saved as its own PNG)
FUTURE_FIGURE_VERSIONS_TO_MAKE = ["verA", "verB", "verC", "verD"]
# What the model-only column shows: the same two runs the overlays use (no SAI, and the SAI_CONTROLLER
# run), told apart by line style. Years shown: the same 2015-2100 window as the SSP2-4.5 and ARISE columns.
# Change to (1850, 2100) to show the model's full run instead.
SIMPLE_MODEL_COLUMN_RUNS = [(None, ":"), (SAI_CONTROLLER, "--")]  # (sai_controller, linestyle)
SIMPLE_MODEL_COLUMN_XLIMS = (2015, 2100)

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
    show_left_label=True,
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
    show_left_label : bool, default True
        Whether to draw the "ASR, OLR" label on the left-hand y-axis
    
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

    # define annual mean line width and alpha for individual ensemble members
    ln_wdth_annual = 0.5
    alpha_annual = 0.4
    
    # Plot annual means from individual ensemble mebers (thin, semi-transparent)
    if "ens" in asr_annual.dims:
        for ens_member in asr_annual["ens"].values:
            ax1.plot(asr_annual[time_dim], asr_annual.sel(ens=ens_member), color=asr_color, linestyle="-", 
                     linewidth=ln_wdth_annual, alpha=alpha_annual)
            ax1.plot(olr_annual[time_dim], olr_annual.sel(ens=ens_member), color=olr_color, linestyle="-", 
                     linewidth=ln_wdth_annual, alpha=alpha_annual)
            ax2.plot(eei_annual[time_dim], eei_annual.sel(ens=ens_member), color=eei_color, linestyle="-", 
                     linewidth=ln_wdth_annual, alpha=alpha_annual)
        # Now convert to the ensemble mean for the next step
        asr_decadal = asr_decadal.mean(dim="ens")
        olr_decadal = olr_decadal.mean(dim="ens")
        eei_decadal = eei_decadal.mean(dim="ens")
    else:
        ax1.plot(asr_annual[time_dim], asr_annual, color=asr_color, linestyle="-", 
                    linewidth=ln_wdth_annual, alpha=alpha_annual)
        ax1.plot(olr_annual[time_dim], olr_annual, color=olr_color, linestyle="-", 
                    linewidth=ln_wdth_annual, alpha=alpha_annual)
        ax2.plot(eei_annual[time_dim], eei_annual, color=eei_color, linestyle="-", 
                    linewidth=ln_wdth_annual, alpha=alpha_annual)
    
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
    # Two-colored label drawn as text, so set_ylabel('') cannot remove it; use show_left_label instead
    if show_left_label:
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
    show_left_label=True,
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax1 = ax
    ax2 = ax.twinx()
    ax1.grid(True, alpha=0.3)
    ieei_color = colors[0]
    ts_color = colors[1]
    ohc_color = colors[2]

    ln_wdth_annual = 0.5
    alpha_annual = 0.4
    ln_wdth_decadal = 2.5
    alpha_decadal = 1

    # Plot annual means from individual ensemble mebers (thin, semi-transparent)
    if "ens" in ieei_annual.dims:
        for ens_member in ieei_annual["ens"].values:
            ax1.plot(ieei_annual[time_dim], ieei_annual.sel(ens=ens_member), color=ieei_color,          
                     linestyle="-", linewidth=ln_wdth_annual, alpha=alpha_annual)
            ax2.plot(ts_annual[time_dim], ts_annual.sel(ens=ens_member), color=ts_color, linestyle="-", 
                     linewidth=ln_wdth_annual, alpha=alpha_annual)
            ax1.plot(ohc_annual[time_dim], ohc_annual.sel(ens=ens_member), color=ohc_color, linestyle="-", 
                     linewidth=ln_wdth_annual, alpha=alpha_annual)
        # Now convert to the ensemble mean for the next step
        ieei_decadal = ieei_decadal.mean(dim="ens")
        ts_decadal = ts_decadal.mean(dim="ens")
        ohc_decadal = ohc_decadal.mean(dim="ens")
    else:
        ax1.plot(ieei_annual[time_dim], ieei_annual, color=ieei_color, linestyle="-", 
                    linewidth=ln_wdth_annual, alpha=alpha_annual)
        ax2.plot(ts_annual[time_dim], ts_annual, color=ts_color, linestyle="-", 
                    linewidth=ln_wdth_decadal, alpha=alpha_decadal)
        ax1.plot(ohc_annual[time_dim], ohc_annual, color=ohc_color, linestyle="-", 
                    linewidth=ln_wdth_annual, alpha=alpha_annual)

    # Plot decadal means (thick, solid)
    ax1.plot(ieei_decadal[time_dim], ieei_decadal, color=ieei_color, linestyle="-", 
                linewidth=2.5, alpha=1, label="iEEI (decadal)")
    ax2.plot(ts_decadal[time_dim], ts_decadal, color=ts_color, linestyle="-", 
                linewidth=2.5, alpha=1, label="Surface Temperature (decadal)")
    ax1.plot(ohc_decadal[time_dim], ohc_decadal, color=ohc_color, linestyle="-", 
                linewidth=2.5, alpha=1, label="OHC (decadal)")

    ax1.set_xlabel("Year", fontsize=fontsize)
    # Two-colored label drawn as text, so set_ylabel('') cannot remove it; use show_left_label instead
    if show_left_label:
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


PLACEHOLDER_OHC_LABEL = "PLACEHOLDER OHC: same single CESM2-WACCM series (2035-2084) used for every case; not physically meaningful"


def build_placeholder_ohc_dict(case_configs, ts_dict, ohc_file, ocean_area_m2):
    """
    Build a stand-in for the output of load_data_with_configs() for OHC, using ONE global OHC
    time series for every case. PLACEHOLDER ONLY - remove once the real per-case OHC files exist.

    The returned structure mirrors the real loader: ohc_dict[case_label][case_str] is a Dataset with
      OHC_global_mean (ens, time, ohc_depth) [J m-2 over the ocean]  and  attrs["ocean_area_m2"] [m2]
    so that the downstream lines  ohc_ds["OHC_global_mean"].sel(ohc_depth=-1) * ocean_area_m2 / earth_SA_cam
    work unchanged.

    The 600 monthly values (2035-2084) are laid onto the first 600 months of each case's own
    atmosphere time axis (ts_dict[case_label][case_str]), with NaN afterwards. Without this the series would
    plot at 2035-2084 on x-axes spanning 850 CE or 1850, which is even more misleading.
    The ens dimension is copied from the atmosphere data so the ensemble-handling code still runs.

    Parameters
    ----------
    case_configs : dict, CASE_CONFIGS-style dict (uses the case labels and "subdir_cases")
    ts_dict      : dict, data_dict["TS"] from load_data_with_configs (atmosphere, already loaded)
    ohc_file     : str, path to cesm2_ohc_global_timeseries.nc (variable OHC_global [J m-2])
    ocean_area_m2: float, ocean surface area [m2] used to convert per-ocean-area OHC to a global-mean flux
    """
    logging.warning("=" * 78)
    logging.warning("USING PLACEHOLDER OHC DATA - " + PLACEHOLDER_OHC_LABEL)
    logging.warning("=" * 78)

    # decode_times=False because the file uses a noleap calendar; only the values are needed
    ohc_global_jm2 = xr.open_dataset(ohc_file, decode_times=False)["OHC_global"].values  # shape (600,)

    placeholder = {}
    for case_label, cfg in case_configs.items():
        placeholder[case_label] = {}
        for case_str in cfg["subdir_cases"]:
            # Skip cases the atmosphere loader could not find
            if case_label not in ts_dict or case_str not in ts_dict[case_label]:
                continue
            ts_ds = ts_dict[case_label][case_str]["TS"]
            time = ts_ds["time"]
            n_months = min(len(time), len(ohc_global_jm2))

            # Series padded with NaN beyond the 600 available months, shape (time,)
            series = np.full(len(time), np.nan, dtype="float64")
            series[:n_months] = ohc_global_jm2[:n_months]

            # shape (time, ohc_depth=2). The real code calls .squeeze() and then .sel(ohc_depth=-1), so a length-1
            # ohc_depth would be squeezed away and break .sel(). Depth -1 (full depth) holds the series;
            # the extra level 0 is NaN filler so ohc_depth keeps a real index.
            ohc_da = xr.DataArray(
                np.stack([np.full_like(series, np.nan), series], axis=1),
                dims=("time", "ohc_depth"),
                coords={"time": time.values, "ohc_depth": [0, -1]},
            )
            # Copy the ensemble dimension from the atmosphere data, shape (ens, time, ohc_depth)
            if "ens" in ts_ds.dims:
                ohc_da = ohc_da.expand_dims(ens=ts_ds["ens"].values)

            ds = ohc_da.to_dataset(name="OHC_global_mean")
            ds.attrs["ocean_area_m2"] = ocean_area_m2
            ds.attrs["PLACEHOLDER"] = PLACEHOLDER_OHC_LABEL
            placeholder[case_label][case_str] = ds
    return placeholder


def _overlay_axes(host):
    """
    Return an invisible axes stacked on top of every existing twin of `host`, sharing both of its axes.

    Why this is needed: zorder only orders artists *within* one Axes. Twinned axes are drawn one after
    another in creation order, so everything on ax2 (twinx of ax1) is painted over everything on ax1,
    whatever the zorder. A simple-model curve on ax1 (ASR, OLR) or axb (iEEI, OHC) was therefore covered
    by the CESM2 curves on ax2 (EEI) or axb2 (surface temperature), white outline included.

    The overlay is created last, so the figure draws it last. It shares x and y with `host`, so curves
    plotted on it use host's data coordinates, and any later set_xlim/set_ylim or autoscaling of `host`
    carries over to it. set_axis_off() hides its spines, ticks, labels and background patch, so only
    the plotted lines are visible.

    Do NOT build this with host.twinx(): twinx() also calls host.yaxis.tick_left(), which moves the
    ticks of a right-hand axes (ax2, axb2) onto the left spine, on top of ax1/axb's ticks.
    Instead, add a new subplot in the same grid cell (the same SubplotSpec, so it follows the
    figure layout exactly as the twins do) that shares both axes with host.
    """
    overlay = host.figure.add_subplot(host.get_subplotspec(), sharex=host, sharey=host)
    overlay.set_axis_off()
    return overlay


def overlay_simple_model(ax1, ax2, axb, axb2, model, years, colors, alpha=SIMPLE_MODEL_ALPHA, linestyle=":"):
    """
    Draw simple-model annual means as dotted lines over one column of the future figure.

    Each variable uses the color of its CESM2 curve. The white outline keeps the dots visible where
    they cross a solid curve of the same color. Every curve goes on its own overlay axes created after
    all the CESM2 axes (see _overlay_axes), so the outlined curves sit above all CESM2 lines in every
    panel, not only the ones whose CESM2 data share an axes with them. The model has no internal
    variability, so annual means need no decadal smoothing.

    Parameters
    ----------
    ax1, ax2 : top-row axes for ASR/OLR [W m-2] and EEI [W m-2]
    axb, axb2 : bottom-row axes for iEEI/OHC [J m-2] and surface temperature [K]
    model : dict from simple_model_kamp.simple_model_annual_fields
    years : array of CESM2 years in this column; the model is limited to the same period
    colors : the six J04 colors, ordered ASR, OLR, EEI, iEEI, surface temperature, OHC
    alpha : opacity of the dotted curves and their white outlines (default SIMPLE_MODEL_ALPHA)
    linestyle : line style of every curve; ":" everywhere except the second run in a model-only column
    """
    in_period = (model["year"] >= years.min()) & (model["year"] <= years.max())
    model_years = model["year"][in_period]
    # The stroke copies the line's graphics context, so it inherits `alpha` too
    outline = [path_effects.withStroke(linewidth=4, foreground="white")]
    style = dict(linestyle=linestyle, linewidth=2.5, alpha=alpha, path_effects=outline)

    # One overlay per CESM2 axes, each sharing that axes' y scale: ASR/OLR [W m-2], EEI [W m-2],
    # iEEI/OHC [J m-2], surface temperature [K]
    over1, over2 = _overlay_axes(ax1), _overlay_axes(ax2)
    overb, overb2 = _overlay_axes(axb), _overlay_axes(axb2)

    over1.plot(model_years, model["asr_W_m2"][in_period], color=colors[0], **style)
    over1.plot(model_years, model["olr_W_m2"][in_period], color=colors[1], **style)
    over2.plot(model_years, model["eei_W_m2"][in_period], color=colors[2], **style)
    overb.plot(model_years, model["ieei_J_m2"][in_period], color=colors[3], **style)
    overb2.plot(model_years, model["ts_K"][in_period], color=colors[4], **style)
    overb.plot(model_years, model["ohc_J_m2"][in_period], color=colors[5], **style)


def plot_simple_model_column(ax, axb, runs, colors, xlims, fontsize=14, show_right_label=True):
    """
    Fill one column of the future figure with the three-layer model alone (no CESM2), styled like the
    CESM2 columns so it can sit beside them (verC, verD).

    The axes layout copies plot_eei_timeseries / plot_ieei_ts_ohc: top row ASR, OLR on the left axis and
    EEI on a twinned right axis [W m-2]; bottom row iEEI, OHC on the left [J m-2] and surface
    temperature on a twinned right axis [K]. The curves are drawn by overlay_simple_model, so they look
    exactly like the model curves overlaid on the CESM2 columns in verA/verB.

    This column is never the first one, so it has no left-hand axis titles (the two-colored
    "ASR, OLR" / "iEEI, OHC" titles appear only on the first column).

    Parameters
    ----------
    ax, axb : top- and bottom-row axes of the column
    runs : list of (model, linestyle) pairs, model a dict from simple_model_annual_fields
    colors : the six J04 colors, ordered ASR, OLR, EEI, iEEI, surface temperature, OHC
    xlims : (first year, last year) shown; the model is limited to the same period
    show_right_label : draw the EEI and surface-temperature axis titles (True on the last column)

    Returns
    -------
    ax2, axb2 : the twinned right-hand axes (EEI [W m-2], surface temperature [K]), for setting limits
    """
    ax2 = ax.twinx()
    axb2 = axb.twinx()
    for model, linestyle in runs:
        overlay_simple_model(ax, ax2, axb, axb2, model, np.asarray(xlims), colors, linestyle=linestyle)

    # Right-hand axes: tick labels always colored like the CESM2 columns; titles only on the last column
    ax2.tick_params(axis="y", labelcolor=colors[2])
    axb2.tick_params(axis="y", labelcolor=colors[4])
    if show_right_label:
        ax2.set_ylabel("EEI [Wm$^{-2}$]", fontsize=fontsize, color=colors[2])
        axb2.set_ylabel("Surface Temperature [K]", fontsize=fontsize, color=colors[4])

    for a in (ax, axb):
        a.set_xlabel("Year", fontsize=fontsize)
        a.set_xlim(xlims)
        a.grid(True, alpha=0.3)
        a.set_facecolor("whitesmoke")
    ax.set_title("Three Layer model", fontsize=fontsize)
    # Zero line in the EEI color, as in the CESM2 columns
    ax2.axhline(0, color=colors[2], linestyle="--", linewidth=1)
    return ax2, axb2


# %%

if __name__ == "__main__":
    # Spatially averaged data (J02 output). Case folders (CESM2_LME, ...) sit directly inside it.
    # Laptop: <repo>/data/RadInt_procdata; Casper: PRISM_data/spatial_averages_data
    data_root = f"{SPATIAL_AVERAGES_DIR}/"
    # OHC time series (J09/J10 output): one *.pop.h.OHC.*.nc file per TEMP file, holding
    # OHC_global_mean (time, ohc_depth) [J m-2] and attrs["ocean_area_m2"].
    # On Casper, PRISM_data/spatial_OHC_data can be a symlink to Jonah's copy (see USE_PLACEHOLDER_OHC below).
    data_root_ohc = f"{SPATIAL_OHC_DIR}/"

    CASE_CONFIGS_ATM = deepcopy(CASE_CONFIGS_TEMPLATE)
    CASE_CONFIGS_OCN = deepcopy(CASE_CONFIGS_TEMPLATE)
    # J09 writes ARISE-1.0 OHC to spatial_OHC_data/ARISE-1.0/ (its CASE_CONFIG key), not to the
    # ARISE_SAI/ARISE-SAI-1.0/ layout of the atmosphere data, so the ocean path differs here
    CASE_CONFIGS_OCN["ARISE-1.0"]["path"] = "ARISE-1.0/"

    for config in CASE_CONFIGS_ATM.values():
        config["path"] = str(Path(data_root) / config.pop("path"))
    for config in CASE_CONFIGS_OCN.values():
        config["path"] = str(Path(data_root_ohc) / config.pop("path"))

    # %%
    # Load the data using the generalized loading function
    data_varlist = ['FLNT', 'FSNT', 'TS']
    year_dim = "time"
    # J09 OHC files are found by the "OHC" in their names (*.pop.h.OHC.*.nc), but only the global-mean
    # series is kept: the full OHC map (ohc_depth x 384 x 320 per month) is never used here and,
    # across all cases and members, is large enough to crash the Jupyter kernel.
    ohc_varlist = ["OHC"]
    ohc_keep_vars = ["OHC_global_mean"]

    # Load data using the generalized function (requires 4 nodes on CURC)
    data_dict = {}
    for var in data_varlist:
        data_dict[var] = load_data_with_configs(CASE_CONFIGS_ATM, [var], year_dim=year_dim, load_into_memory=False)

    # ------------------------------------------------------------------
    # OHC source: real per-case files (J09 output) or the PLACEHOLDER
    # The real files are used whenever data_root_ohc exists and contains anything. On Casper,
    # point it at Jonah's copy with a symlink (no copy, no quota):
    #   ln -s /glade/work/jonahshaw/PRISM_data/spatial_OHC_data /glade/work/$USER/PRISM_data/spatial_OHC_data
    # Otherwise (e.g. the laptop) EVERY case is given the SAME single OHC series
    # (cesm2_ohc_global_timeseries.nc, one CESM2-WACCM run, 2035-2084) so that the script runs;
    # those OHC curves are NOT physically meaningful. Force either choice by setting this to True/False.
    # ------------------------------------------------------------------
    USE_PLACEHOLDER_OHC = not (Path(data_root_ohc).is_dir() and any(Path(data_root_ohc).iterdir()))
    placeholder_ohc_file = str(PLACEHOLDER_OHC_FILE)
    # Approximate global ocean surface area [m^2] (~71% of the 5.1e14 m^2 Earth surface); placeholder value only
    placeholder_ocean_area_m2 = 3.6e14

    if USE_PLACEHOLDER_OHC:
        ohc_dict = build_placeholder_ohc_dict(
            CASE_CONFIGS_OCN, data_dict["TS"], placeholder_ohc_file, placeholder_ocean_area_m2
        )
    else:
        logging.info(f"Loading real OHC data from {data_root_ohc}")
        ohc_dict = load_data_with_configs(
            CASE_CONFIGS_OCN, ohc_varlist, year_dim=year_dim, load_into_memory=False, keep_vars=ohc_keep_vars
        )
        # load_data_with_configs skips (with only a warning) any case it finds no files for, and the
        # plotting loops below would then stop with a bare KeyError. List every missing case up front instead.
        missing_ohc = [
            f"{case_label}: {case_str}  (searched {config['path']})"
            for case_label, config in CASE_CONFIGS_OCN.items()
            for case_str in config["subdir_cases"]
            if case_str not in ohc_dict.get(case_label, {})
        ]
        if missing_ohc:
            raise FileNotFoundError(
                "No OHC files found for these cases (expected *<case_str>*.OHC.*nc from J09):\n  "
                + "\n  ".join(missing_ohc)
                + "\nSet USE_PLACEHOLDER_OHC = True to run with the placeholder instead."
            )

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
        # Full-depth (ohc_depth=-1) global-mean OHC [J m-2 of ocean] -> [J m-2 of Earth surface].
        # .compute() only this small series, like the atmosphere variables: the real OHC is a lazy dask
        # array with one chunk per file, and the 10-year rolling mean in compute_decadal2 fails on chunks
        # shorter than 10 years
        ohc_case = ohc_dict[case_label][case_str].squeeze()
        ohc_ds = ohc_case["OHC_global_mean"].sel(ohc_depth=-1).compute() * ohc_case.attrs["ocean_area_m2"] / earth_SA_cam
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
    if USE_PLACEHOLDER_OHC:
        fig.text(0.5, 0.995, PLACEHOLDER_OHC_LABEL, ha="center", va="top", color="red", fontsize=14, fontweight="bold")
    fig.savefig(FIGURES_DIR / f"figure_timeseries_past{FIGURE_SUFFIX}.png", dpi=300, bbox_inches='tight')
    logging.info(f"Saved figure_timeseries_past{FIGURE_SUFFIX}.png")
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
    # The zero-imbalance controller takes model ASR and OLR down to ~233.5 W/m2 by 2099, below the usual
    # 236 limit. Both axes keep a 13-unit span, so one tick on the left equals one tick on the right.
    if SAI_CONTROLLER == "zero_imbalance":
        ALL_SHARED_PLOT_LIMITS.update(ax1_lims=(232, 245), ax2_lims=(-9, 4))
    SOME_SHARED_PLOT_LIMITS = {
        "xlims": (2015, 2100),
        "keep_left_axes": False,
        "keep_right_axes": False,
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
    # PLOT 2: Future scenarios, drawn once per layout in FUTURE_FIGURE_VERSIONS_TO_MAKE (verA, verB, ...)
    # Integration starts from year 1850
    logging.info(f"Creating Plot 2: Future scenarios, versions {FUTURE_FIGURE_VERSIONS_TO_MAKE}")
    PLOT_CONFIGS = PLOT_CONFIGS2
    # One continuous 1850-2100 run: its iEEI and OHC are relative to 1850, so columns join with no offset
    # One run per controller setting in use (None = no SAI), keyed by that setting. The model-only
    # column (verC, verD) uses the same runs as the overlays.
    simple_model_runs = {
        controller: simple_model_annual_fields(SIMPLE_MODEL_SCENARIO, sai_controller=controller)
        for controller in set(SIMPLE_MODEL_COLUMNS.values()) | {c for c, _ in SIMPLE_MODEL_COLUMN_RUNS}
    }

    start_year_2 = 1850
    ts_var = "TS"
    olr_var = "FLNT"
    asr_var = "FSNT"
    colors = sns.color_palette("colorblind", n_colors=6)

    # -- CESM2 annual and decadal series, computed once and reused by every version --------------
    # Every CESM2 column any version draws. The historical run comes first: the later cases list it as
    # their "preceding_case" and add its final OHC and iEEI values to their own, for continuity.
    cesm2_cases = []
    for version in FUTURE_FIGURE_VERSIONS_TO_MAKE:
        for case_label in FUTURE_FIGURE_VERSIONS[version]["columns"]:
            if case_label != SIMPLE_MODEL_COLUMN and case_label not in cesm2_cases:
                cesm2_cases.append(case_label)
    cesm2_cases.sort(key=lambda label: PLOT_CONFIGS[label]["preceding_case"] is not None)

    case_end_ohc_ieei_offsets = {}  # Dictionary to store the ending OHC and iEEI values for each case
    cesm2_series = {}  # case_label -> the annual and decadal series plotted in its column

    for case_label in cesm2_cases:
        logging.info(f"Computing series for case: {case_label}")

        # Get case_str from config
        case_str = PLOT_CONFIGS[case_label]["case_str"]

        # Extract ASR and OLR data
        asr_ds = data_dict[asr_var][case_label][case_str][asr_var].sel(spatial="G").squeeze().compute()
        olr_ds = data_dict[olr_var][case_label][case_str][olr_var].sel(spatial="G").squeeze().compute()
        ts_ds = data_dict[ts_var][case_label][case_str][ts_var].sel(spatial="G").squeeze().compute()
        # Full-depth (ohc_depth=-1) global-mean OHC [J m-2 of ocean] -> [J m-2 of Earth surface].
        # .compute() only this small series, like the atmosphere variables: the real OHC is a lazy dask
        # array with one chunk per file, and the 10-year rolling mean in compute_decadal2 fails on chunks
        # shorter than 10 years
        ohc_case = ohc_dict[case_label][case_str].squeeze()
        ohc_ds = ohc_case["OHC_global_mean"].sel(ohc_depth=-1).compute() * ohc_case.attrs["ocean_area_m2"] / earth_SA_cam
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

        cesm2_series[case_label] = dict(
            asr_annual=asr_annual, olr_annual=olr_annual, eei_annual=eei_annual, ts_annual=ts_annual,
            ohc_annual=ohc_annual, ieei_annual=ieei_annual,
            asr_decadal=asr_decadal, olr_decadal=olr_decadal, eei_decadal=eei_decadal, ts_decadal=ts_decadal,
            ohc_decadal=ohc_decadal, ieei_decadal=ieei_decadal,
        )

    # -- One figure per version ----------------------------------------------------------------------
    for version in FUTURE_FIGURE_VERSIONS_TO_MAKE:
        columns = FUTURE_FIGURE_VERSIONS[version]["columns"]
        overlay = FUTURE_FIGURE_VERSIONS[version]["overlay_simple_model"]
        n_cols = len(columns)
        logging.info(f"Drawing future figure {version}: {columns}")

        # 5 inches per column, as in the original 4-column (20 x 10 in) layout
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
        # Inner y-axis titles are hidden, so the gaps can be 20% narrower than the wspace=0.35 layout. wspace is a
        # fraction of panel width, and panels widen as gaps shrink: 0.8 * 0.35 / (1 + 3*0.35*0.2/4) = 0.266
        # (0.267 for 3 columns, 0.265 for 5: the same to plotting precision)
        fig.subplots_adjust(wspace=0.266)

        for idx, (ax, axb, case_label) in enumerate(zip(axes[0], axes[1], columns)):
            # y-axis titles only on the outer columns: left titles on the first, right titles on the last.
            # This reproduces verA's keep_left_axes / keep_right_axes, and moves the right-hand titles to
            # whichever column ends a version (ARISE-SAI in verB, the model column in verC and verD).
            keep_left = idx == 0
            keep_right = idx == n_cols - 1

            if case_label == SIMPLE_MODEL_COLUMN:
                logging.info("Plotting the three-layer model column")
                ax2, axb2 = plot_simple_model_column(
                    ax, axb,
                    runs=[(simple_model_runs[controller], ls) for controller, ls in SIMPLE_MODEL_COLUMN_RUNS],
                    colors=colors, xlims=SIMPLE_MODEL_COLUMN_XLIMS, fontsize=14, show_right_label=keep_right,
                )
                # Same y-limits as the CESM2 future columns, so the model column can be read across
                ax.set_ylim(*ALL_SHARED_PLOT_LIMITS["ax1_lims"])
                ax2.set_ylim(*ALL_SHARED_PLOT_LIMITS["ax2_lims"])
                axb.set_ylim(*ALL_SHARED_PLOT_LIMITS["axb1_lims"])
                axb2.set_ylim(*ALL_SHARED_PLOT_LIMITS["axb2_lims"])
                continue

            logging.info(f"Plotting case: {case_label}")
            s = cesm2_series[case_label]

            # Plot
            ax1, ax2, ax3 = plot_eei_timeseries(
                s["asr_annual"], s["olr_annual"], s["eei_annual"], None,
                s["asr_decadal"], s["olr_decadal"], s["eei_decadal"], None,
                ax=ax, fontsize=14, case_name=COLUMN_TITLES.get(case_label, case_label),
                colors=colors, show_left_label=keep_left,
            )

            # Plot iEEI, TS, and OHC
            axb, axb2 = plot_ieei_ts_ohc(
                ieei_annual=s["ieei_annual"], ieei_decadal=s["ieei_decadal"],
                ts_annual=s["ts_annual"], ts_decadal=s["ts_decadal"],
                ohc_annual=s["ohc_annual"], ohc_decadal=s["ohc_decadal"],
                ax=axb, colors=colors[3:], fontsize=14, time_dim="year",
                show_left_label=keep_left,
            )

            if overlay and case_label in SIMPLE_MODEL_COLUMNS:
                overlay_simple_model(
                    ax1, ax2, axb, axb2, simple_model_runs[SIMPLE_MODEL_COLUMNS[case_label]], s["asr_annual"]["year"].values, colors
                )

            ax1.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
            axb.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
            if not keep_left:
                ax1.set_ylabel('')
                axb.set_ylabel('')
            if not keep_right:
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
            if "ens" in s["asr_annual"].dims:
                n_ens = len(s["asr_annual"]["ens"])
            ax1.text(0.98, 0.98, f"N={n_ens}", fontsize=12, fontweight="bold",
                    transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right')

            # Add a horizontal line at y=0 for the EEI subplot
            # Drawn in the EEI color to show it belongs to the right-hand axis
            ax2.axhline(0, color=colors[2], linestyle='--', linewidth=1)
            ax.set_facecolor("whitesmoke")
            axb.set_facecolor("whitesmoke")

        # Create panel labels
        panel_labels = [f"{chr(97 + i)}." for i in range(len(axes.flat))]
        for i, (ax, label) in enumerate(zip(axes.flat, panel_labels)):
            ax.text(0.02, 0.98, label, fontsize=14, fontweight="bold",
                    transform=ax.transAxes, verticalalignment='top')

        # Line-style key; the colors are already identified by the colored y-axis titles
        legend_handles = [Line2D([], [], color="black", linestyle="-", linewidth=2.5, label="CESM2")]
        if overlay:
            legend_handles.append(Line2D([], [], color="black", linestyle=":", linewidth=2.5,
                                         label=f"Three Layer model ({SIMPLE_MODEL_SCENARIO}; ARISE-SAI column: {SAI_CONTROLLER} controller)"))
        if SIMPLE_MODEL_COLUMN in columns:
            for controller, ls in SIMPLE_MODEL_COLUMN_RUNS:
                run_label = "no SAI" if controller is None else f"SAI, {controller} controller"
                legend_handles.append(Line2D([], [], color="black", linestyle=ls, linewidth=2.5,
                                             label=f"Three Layer model ({SIMPLE_MODEL_SCENARIO}, {run_label})"))
        fig.legend(
            handles=legend_handles,
            loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(legend_handles), frameon=False, fontsize=14,
        )

        if USE_PLACEHOLDER_OHC:
            fig.text(0.5, 0.995, PLACEHOLDER_OHC_LABEL, ha="center", va="top", color="red", fontsize=14, fontweight="bold")
        # verA keeps its original file name (FIGURE_SUFFIX ends in _verA)
        future_suffix = FIGURE_SUFFIX.replace("_verA", f"_{version}")
        out_name = f"figure_timeseries_future{future_suffix}_{SIMPLE_MODEL_SCENARIO}_{SAI_CONTROLLER}.png"
        fig.savefig(FIGURES_DIR / out_name, dpi=300, bbox_inches='tight')
        logging.info(f"Saved {out_name}")
        plt.close(fig)
    # %%

    # ------------------------------------------------------------------
    # Standalone figure: simple-model earth albedo and surface temperature, 1850-2100
    # ------------------------------------------------------------------
    # Simple model only (no CESM2). Uses the no-SAI run for SIMPLE_MODEL_SCENARIO, the same run drawn
    # in the Historical and SSP2-4.5 columns above, so it spans the full analysis period in one piece.
    # Note: alpha_earth = ALBEDO_A - ALBEDO_B * T_a (simple_model_kamp.alpha_earth, Budyko 1969-style
    # linear fit). Unless T_a reaches the clip limits, the albedo curve is an exact mirror image of
    # the temperature curve; the two y-axes only differ in scale.
    albedo_run = simple_model_annual_fields(SIMPLE_MODEL_SCENARIO, sai_controller=None)
    albedo_years = albedo_run["year"]  # calendar years; RUN_START_YEAR .. RUN_END_YEAR - 1

    fig, ax_alb = plt.subplots(1, 1, figsize=(8, 5))
    ax_ts = ax_alb.twinx()  # right y-axis for surface temperature [K], as in the bottom-row panels

    # Same colors as the future figure: colorblind palette, surface temperature = index 4 (pink).
    # Albedo is a new variable, so it takes sky blue (index 9 of the 10-color colorblind palette),
    # a color not used by any other J04 variable.
    palette_10 = sns.color_palette("colorblind", n_colors=10)
    ts_color = palette_10[4]
    albedo_color = palette_10[9]

    # Same style as the simple-model curves on the other figures: dotted, white outline, toggled
    # opacity. Each curve sits on its own overlay axes so neither can be hidden by the other twin.
    outline = [path_effects.withStroke(linewidth=4, foreground="white")]
    sm_style = dict(linestyle=":", linewidth=2.5, alpha=SIMPLE_MODEL_ALPHA, path_effects=outline)
    _overlay_axes(ax_alb).plot(albedo_years, albedo_run["alpha_earth"], color=albedo_color, **sm_style)
    _overlay_axes(ax_ts).plot(albedo_years, albedo_run["ts_K"], color=ts_color, **sm_style)

    # Axis formatting matches plot_ieei_ts_ohc: fontsize 14, colored y-axis titles and tick labels,
    # light grid, whitesmoke background
    fontsize = 14
    ax_alb.set_xlim(RUN_START_YEAR, RUN_END_YEAR)  # 1850-2100, the simple model's full run
    ax_alb.set_xlabel("Year", fontsize=fontsize)
    ax_alb.set_ylabel(r"Earth albedo, $\alpha_{earth}$ [-]", fontsize=fontsize, color=albedo_color)
    ax_alb.tick_params(axis="y", labelcolor=albedo_color)
    ax_ts.set_ylabel("Surface Temperature [K]", fontsize=fontsize, color=ts_color)
    ax_ts.tick_params(axis="y", labelcolor=ts_color)
    ax_alb.grid(True, alpha=0.3)
    ax_alb.set_facecolor("whitesmoke")
    ax_alb.set_title(f"Three Layer model ({SIMPLE_MODEL_SCENARIO}, no SAI)", fontsize=fontsize)

    fig.savefig(FIGURES_DIR / f"figure_simpleModel_albedo_ts{FIGURE_SUFFIX}_{SIMPLE_MODEL_SCENARIO}.png", dpi=300, bbox_inches="tight")
    logging.info(f"Saved figure_simpleModel_albedo_ts{FIGURE_SUFFIX}_{SIMPLE_MODEL_SCENARIO}.png")
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
        # Full-depth (ohc_depth=-1) global-mean OHC [J m-2 of ocean] -> [J m-2 of Earth surface].
        # .compute() only this small series, like the atmosphere variables: the real OHC is a lazy dask
        # array with one chunk per file, and the 10-year rolling mean in compute_decadal2 fails on chunks
        # shorter than 10 years
        ohc_case = ohc_dict[case_label][case_str].squeeze()
        ohc_ds = ohc_case["OHC_global_mean"].sel(ohc_depth=-1).compute() * ohc_case.attrs["ocean_area_m2"] / earth_SA_cam
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
    if USE_PLACEHOLDER_OHC:
        fig.text(0.5, 0.995, PLACEHOLDER_OHC_LABEL, ha="center", va="top", color="red", fontsize=14, fontweight="bold")
    fig.savefig(FIGURES_DIR / f"figure_timeseries_futureALL{FIGURE_SUFFIX}.png", dpi=300, bbox_inches='tight')
    logging.info(f"Saved figure_timeseries_futureALL{FIGURE_SUFFIX}.png")
    plt.close(fig)

# %%
