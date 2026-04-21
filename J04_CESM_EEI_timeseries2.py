# %%
from pathlib import Path
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

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

    
# %%

if __name__ == "__main__":
    CASE_CONFIGS = {
        "CESM-LME": {
            "path": "data/RadInt_procdata/CESM_LME/",
            "case_str": "BLMTRC5CN.f19_g16.003",
            "append_case": None,
            "ufunc": lambda ds: ds.sel(time=slice(None, "1849-12-31")),
            # "ufunc": None,
        },
        "CESM2-LME": {
            "path": "data/RadInt_procdata/CESM2_LME/",
            "case_str": "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002",
            "append_case": None,
            "ufunc": None,
        },
        "CESM2-LE": {
            "path": "data/RadInt_procdata/CESM2_LE/",
            "case_str": "b.e21.BHISTcmip6.f09_g17.LE2-1301.001",
            "append_case": None,
            "ufunc": None,
        },
        "CESM2-SSP2-4.5": {
            "path": "data/RadInt_procdata/CESM2_WACCM_SSP2-4.5/",
            "case_str": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
            "append_case": "CESM2-LE",
            "ufunc": lambda ds: ds.sel(time=slice(None, "2084-12-31")),
        },
        "ARISE-SAI": {
            "path": "data/RadInt_procdata/ARISE_SAI/",
            "case_str": "1p5K-SAI.001",
            "append_case": "CESM2-SSP2-4.5",
            "ufunc": None,
        },
        "CESM2-SSP2-4.5_MCB": {
            "path": "data/RadInt_procdata/CESM2_WACCM_SSP2-4.5_MCB/",
            "case_str": "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.001",
            "append_case": "CESM2-SSP2-4.5",
            "ufunc": None,
        },
        # Add new cases here when ready
    }
    asr_var = "FSNT"
    olr_var = "FLNT"
    ts_var = "TS"

    load_var_list = [asr_var, olr_var, ts_var]
    data_dict = {}

    for case_label in CASE_CONFIGS.keys():
        datapath = CASE_CONFIGS[case_label]["path"]
        case_str = CASE_CONFIGS[case_label]["case_str"]

        all_files = []
        for var in load_var_list:
            var_files = crawl_and_list_glob(datapath, f"**/*{case_str}*.{var}.*nc")
            all_files.extend(var_files)
        if len(all_files) == 0:
            logging.warning(f"No files found for case {case_label} with case string {case_str} in path {datapath}")
            continue
        
        all_ds = xr.open_mfdataset(all_files)

        # Handle the CESM time coordinate issue and challenges with cftime.DatetimeNoLeap
        if all_ds["time"][0]["time.month"] == 2:
            all_ds = all_ds.assign_coords(
                time=shift_noleap_time_back_one_month(all_ds["time"].values)
            )

        # If there is an append case specified, append the data from that case to the current dataset along the time dimension
        # e.g. for ARISE-SAI, we want to append the CESM2-SSP2-4.5 data it is branched from. We will assume that the append case has already been loaded and is available in data_dict.
        if CASE_CONFIGS[case_label]["append_case"] is not None:
            append_case_label = CASE_CONFIGS[case_label]["append_case"]
            if append_case_label not in data_dict:
                logging.warning(f"Append case {append_case_label} not found in data_dict for case {case_label}. Skipping append.")
            else:
                append_ds = data_dict[append_case_label].sel(time=slice(None, str(all_ds["time.year"][0].values - 1)))
                all_ds = xr.concat([append_ds, all_ds], dim="time")

        if CASE_CONFIGS[case_label]["ufunc"] is not None:
            all_ds = CASE_CONFIGS[case_label]["ufunc"](all_ds)

        data_dict[case_label] = all_ds

    # %%
    PLOT_CONFIGS = {
        "CESM-LME": {
            "ax1_lims": (228, 237),
            "ax2_lims": (-3, 6),
            "axb1_lims": (-1.5e24, 1.5e24),
            "axb2_lims": (286.25, 287.75),
            "axb1_yticks": np.arange(-1.5e24, 1.5e24+0.01e24, 0.5e24),
            "axb2_yticks": np.arange(286.25, 287.75+0.01, 0.25),
            "xlims": (850, 1850),
            "keep_left_axes": True,
            "keep_right_axes": True,
        },
        "CESM2-LME": {
            "ax1_lims": (235, 244),
            "ax2_lims": (-3, 6),
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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.40)
    
    case_list_1 = ["CESM-LME", "CESM2-LME"]
    start_year_1 = 850

    for ax, axb, case_label in zip(axes[0], axes[1], case_list_1):
        logging.info(f"Plotting case: {case_label}")
        
        # Extract ASR and OLR data
        asr_ds = data_dict[case_label][asr_var].sel(spatial="G")
        olr_ds = data_dict[case_label][olr_var].sel(spatial="G")
        ts_ds = data_dict[case_label][ts_var].sel(spatial="G")
        
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

        # Add a horizontal line at y=0 for the EEI subplot
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)

    # fig.savefig("figures/figure2_toprow.png", dpi=300, bbox_inches='tight')
    # logging.info("Saved figure2_toprow.png")
    # plt.close(fig)
    # %%
    PLOT_CONFIGS = {
        "CESM2-SSP2-4.5": {
            "ax1_lims": (237, 244),
            "ax2_lims": (-4, 3),
            # "axb1_lims": (0.0e24, 2.5e24),
            "axb1_lims": (-0.5e24, 2.5e24),
            "axb2_lims": (286.0, 292.0),
            "xlims": (1850, 2100),
            "keep_left_axes": True,
            "keep_right_axes": False,
        },
        "ARISE-SAI": {
            "ax1_lims": (237, 244),
            "ax2_lims": (-4, 3),
            # "axb1_lims": (0.0e24, 2.5e24),
            "axb1_lims": (-0.5e24, 2.5e24),
            "axb2_lims": (286.0, 292.0),
            "xlims": (1850, 2100),
            "keep_left_axes": False,
            "keep_right_axes": False,
        },
        "CESM2-SSP2-4.5_MCB": {
            "ax1_lims": (237, 244),
            "ax2_lims": (-4, 3),
            # "axb1_lims": (0.0e24, 2.5e24),
            "axb1_lims": (-0.5e24, 2.5e24),
            "axb2_lims": (286.0, 292.0),
            "xlims": (1850, 2100),
            "keep_left_axes": False,
            "keep_right_axes": True,
        },
    }

    # %%
    # PLOT 2: Future scenarios (CESM2-SSP2-4.5, ARISE-SAI, CESM2-SSP2-4.5_MCB)
    # Integration starts from year 1850
    logging.info("Creating Plot 2: Future scenarios (CESM2-SSP2-4.5, ARISE-SAI, CESM2-SSP2-4.5_MCB)")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(wspace=0.35)
    
    case_list_2 = ["CESM2-SSP2-4.5", "ARISE-SAI", "CESM2-SSP2-4.5_MCB"]
    start_year_2 = 1850

    for ax, axb, case_label in zip(axes[0], axes[1], case_list_2):
        logging.info(f"Plotting case: {case_label}")

        # Extract ASR and OLR data
        asr_ds = data_dict[case_label][asr_var].sel(spatial="G")
        olr_ds = data_dict[case_label][olr_var].sel(spatial="G")
        ts_ds = data_dict[case_label][ts_var].sel(spatial="G")
        
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
        
        # Compute IEEI starting from start_year_1
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, start_year_1)
        
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

    # fig.savefig("figures/figure2_bottomrow.png", dpi=300, bbox_inches='tight')
    # logging.info("Saved figure2_bottomrow.png")
    # plt.close(fig)
    # %%