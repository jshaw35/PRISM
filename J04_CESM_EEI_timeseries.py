# %%
from pathlib import Path
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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
    ax1.plot(asr_annual["year"], asr_annual, color=asr_color, linestyle="-", 
             linewidth=0.7, alpha=0.5, label="ASR (annual)")
    ax1.plot(olr_annual["year"], olr_annual, color=olr_color, linestyle="-", 
             linewidth=0.7, alpha=0.5, label="OLR (annual)")
    
    ax2.plot(eei_annual["year"], eei_annual, color=eei_color, linestyle="-", 
             linewidth=0.7, alpha=0.5, label="EEI (annual)")
    
    ax3.plot(ieei_annual["year"], ieei_annual, color=ieei_color, linestyle="-", 
             linewidth=0.7, alpha=0.5, label="IEEI (annual)")
    
    # Plot decadal means (thick, solid)
    ax1.plot(asr_decadal["year"], asr_decadal, color=asr_color, linestyle="-", 
             linewidth=2.5, label="ASR (decadal)")
    ax1.plot(olr_decadal["year"], olr_decadal, color=olr_color, linestyle="-", 
             linewidth=2.5, label="OLR (decadal)")
    
    ax2.plot(eei_decadal["year"], eei_decadal, color=eei_color, linestyle="-", 
             linewidth=2.5, label="EEI (decadal)")
    
    ax3.plot(ieei_decadal["year"], ieei_decadal, color=ieei_color, linestyle="-", 
             linewidth=2.5, label="IEEI (decadal)")
    
    # Set axis labels and colors
    ax1.set_xlabel("Year", fontsize=fontsize)
    ax1.set_ylabel("ASR, OLR [W/m²]", fontsize=fontsize, color=asr_color)
    ax1.tick_params(axis="y", labelcolor=asr_color)
    
    ax2.set_ylabel("EEI [W/m²]", fontsize=fontsize, color=eei_color)
    ax2.tick_params(axis="y", labelcolor=eei_color)
    
    ax3.set_ylabel("IEEI [W]", fontsize=fontsize, color=ieei_color)
    ax3.tick_params(axis="y", labelcolor=ieei_color)
    
    # Add title and grid
    ax1.set_title(case_name, fontsize=fontsize)
    ax1.grid(True, alpha=0.3)
    
    return ax1, ax2, ax3


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
            "ax1_lims": (230, 237),
            # "xlims": (230, 237),
            # "cbar_ticks": np.arange(850, 2005, 100),
            # "cbar_ylabel": None,
        },
        "CESM2-LME": {
            "ax1_lims": (236, 244),
            # "xlims": (236, 244),
            # "cbar_ticks": np.arange(850, 1851, 100),
            # "cbar_ylabel": None,
        },
    }

    # %%
    # PLOT 1: Historical scenarios (CESM-LME, CESM2-LME)
    # Integration starts from year 850
    logging.info("Creating Plot 1: Historical scenarios (CESM-LME, CESM2-LME)")
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.40)
    
    case_list_1 = ["CESM-LME", "CESM2-LME"]
    start_year_1 = 850
    keep_left_axes = {
        "CESM-LME": True,
        "CESM2-LME": False,
    } # only keep y-axis labels and ticks on the left subplot to avoid redundancy
    keep_right_axes = {
        "CESM-LME": False,
        "CESM2-LME": True
    } # only keep y-axis labels and ticks on the right subplot to avoid redundancy
    for ax, case_label in zip(axs, case_list_1):
        logging.info(f"Plotting case: {case_label}")
        
        # Extract ASR and OLR data
        asr_ds = data_dict[case_label][asr_var].sel(spatial="G")
        olr_ds = data_dict[case_label][olr_var].sel(spatial="G")
        
        # Compute annual means
        asr_annual = asr_ds.groupby("time.year").mean()
        olr_annual = olr_ds.groupby("time.year").mean()
        eei_annual = asr_annual - olr_annual
        
        # Compute decadal means
        asr_decadal = asr_ds.resample(time='10YE').mean().groupby("time.year").mean()
        olr_decadal = olr_ds.resample(time='10YE').mean().groupby("time.year").mean()
        eei_decadal = asr_decadal - olr_decadal
        
        # Compute IEEI starting from start_year_1
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, start_year_1)
        
        # Create annual and decadal means for IEEI by grouping years
        ieei_annual = ieei_ds.groupby("time.year").mean()
        ieei_decadal = ieei_ds.resample(time='10YE').mean().groupby("time.year").mean()
        
        # Plot
        ax1, ax2, ax3 = plot_eei_timeseries(
            asr_annual, olr_annual, eei_annual, ieei_annual,
            asr_decadal, olr_decadal, eei_decadal, ieei_decadal,
            ax=ax, fontsize=14, case_name=case_label
        )
        if not keep_left_axes[case_label]:
            ax1.set_ylabel('')
        if not keep_right_axes[case_label]:
            ax2.set_ylabel('')
            ax3.set_ylabel('')
        # Set y-axis limits
        ax1.set_ylim(*PLOT_CONFIGS[case_label]["ax1_lims"])

    # fig.savefig("figures/figure2_toprow.png", dpi=300, bbox_inches='tight')
    # logging.info("Saved figure2_toprow.png")
    # plt.close(fig)
    
    # %%
    # PLOT 2: Future scenarios (CESM2-SSP2-4.5, ARISE-SAI, CESM2-SSP2-4.5_MCB)
    # Integration starts from year 1850
    logging.info("Creating Plot 2: Future scenarios (CESM2-SSP2-4.5, ARISE-SAI, CESM2-SSP2-4.5_MCB)")
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.35)
    
    case_list_2 = ["CESM2-SSP2-4.5", "ARISE-SAI", "CESM2-SSP2-4.5_MCB"]
    start_year_2 = 1850

    keep_left_axes = {
        "CESM2-SSP2-4.5": True,
        "ARISE-SAI": False,
        "CESM2-SSP2-4.5_MCB": False
    } # only keep y-axis labels and ticks on the left subplot to avoid redundancy
    keep_right_axes = {
        "CESM2-SSP2-4.5": False,
        "ARISE-SAI": False,
        "CESM2-SSP2-4.5_MCB": True
    } # only keep y-axis labels and ticks on the right subplot to avoid redundancy

    # First pass: collect all data and compute axis limits
    logging.info("Computing shared axis limits for Plot 2")
    asr_min, asr_max = np.inf, -np.inf
    olr_min, olr_max = np.inf, -np.inf
    eei_min, eei_max = np.inf, -np.inf
    ieei_min, ieei_max = np.inf, -np.inf
    
    case_data_2 = {}
    for case_label in case_list_2:
        # Extract ASR and OLR data
        asr_ds = data_dict[case_label][asr_var].sel(spatial="G")
        olr_ds = data_dict[case_label][olr_var].sel(spatial="G")
        
        # Compute annual means
        asr_annual = asr_ds.groupby("time.year").mean()
        olr_annual = olr_ds.groupby("time.year").mean()
        eei_annual = asr_annual - olr_annual
        
        # Compute decadal means
        asr_decadal = asr_ds.resample(time='10YE').mean().groupby("time.year").mean()
        olr_decadal = olr_ds.resample(time='10YE').mean().groupby("time.year").mean()
        eei_decadal = asr_decadal - olr_decadal
        
        # Compute IEEI starting from start_year_2
        ieei_ds = compute_ieei_with_start_year(asr_ds, olr_ds, start_year_2)
        
        # Create annual and decadal means for IEEI by grouping years
        ieei_annual = ieei_ds.groupby("time.year").mean()
        ieei_decadal = ieei_ds.resample(time='10YE').mean().groupby("time.year").mean()
        
        # Store data
        case_data_2[case_label] = {
            'asr_annual': asr_annual,
            'olr_annual': olr_annual,
            'eei_annual': eei_annual,
            'ieei_annual': ieei_annual,
            'asr_decadal': asr_decadal,
            'olr_decadal': olr_decadal,
            'eei_decadal': eei_decadal,
            'ieei_decadal': ieei_decadal,
        }
        
        # Update axis limits
        asr_min = min(asr_min, asr_annual.min(), asr_decadal.min())
        asr_max = max(asr_max, asr_annual.max(), asr_decadal.max())
        olr_min = min(olr_min, olr_annual.min(), olr_decadal.min())
        olr_max = max(olr_max, olr_annual.max(), olr_decadal.max())
        eei_min = min(eei_min, eei_annual.min(), eei_decadal.min())
        eei_max = max(eei_max, eei_annual.max(), eei_decadal.max())
        ieei_min = min(ieei_min, ieei_annual.min(), ieei_decadal.min())
        ieei_max = max(ieei_max, ieei_annual.max(), ieei_decadal.max())
    
    # Add some padding to the limits for better visualization
    asr_padding = (asr_max - asr_min) * 0.05
    olr_padding = (olr_max - olr_min) * 0.05
    eei_padding = (eei_max - eei_min) * 0.05
    ieei_padding = (ieei_max - ieei_min) * 0.05
    
    asr_lims = (asr_min - asr_padding, asr_max + asr_padding)
    olr_lims = (olr_min - olr_padding, olr_max + olr_padding)
    eei_lims = (eei_min - eei_padding, eei_max + eei_padding)
    ieei_lims = (ieei_min - ieei_padding, ieei_max + ieei_padding)
    
    # Second pass: plot with shared axis limits
    axes_1 = []
    axes_2 = []
    axes_3 = []
    
    for ax, case_label in zip(axs, case_list_2):
        logging.info(f"Plotting case: {case_label}")
        
        # Retrieve stored data
        asr_annual = case_data_2[case_label]['asr_annual']
        olr_annual = case_data_2[case_label]['olr_annual']
        eei_annual = case_data_2[case_label]['eei_annual']
        ieei_annual = case_data_2[case_label]['ieei_annual']
        asr_decadal = case_data_2[case_label]['asr_decadal']
        olr_decadal = case_data_2[case_label]['olr_decadal']
        eei_decadal = case_data_2[case_label]['eei_decadal']
        ieei_decadal = case_data_2[case_label]['ieei_decadal']
        
        # Plot
        ax1, ax2, ax3 = plot_eei_timeseries(
            asr_annual, olr_annual, eei_annual, ieei_annual,
            asr_decadal, olr_decadal, eei_decadal, ieei_decadal,
            ax=ax, fontsize=14, case_name=case_label
        )
        
        # Store axes for later limit setting
        axes_1.append(ax1)
        axes_2.append(ax2)
        axes_3.append(ax3)

        if not keep_left_axes[case_label]:
            ax1.set_ylabel('')
        if not keep_right_axes[case_label]:
            ax2.set_ylabel('')
            ax3.set_ylabel('')
    
    # Set shared axis limits across all subplots
    for ax1 in axes_1:
        ax1.set_ylim(asr_lims)
    for ax2 in axes_2:
        ax2.set_ylim(eei_lims)
    for ax3 in axes_3:
        ax3.set_ylim(ieei_lims)

    for ax3 in axes_3[:-1]:
        ax3.set_axis_off()

    fig.savefig("figures/figure2_bottomrow.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure2_bottomrow.png")
    plt.close(fig)
    # %%