# %%
from pathlib import Path
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import logging
from OLR_ASR_plotexample import plot_radiative_imbalance, plot_radiative_imbalance_annual

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
        ds_decadal = ds_annual.rolling(year=10, min_periods=10, center=True).mean(dim="year")
    elif "year" in ds.coords:
        ds_decadal = ds.rolling(year=10, min_periods=10, center=True).mean(dim="year")
    return ds_decadal


def plot_eei(
    eei_ds,
    ax=None,
    plot_kwargs=None,
    fontsize=14,
    # cmap=sns.color_palette("viridis", as_cmap=True),
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.plot(
        eei_ds["time"],
        eei_ds,
        # **plot_kwargs,
    )
    ax.set_xlabel("Time", fontsize=fontsize)
    ax.set_ylabel("EEI (W)", fontsize=fontsize)


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
        "CESM2_WACCM_1850control" :{
            "path": "data/RadInt_procdata/CESM2_WACCM_1850control/",
            "case_str": "b.e21.BW1850.f09_g17.CMIP6-piControl.001",
            "append_case": None,
            "ufunc": None,
        },
        "CESM2-WACCM-HIST": {
            "path": "data/RadInt_procdata/CESM2_WACCM_HIST/",
            "case_str": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001",
            "append_case": None,
            "ufunc": None,
        },
        "CESM2-SSP2-4.5": {
            "path": "data/RadInt_procdata/CESM2_WACCM_SSP2-4.5/",
            "case_str": "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001",
            "append_case": "CESM2-WACCM-HIST",
            "ufunc": None,
        },
        "ARISE-SAI": {
            "path": "data/RadInt_procdata/ARISE_SAI/",
            "case_str": "1p5K-SAI.001",
            "append_case": "CESM2-SSP2-4.5",
            "ufunc": None,
        },
        "ARISE-SAI_extended": {
            "path": "data/RadInt_procdata/ARISE_SAI/",
            "case_str": "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.001",
            "append_case": "ARISE-SAI",
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
            "ylims": (230, 237),
            "xlims": (230, 237),
            "cbar_ticks": np.arange(850, 1851, 100),
            "cbar_ylabel": None,
        },
        "CESM2-LME": {
            "ylims": (236, 244),
            "xlims": (236, 244),
            "cbar_ticks": np.arange(850, 1851, 100),
            "cbar_ylabel": None,
        },
        "CESM2_WACCM_1850control" :{
            "ylims": (236, 244),
            "xlims": (236, 244),
            "cbar_ticks": np.arange(0, 501, 50),
            "cbar_ylabel": None,
        },
        "CESM2-WACCM-HIST": {
            "ylims": (230, 237),
            "xlims": (230, 237),
            "cbar_ticks": np.arange(1850, 2016, 10),
            "cbar_ylabel": None,
        },
        "CESM2-SSP2-4.5": {
            "ylims": (236, 245),
            "xlims": (236, 245),
            "cbar_ticks": np.arange(1850, 2101, 10),
            "cbar_ylabel": None,
        },
        "ARISE-SAI": {
            "ylims": (236, 245),
            "xlims": (236, 245),
            "cbar_ticks": np.arange(1850, 2086, 10),
            "cbar_ylabel": None,
        },
        "ARISE-SAI_extended": {
            "ylims": (236, 245),
            "xlims": (236, 245),
            "cbar_ticks": np.arange(1850, 2101, 10),
            "cbar_ylabel": None,
        },
        "CESM2-SSP2-4.5_MCB": {
            "ylims": (236, 245),
            "xlims": (236, 245),
            "cbar_ticks": np.arange(1850, 2101, 10),
            "cbar_ylabel": None,
        },
    }

    # %%
    # Plot the CESM LME and CESM2 LME data annually and decadally for the global mean in a 1x3 subplot grid
    fig,axs = plt.subplots(1,2, figsize=(12,4.5))
    fig.subplots_adjust(wspace=0.35)
    caxes = [fig.add_axes([0.46, 0.15, 0.01, 0.7]), fig.add_axes([0.92, 0.15, 0.01, 0.7])] # create separate colorbar axes for each subplot
    # cax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # create separate colorbar axes for each subplot
    # caxes = [cax, cax]
    outs_list = []
    # case_list = ["CESM-LME", "CESM2-LME"]
    case_list = ["CESM2_WACCM_1850control", "CESM2-LME"]
    add_colorbar = {
        "CESM-LME": True,
        "CESM2-LME": True,
        "CESM2_WACCM_1850control": True,
    }
    year_steps = {
        "CESM-LME": 20,
        "CESM2-LME": 20,
        "CESM2_WACCM_1850control": 20,
    }
    tick_steps = {
        "CESM-LME": 100,
        "CESM2-LME": 100,
        "CESM2_WACCM_1850control": 50,
    }
    cmap = sns.color_palette("viridis", as_cmap=True)

    for ax, cax, case_label in zip(axs, caxes, case_list):
        logging.info(f"Plotting case: {case_label}")
        olr_ds = data_dict[case_label][olr_var].compute()
        asr_ds = data_dict[case_label][asr_var].compute()

        # Compute annual means for ASR and OLR
        asr_annual = weighted_annualmean(asr_ds.sel(spatial="G"))
        olr_annual = weighted_annualmean(olr_ds.sel(spatial="G"))

        # Compute decadal means for ASR and OLR and set time coordinate to the first year in the decade
        asr_decadal = compute_decadal2(asr_ds.sel(spatial="G"))[5::10]
        olr_decadal = compute_decadal2(olr_ds.sel(spatial="G"))[5::10]

        year_bounds = [PLOT_CONFIGS[case_label]["cbar_ticks"][0], PLOT_CONFIGS[case_label]["cbar_ticks"][-1]]
        norm = mpl.colors.BoundaryNorm(PLOT_CONFIGS[case_label]["cbar_ticks"], cmap.N, extend='neither')
        outs1 = plot_radiative_imbalance_annual(
            olr_annual,
            asr_annual,
            ax=ax,
            add_colorbar=add_colorbar[case_label],
            cax=cax,
            plot_kwargs={"s": 5, "alpha": 0.5},
            connected=False,
            line11=False,
            norm=norm,
        )
        outs2 = plot_radiative_imbalance_annual(
            olr_decadal,
            asr_decadal,
            ax=ax,
            add_colorbar=add_colorbar[case_label],
            cax=cax,
            plot_kwargs={"s": 50, "facecolors": "none", "edgecolors": "black"},
            connected=False,
            line11=False,
            norm=norm,
        )
        outs_list.append((outs1, outs2))
        ax.set_title(case_label)
    
    # Apply the plot config settings for each subplot
    for ax, cax, case_label in zip(axs, caxes, case_list):
        ax.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
        ax.set_ylim(PLOT_CONFIGS[case_label]["ylims"])
        cax.set_yticks(PLOT_CONFIGS[case_label]["cbar_ticks"])
        cax.set_ylabel(PLOT_CONFIGS[case_label]["cbar_ylabel"])
        
        # Add 1-1 lines over the new domain
        min_val = min(PLOT_CONFIGS[case_label]["xlims"][0], PLOT_CONFIGS[case_label]["ylims"][0])
        max_val = min(PLOT_CONFIGS[case_label]["xlims"][1], PLOT_CONFIGS[case_label]["ylims"][1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="grey",
            linestyle="--",
            zorder=0,
        )

    fig.savefig("figures/figure1b_toprow.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure1b_toprow.png")
    plt.close(fig)

    # %%
    # Plot the CESM LME, SSP2-4.5, and ARISE-SAI data annually and decadally for the global mean in a 1x3 subplot grid
    fig,axs = plt.subplots(1,3, figsize=(16,4.5))
    fig.subplots_adjust(wspace=0.35)
    # caxes = [fig.add_axes([0.33, 0.15, 0.01, 0.7]), fig.add_axes([0.62, 0.15, 0.01, 0.7]), fig.add_axes([0.905, 0.15, 0.01, 0.7])] # create separate colorbar axes for each subplot
    # case_list = ["CESM2-SSP2-4.5", "ARISE-SAI", "CESM2-SSP2-4.5_MCB"]
    case_list = ["CESM2-SSP2-4.5", "ARISE-SAI_extended", "CESM2-SSP2-4.5_MCB"]

    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # create separate colorbar axes for each subplot
    outs_list = []
    caxes = [cax, cax, cax]
    add_colorbar = {
        "CESM2-SSP2-4.5": True,
        "ARISE-SAI_extended": False,
        # "ARISE-SAI": False,
        "CESM2-SSP2-4.5_MCB": False,
    }
    cmap = sns.color_palette("viridis", as_cmap=True)

    year_step = 5
    tick_step = 25
    year_min = min([*[PLOT_CONFIGS[i]["cbar_ticks"][0] for i in case_list]])
    year_max = max([*[PLOT_CONFIGS[i]["cbar_ticks"][-1] for i in case_list]])
    year_bounds = np.arange(year_min, year_max + 1, year_step)
    cbar_ticks = np.arange(year_min, year_max + 1, tick_step)

    min_val = min(PLOT_CONFIGS[case_label]["xlims"][0], PLOT_CONFIGS[case_label]["ylims"][0])
    max_val = min(PLOT_CONFIGS[case_label]["xlims"][1], PLOT_CONFIGS[case_label]["ylims"][1])

    norm = mpl.colors.BoundaryNorm(year_bounds, cmap.N, extend="neither")

    for ax, cax, case_label in zip(axs, caxes, case_list):
        logging.info(f"Plotting case: {case_label}")
        olr_ds = data_dict[case_label][olr_var].compute()
        asr_ds = data_dict[case_label][asr_var].compute()

        # Compute annual means for ASR and OLR
        asr_annual = weighted_annualmean(asr_ds.sel(spatial="G"))
        olr_annual = weighted_annualmean(olr_ds.sel(spatial="G"))

        # Compute decadal means for ASR and OLR and set time coordinate to the first year in the decade
        asr_decadal = compute_decadal2(asr_ds.sel(spatial="G"))[5::10]
        olr_decadal = compute_decadal2(olr_ds.sel(spatial="G"))[5::10]

        outs1 = plot_radiative_imbalance_annual(
            olr_annual,
            asr_annual,
            ax=ax,
            cax=cax,
            add_colorbar=add_colorbar[case_label],
            plot_kwargs={"s": 5, "alpha": 0.5},
            connected=False,
            line11=False,
            norm=norm,
        )
        outs2 = plot_radiative_imbalance_annual(
            olr_decadal,
            asr_decadal,
            ax=ax,
            cax=cax,
            add_colorbar=add_colorbar[case_label],
            plot_kwargs={"s": 50, "facecolors": "none", "edgecolors": "black"},
            connected=False,
            line11=False,
            norm=norm,
        )
        outs_list.append((outs1, outs2))
        ax.set_title(case_label)
    
    # Apply the plot config settings for each subplot
    for ax, cax, case_label in zip(axs, caxes, case_list):
        ax.set_xlim(PLOT_CONFIGS[case_label]["xlims"])
        ax.set_ylim(PLOT_CONFIGS[case_label]["ylims"])
        cax.set_yticks(PLOT_CONFIGS[case_label]["cbar_ticks"])
        cax.set_ylabel(PLOT_CONFIGS[case_label]["cbar_ylabel"])
        
        # Add 1-1 lines over the new domain
        min_val = min(PLOT_CONFIGS[case_label]["xlims"][0], PLOT_CONFIGS[case_label]["ylims"][0])
        max_val = min(PLOT_CONFIGS[case_label]["xlims"][1], PLOT_CONFIGS[case_label]["ylims"][1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="grey",
            linestyle="--",
            zorder=0,
        )
    cbar = outs_list[0][-1][-1]
    cbar.set_ticks(cbar_ticks)

    # %%
    fig.savefig("figures/figure1_bottomrow.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure1_bottomrow.png")
    plt.close(fig)

    # %%