# %%
from pathlib import Path
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# import fnmatch
# import dask

import logging
from J15_shared_functions import (
    weighted_annualmean,
    compute_decadal2,
    load_data_with_configs,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%
# Define backbone plotting function
# ASR-OLR scatter plot with color gradient for time dimension
def plot_radiative_imbalance_annual(
    olr_da,
    asr_da,
    ax=None,
    fig=None,
    plot_kwargs=None,
    fontsize=14,
    cmap=mpl.colormaps['viridis'],
    add_colorbar=True,
    cax=None,
    connected=True,
    line11=True,
    norm=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if plot_kwargs is None:
        plot_kwargs = {}

    # Plot OLR vs ASR with color gradient for year dimension
    if "ens" not in olr_da.dims:
        if connected:
            ax.plot(
                olr_da,
                asr_da,
                color="black",
                alpha=0.5,
                zorder=5,
                linewidth=0.5,
            )

        scatter = ax.scatter(
            olr_da,
            asr_da,
            c=olr_da['year'],
            marker='o',
            cmap=cmap,
            norm=norm,
            zorder=10,
            **plot_kwargs,
        )
    else:
        for i in range(len(olr_da['ens'])):
            if connected:
                ax.plot(
                    olr_da.isel(ens=i),
                    asr_da.isel(ens=i),
                    color="black",
                    alpha=0.5,
                    zorder=5,
                    linewidth=0.5,
                )
            scatter = ax.scatter(
                olr_da.isel(ens=i),
                asr_da.isel(ens=i),
                c=olr_da['year'],
                marker='o',
                cmap=cmap,
                norm=norm,
                zorder=10,
                **plot_kwargs,
            )

    # Add a colorbar with discrete intervals and extend='both' keyword
    if norm is None:
        bounds = np.arange(
            olr_da['year'].min(),
            olr_da['year'].max(),
            max(1, (olr_da['year'].max() - olr_da['year'].min()) / 255),
        )
        norm = mpl.colors.BoundaryNorm(np.array(bounds), cmap.N, extend='both')

    if add_colorbar:
        if cax is None:
            cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical',
                label="Year",
            )
        else:
            cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax, orientation='vertical',
                label="Year",
            )

    # Plot the 1:1 line
    if line11:
        min_val = min(olr_da.min(), asr_da.min())
        max_val = max(olr_da.max(), asr_da.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="grey",
            linestyle="--",
            zorder=0,
        )
    ax.set_xlabel("OLR [Wm$^{-2}$]", fontsize=fontsize)
    ax.set_ylabel("ASR [Wm$^{-2}$]", fontsize=fontsize)
    if not line11:
        ax.set_xlim(olr_da.min(), olr_da.max())
        ax.set_ylim(asr_da.min(), asr_da.max())

    if add_colorbar:
        if fig is None:
            return ax, cbar
        else:
            return fig, ax, cbar
    if fig is None:
        return ax
    else:
        return fig, ax


# %%
# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

if __name__ == "__main__":
    data_root = "/glade/work/jonahshaw/PRISM_data/spatial_averages_data/"    
    CASE_CONFIGS = {
        "CESM2-LM": {
            "path": f"{data_root}/CESM2_LME/",
            "subdir_cases": ["b.e21.BWmaHIST.f19_g17.PMIP4-past1000.0??"],
            "append_cases": {
                "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.0??": None,
            },
            "ufunc": None,
        },
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
                #  "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001",
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
    
    asr_var = "FSNT"
    olr_var = "FLNT"
    ts_var = "TS"

    load_var_list = [asr_var, olr_var, ts_var]

    # %%
    # ============================================================================
    # LOAD DATA WITH MULTI-ENSEMBLE SUPPORT
    # ============================================================================
    
    logging.info("Starting multi-ensemble loading")
    
    # Load all cases with multi-ensemble support
    # Load into memory because the spatially averaged data is small enough to fit in memory
    data_dict = {}
    for var in load_var_list:
        data_dict[var] = load_data_with_configs(CASE_CONFIGS, [var], load_into_memory=True)

    # %%
    # Set plotting settings
    lowlim_control = 236
    highlim_control = 244
    lowlim_ssp245 = 235
    highlim_ssp245 = 245
    PLOT_CONFIGS = {
        "CESM2-LM": {
            "subcase": 'b.e21.BWmaHIST.f19_g17.PMIP4-past1000.0??',
            "titles": {
                "b.e21.BWmaHIST.f19_g17.PMIP4-past1000.0??": "CESM2 LME",
            },
            "ylims": (lowlim_control, highlim_control),
            "xlims": (lowlim_control, highlim_control),
            "cbar_ticks": np.arange(850, 1851, 100),
            "cbar_ylabel": None,
        },
        "CESM2_WACCM_1850control" :{
            "subcase": 'b.e21.BW1850.f09_g17.CMIP6-piControl.001',
            "titles": {
                "b.e21.BW1850.f09_g17.CMIP6-piControl.001": "CESM2 WACCM 1850 Control",
            },
            "ylims": (lowlim_control, highlim_control),
            "xlims": (lowlim_control, highlim_control),
            "cbar_ticks": np.arange(0, 501, 50),
            "cbar_ylabel": None,
        },
        "CESM2-WACCM-HIST": {
            "titles": {
                "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??": "CESM2 WACCM HIST",
            },
            "ylims": (lowlim_ssp245, highlim_ssp245),
            "xlims": (lowlim_ssp245, highlim_ssp245),
            "cbar_ticks": np.arange(1850, 2016, 10),
            "cbar_ylabel": None,
        },
        "CESM2_WACCM_SSP2-4.5": {
            "titles": {
                "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??": "CESM2 WACCM SSP2-4.5",
            },
            "ylims": (lowlim_ssp245, highlim_ssp245),
            "xlims": (lowlim_ssp245, highlim_ssp245),
            "cbar_ticks": np.arange(1850, 2101, 10),
            "cbar_ylabel": None,
        },
        "ARISE-SAI": {
            "titles": {
                "1p5K-SAI.0??": "ARISE-SAI 1p5K",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??": "ARISE-SAI-1.5",
                "b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??": "ARISE-SAI-1.5 EXTENDED",
            },
            "ylims": (lowlim_ssp245, highlim_ssp245),
            "xlims": (lowlim_ssp245, highlim_ssp245),
            "cbar_ticks": np.arange(1850, 2086, 10),
            "cbar_ylabel": None,
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "titles": {
                "b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??": "MCB SMBB-050PCT",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000": "MCB CMIP6 baseline",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000": "MCB CMIP6-025PCT",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000": "MCB CMIP6-050PCT",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000": "MCB CMIP6-075PCT",
                "b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000": "MCB CMIP6-125PCT",
            },
            "ylims": (lowlim_ssp245, highlim_ssp245),
            "xlims": (lowlim_ssp245, highlim_ssp245),
            "cbar_ticks": np.arange(1850, 2101, 10),
            "cbar_ylabel": None,
        },
        "ARISE-1.0": {
            "titles": {
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??": "ARISE-SAI-1.0",
                "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??": "ARISE-SAI-1.37-2045",
            },
            "ylims": (lowlim_ssp245, highlim_ssp245),
            "xlims": (lowlim_ssp245, highlim_ssp245),
            "cbar_ticks": np.arange(1850, 2101, 10),
            "cbar_ylabel": None,
        },
    }
    # %%

    fig,axs = plt.subplots(1,2, figsize=(12,4.5))
    fig.subplots_adjust(wspace=0.35)
    caxes = [fig.add_axes([0.46, 0.15, 0.01, 0.7]), fig.add_axes([0.92, 0.15, 0.01, 0.7])] # create separate colorbar axes for each subplot
    outs_list = []
    case_list = ["CESM2_WACCM_1850control", "CESM2-LM"]
    add_colorbar = {
        "CESM2-LM": True,
        "CESM2_WACCM_1850control": True,
    }
    year_steps = {
        "CESM2-LM": 20,
        "CESM2_WACCM_1850control": 20,
    }
    tick_steps = {
        "CESM2-LM": 100,
        "CESM2_WACCM_1850control": 50,
    }
    cmap = sns.color_palette("viridis", as_cmap=True)

    for ax, cax, case_label in zip(axs, caxes, case_list):
        logging.info(f"Plotting case: {case_label}")
        config = PLOT_CONFIGS[case_label]
        olr_ds = data_dict[olr_var][case_label][config["subcase"]][olr_var].compute()
        asr_ds = data_dict[asr_var][case_label][config["subcase"]][asr_var].compute()

        # Compute annual means for ASR and OLR
        asr_annual = weighted_annualmean(asr_ds.sel(spatial="G"))
        olr_annual = weighted_annualmean(olr_ds.sel(spatial="G"))

        # Compute decadal means for ASR and OLR and set time coordinate to the first year in the decade
        asr_decadal = compute_decadal2(asr_ds.sel(spatial="G")).isel(year=slice(5, None, 10))
        olr_decadal = compute_decadal2(olr_ds.sel(spatial="G")).isel(year=slice(5, None, 10))

        year_bounds = [config["cbar_ticks"][0], config["cbar_ticks"][-1]]
        norm = mpl.colors.BoundaryNorm(config["cbar_ticks"], cmap.N, extend='neither')
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
        ax.set_facecolor("whitesmoke")
        ax.set_title(case_label, fontsize=14)
    panel_labels = [f"{chr(97 + i)}." for i in range(len(case_list))]
    # Apply the plot config settings for each subplot
    for ax, cax, case_label, panel_label in zip(axs, caxes, case_list, panel_labels):
        config = PLOT_CONFIGS[case_label]
        ax.set_xlim(config["xlims"])
        ax.set_ylim(config["ylims"])
        cax.set_yticks(config["cbar_ticks"])
        cax.set_ylabel(config["cbar_ylabel"])
        # Add subplot label
        ax.text(236.1, 243.5, panel_label, fontsize=14, fontweight="bold")
        # Add 1-1 lines over the new domain
        min_val = min(config["xlims"][0], config["ylims"][0])
        max_val = min(config["xlims"][1], config["ylims"][1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="grey",
            linestyle="--",
            zorder=0,
        )
    axs[-1].set_ylabel("")
    # %%
    fig.savefig("figures/figure_OLR_ASR_past.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure_OLR_ASR_past.png")
    plt.close(fig)

    # %%
    # Plot the CESM LME, SSP2-4.5, and ARISE-SAI data annually and decadally for the global mean in a 1x3 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    axs = axes.flatten()
    fig.subplots_adjust(wspace=0.18, hspace=0.18)
    # Documentation for naming: https://www.cesm.ucar.edu/community-projects/arise-sai
    # The control experiments are documented here: https://data.ucar.edu/dataset/cesm2-waccm6-ssp245
    # The first 5 to out to 2100 and the later five only go out to 2069.
    subplot_pairs = [
        ["CESM2-WACCM-HIST", 'b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.0??'],
        ["CESM2_WACCM_SSP2-4.5", 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??'],
        ["ARISE-SAI", '1p5K-SAI.0??'],
        ["ARISE-SAI", 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0??'],
        ["ARISE-SAI", 'b.e21.BW.f09_g17.SSP245-TSMLT-ARISE-EXTENDED.0??'],
        ["ARISE-1.0", 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.0??'],
        ["ARISE-1.0", 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.0??'],
        ["CESM2_WACCM_SSP2-4.5_MCB", 'b.e21.BSSP245smbb.f09_g17.MCB-050PCT.0??'],
        # ["CESM2_WACCM_SSP2-4.5_MCB", 'b.e21.BSSP245cmip6.f09_g17.CMIP6-baseline.000', "MCB CMIP6 baseline"],
        ["CESM2_WACCM_SSP2-4.5_MCB", 'b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-025PCT.000'],
        ["CESM2_WACCM_SSP2-4.5_MCB", 'b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-050PCT.000'],
        ["CESM2_WACCM_SSP2-4.5_MCB", 'b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-075PCT.000'],
        ["CESM2_WACCM_SSP2-4.5_MCB", 'b.e21.BSSP245cmip6.f09_g17.CMIP6-MCB-125PCT.000'],
    ]

    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # create separate colorbar axes for each subplot
    outs_list = []
    caxes = [cax for _ in subplot_pairs]
    cmap = sns.color_palette("viridis", as_cmap=True)

    year_step = 5
    tick_step = 25
    year_min = 1850
    year_max = 2100
    # year_min = min([*[PLOT_CONFIGS[i]["cbar_ticks"][0] for i in subplot_pairs]])
    # year_max = max([*[PLOT_CONFIGS[i]["cbar_ticks"][-1] for i in subplot_pairs]])
    year_bounds = np.arange(year_min, year_max + 1, year_step)
    cbar_ticks = np.arange(year_min, year_max + 1, tick_step)

    min_val = min(PLOT_CONFIGS[case_label]["xlims"][0], PLOT_CONFIGS[case_label]["ylims"][0])
    max_val = min(PLOT_CONFIGS[case_label]["xlims"][1], PLOT_CONFIGS[case_label]["ylims"][1])

    norm = mpl.colors.BoundaryNorm(year_bounds, cmap.N, extend="neither")

    for ax, cax, (case_label, experiment_label) in zip(axs, caxes, subplot_pairs):
        logging.info(f"Plotting case: {case_label}")
        # title = f"{case_label}: {experiment_label.split('.')[-2]}"# (N={N:d})"
        config = PLOT_CONFIGS[case_label]
        title = config["titles"][experiment_label]
        olr_ds = data_dict[olr_var][case_label][experiment_label][olr_var].sel(spatial="G").compute()
        asr_ds = data_dict[asr_var][case_label][experiment_label][asr_var].sel(spatial="G").compute()

        # Compute decadal means for ASR and OLR and set time coordinate to the first year in the decade
        asr_decadal = compute_decadal2(asr_ds).isel(year=slice(5, None, 10))
        olr_decadal = compute_decadal2(olr_ds).isel(year=slice(5, None, 10))
        
        asr_annual = weighted_annualmean(asr_ds)
        olr_annual = weighted_annualmean(olr_ds)

        # Take the mean across ensemble members
        N = 1
        if "ens" in olr_ds.dims:
            N = len(olr_ds["ens"])
            asr_decadal = asr_decadal.mean(dim="ens")
            olr_decadal = olr_decadal.mean(dim="ens")

        add_colorbar = False

        outs1 = plot_radiative_imbalance_annual(
            olr_annual,
            asr_annual,
            ax=ax,
            cax=cax,
            add_colorbar=add_colorbar,
            plot_kwargs={"s": 5, "alpha": 0.5},
            connected=False,
            line11=False,
            norm=norm,
        )
        if ax == axs[0]:
            add_colorbar = True
        outs2 = plot_radiative_imbalance_annual(
            olr_decadal,
            asr_decadal,
            ax=ax,
            cax=cax,
            add_colorbar=add_colorbar,
            plot_kwargs={"s": 50, "facecolors": "none", "edgecolors": "black"},
            connected=False,
            line11=False,
            norm=norm,
        )
        outs_list.append((outs1, outs2))
        ax.set_title(title, fontsize=12)
        ax.text(242.8, 244.4, f"N={N}", fontsize=12, fontweight="bold")
    # Apply the plot config settings for each subplot
    panel_labels = [f"{chr(97 + i)}." for i in range(len(subplot_pairs))]
    for ax, cax, panel_label, (case_label, experiment_label) in zip(axs, caxes, panel_labels, subplot_pairs):
        config = PLOT_CONFIGS[case_label]
        ax.set_xlim(config["xlims"])
        ax.set_ylim(config["ylims"])
        # cax.set_yticks(config["cbar_ticks"])
        cax.set_ylabel(config["cbar_ylabel"])
        ax.text(235.2, 244.2, panel_label, fontsize=14, fontweight="bold")
        # Add 1-1 lines over the new domain
        min_val = min(config["xlims"][0], config["ylims"][0])
        max_val = min(config["xlims"][1], config["ylims"][1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="grey",
            linestyle="--",
            zorder=0,
        )
        # Set the face color of the subplot
        ax.set_facecolor("whitesmoke")
    cbar = outs_list[0][-1][-1]
    cbar.set_ticks(cbar_ticks)

    # Hide the x axis labels outside of the bottom row and the y axis labels outside of the leftmost column
    for ax in axes[:-1, :].flat:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    # Add text to the whole figure to serve as x-axis for all subplots
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.text(0.51, 0.07, "OLR (Wm$^{-2}$)", ha="center", va="center", fontsize=15)
    fig.text(0.09, 0.5, "ASR (Wm$^{-2}$)", ha="center", va="center", fontsize=15, rotation=90)

    # %%
    fig.savefig("figures/figure_OLR_ASR_future.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure_OLR_ASR_future.png")
    plt.close(fig)
    # %%