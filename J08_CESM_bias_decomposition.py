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
import seaborn as sns
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


def plot_error_tseries_multiensemble(
    data_da,
    error_component: str = "NMSE",
    ax=None,
    fontsize=14,
    time_dim="year",
    color=None,
    label=None,
    plot_annual_members=True,
):
    """
    Plot time series of error components with individual ensemble member lines
    (thin, semi-transparent) and decadal ensemble mean lines (thick, solid).

    Parameters
    ----------
    data_da : xr.DataArray
        DataArray with dims (year, [ens]) and coordinate error_component.
    error_component : str
        Error component to plot (e.g. "NMSE", "U").
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    fontsize : int, default 14
        Font size for labels and titles.
    time_dim : str, default "year"
        Name of the time dimension.
    colors : Color to plot

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    component_data = data_da.sel(error_component=error_component)

    if plot_annual_members:
        if "ens" in component_data.dims:
            for ens_member in component_data["ens"].values:
                annual_member = component_data.sel(ens=ens_member)
                ax.plot(
                    annual_member[time_dim],
                    annual_member,
                    color=color,
                    linestyle="-",
                    linewidth=0.7,
                    alpha=0.5,
                )
            decadal = compute_decadal2(component_data).mean(dim="ens")
        else:
            annual = component_data
            ax.plot(
                annual[time_dim],
                annual,
                color=color,
                linestyle="-",
                linewidth=0.7,
                alpha=0.5,
            )
            decadal = compute_decadal2(component_data)
    else:
        if "ens" in component_data.dims:
            decadal = compute_decadal2(component_data).mean(dim="ens")
        else:
            decadal = compute_decadal2(component_data)

    ax.plot(
        decadal[time_dim],
        decadal,
        color=color,
        linestyle="-",
        linewidth=2.5,
        label=f"{label if label is not None else ''}",
    )

    return fig, ax


def plot_control_uncertainty_tseries(
    control_da,
    ax,
    error_component,
    xlims,
    time_dim="year",
    unc_gauss=False,
    label="Control uncertainty",
    annual_uncertainty=True,
    decadal_uncertainty=True,
):
    """
    Plot control-period uncertainty as horizontal shaded bands for each error component.

    Parameters
    ----------
    control_da : xr.DataArray
        DataArray from the control simulation with error_component coordinate.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    error_components : list of str
        Error components to shade uncertainty for.
    xlims : tuple of int
        (min_year, max_year) defining the x-axis extent of the shading.
    time_dim : str, default "year"
        Time dimension over which to compute standard deviation or quantiles.
    unc_gauss : bool, default True
        If True, use mean ± 2*stddev. If False, use 5th/95th quantiles.
    """

    control_component_data = control_da.sel(error_component=error_component)

    std_dims = [time_dim]
    if "ens" in control_component_data.dims and "ens" not in std_dims:
        std_dims.append("ens")

    # Compute uncertainty bounds based on the specified method
    if unc_gauss:
        if annual_uncertainty:
            control_mean_annual = control_component_data.mean(dim=std_dims)
            control_stddev_annual = control_component_data.std(dim=std_dims)
            low_bound_annual = control_mean_annual - 2 * control_stddev_annual
            high_bound_annual = control_mean_annual + 2 * control_stddev_annual
        if decadal_uncertainty:
            control_decadal = compute_decadal2(control_component_data)
            # Select decadal means to avoid overlapping windows that inflate sample size
            control_decadal = control_decadal.isel(time=slice(0, None, 10))
            control_mean_decadal = control_decadal.mean(dim=std_dims)
            control_stddev_decadal = control_decadal.std(dim=std_dims)
            low_bound_decadal = control_mean_decadal - 2 * control_stddev_decadal
            high_bound_decadal = control_mean_decadal + 2 * control_stddev_decadal
    else:
        if annual_uncertainty:
            low_bound_annual = control_component_data.quantile(0.05, dim=std_dims)
            high_bound_annual = control_component_data.quantile(0.95, dim=std_dims)
        if decadal_uncertainty:
            control_decadal = compute_decadal2(control_component_data)
            low_bound_decadal = control_decadal.quantile(0.05, dim=std_dims)
            high_bound_decadal = control_decadal.quantile(0.95, dim=std_dims)

    # Plot the uncertainty bands
    if annual_uncertainty:
        ax.fill_between(
            xlims,
            low_bound_annual,
            high_bound_annual,
            color="grey",
            linestyle="-",
            alpha=0.3,
            label=f"{label} (annual)",
        )
    if decadal_uncertainty:
        ax.fill_between(
            xlims,
            low_bound_decadal,
            high_bound_decadal,
            color="grey",
            linestyle="-",
            alpha=0.5,
            label=f"{label} (decadal)",
        )


def plot_error_scatter_multiexperiment(
    data_dict,
    error_component,
    case_list,
    ax=None,
    fontsize=14,
    add_colorbar=False,
    cax=None,
    norm=None,
):
    """
    Scatter plot of error component vs year, colored by experiment, showing
    individual ensemble members (small points) and decadal ensemble mean (open circles).

    Parameters
    ----------
    data_dict : dict
        Nested dictionary: data_dict[case_label][subcase] = xr.DataArray
        (or xr.Dataset containing a variable with an error_component coordinate),
        with dims (year, [ens]).
    error_component : str
        Single error component to plot (e.g. "NMSE").
    case_list : list of tuple
        Each tuple is (case_label, subdir, display_name, color).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    fontsize : int, default 14
        Font size for labels and titles.
    add_colorbar : bool, default False
        Whether to add a colorbar (unused in this version, kept for flexibility).
    cax : matplotlib.axes.Axes, optional
        Colorbar axis.
    norm : matplotlib.colors.Normalize, optional
        Normalization for colormap.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    for case_label, subdir, display_name, color in case_list:
        leaf = data_dict[case_label][subdir]
        if isinstance(leaf, xr.Dataset):
            matching = [v for v in leaf.data_vars if "error_component" in leaf[v].coords or "error_component" in leaf[v].dims]
            if len(matching) == 0:
                raise ValueError(f"No variable with error_component found in Dataset for {case_label} / {subdir}")
            data_da = leaf[matching[0]]
        else:
            data_da = leaf
        component_data = data_da.sel(error_component=error_component)

        if "ens" in component_data.dims:
            for ens_member in component_data["ens"].values:
                annual_member = component_data.sel(ens=ens_member)
                ax.scatter(
                    annual_member["year"],
                    annual_member,
                    color=color,
                    marker="o",
                    s=5,
                    alpha=0.5,
                    zorder=5,
                )
            decadal = compute_decadal2(component_data).isel(year=slice(5, None, 10)).mean(dim="ens")
        else:
            annual = component_data
            ax.scatter(
                annual["year"],
                annual,
                color=color,
                marker="o",
                s=5,
                alpha=0.5,
                zorder=5,
            )
            decadal = compute_decadal2(component_data).isel(year=slice(5, None, 10))

        ax.scatter(
            decadal["year"],
            decadal,
            s=50,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            zorder=10,
            label=display_name,
        )

    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_ylabel(f"{error_component} Error", fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_control_uncertainty_scatter(
    control_da,
    ax,
    error_component,
    xlims,
    time_dim="year",
    unc_gauss=True,
):
    """
    Plot control-period uncertainty as a horizontal shaded band spanning the x-axis.

    Parameters
    ----------
    control_da : xr.DataArray
        DataArray from the control simulation with error_component coordinate.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    error_component : str
        Single error component to shade uncertainty for.
    xlims : tuple of int
        (min_year, max_year) defining the x-axis extent of the shading.
    time_dim : str, default "year"
        Name of the time dimension.
    unc_gauss : bool, default True
        If True, use mean ± 2*stddev. If False, use 5th/95th quantiles.
    """
    x_range = np.arange(xlims[0], xlims[1] + 1, 1)

    control_component_data = control_da.sel(error_component=error_component)

    if unc_gauss:
        control_mean = control_component_data.mean(dim=time_dim)
        control_stddev = control_component_data.std(dim=time_dim)
        low_bound = control_mean - 2 * control_stddev
        high_bound = control_mean + 2 * control_stddev
    else:
        low_bound = control_component_data.quantile(0.05, dim=time_dim)
        high_bound = control_component_data.quantile(0.95, dim=time_dim)

    ax.fill_between(
        x_range,
        low_bound,
        high_bound,
        color="black",
        linestyle="-",
        alpha=0.3,
        label="Control uncertainty",
    )


# %%

if __name__ == "__main__":
    control_case = "CESM2_WACCM_HIST_2000_2014"
    data_root = f"/glade/work/jonahshaw/PRISM_data/error_relativetobaseline_atm/{control_case}/"
    CASE_CONFIGS_ATM = deepcopy(CASE_CONFIGS_TEMPLATE)

    # Remove cases that are not used here
    CASE_CONFIGS_ATM.pop("CESM2-LM", None)
    CASE_CONFIGS_ATM.pop("CESM2_WACCM_1850control", None)

    for config in CASE_CONFIGS_ATM.values():
        config["path"] = str(Path(data_root) / config.pop("path"))

    # %%
    # Load the data using the generalized loading function
    data_varlist = ["FLNT", "FSNT", "FNNT", "FLNS", "FSNS", "TS", "PRECIP_THERMO", "LHFLX", "SHFLX", "PRECT"]
    # data_varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO", "FNNT"]
    year_dim = "year"
    ohc_varlist = ["OHC", "OHC_global_mean"]

    # Load data lazily using the generalized function
    data_dict = {}
    for var in data_varlist:
        data_dict[var] = load_data_with_configs(CASE_CONFIGS_ATM, [var], year_dim=year_dim, load_into_memory=False)

    # %%
    # fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    # fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axs = axes.flatten()
    plot_vars = ["FNNT", "FLNT", "FSNT", "FLNS", "FSNS", "TS", "PRECIP_THERMO", "LHFLX", "SHFLX", "PRECT"]
    # plot_vars = ["FLNT", "FSNT", "FNNT", "FLNS", "FSNS", "TS", "PRECIP_THERMO", "LHFLX", "PRECT"]
    # plot_vars = ["FLNT", "FSNT", "FNNT", "TS", "PRECIP_THERMO", "LHFLX"]
    time_dim = "year"
    # control_case = "CESM2_WACCM_1850control_0100_0499"
    nullhypothesis_case = "CESM2_WACCM_SSP2-4.5"
    nullhypothesis_experiment = "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0??"
    nullhypothesis_tsel = slice(2015, 2034)
    test_cases = ['CESM2_WACCM_SSP2-4.5', 'ARISE-SAI', 'ARISE-1.0', 'CESM2_WACCM_SSP2-4.5_MCB'] # 'CESM2-WACCM-HIST'
    colors = sns.color_palette("colorblind", n_colors=13)
    fontsize = 12

    for ax, var in zip(axs, plot_vars):
        var_dict = data_dict[var]
        i = 0
        for case in test_cases:
            logging.info(f"{case}")
            case_dict = var_dict[case]
            for subdir in case_dict:
                exp_ds = case_dict[subdir].compute()
                N = exp_ds.sizes["ens"] if "ens" in exp_ds.dims else 1
                if N == 1: continue  # Skip cases with only one ensemble member
                title = f"{title_dict[case][subdir]} (N={N})"
                plot_error_tseries_multiensemble(
                    exp_ds[var],
                    ax=ax,
                    plot_annual_members=False,
                    label=title,
                    color=colors[i],
                )
                i += 1

        # Plot the null hypothesis as a shaded band
        control_da = var_dict[nullhypothesis_case][nullhypothesis_experiment][var].sel({time_dim: nullhypothesis_tsel}).compute()
        xlims = ax.get_xlim()
        plot_control_uncertainty_tseries(
            control_da=control_da,
            ax=ax,
            error_component="NMSE",
            xlims=xlims
        )
        ax.set_title(f"{var}", fontsize=fontsize+5)
        ax.set_xlabel("Year", fontsize=fontsize)
        ax.set_ylabel("NMSE", fontsize=fontsize)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("whitesmoke")
    axs[2].legend()

    # %%
    fig.savefig("figures/figure_NMSEdraft.png", dpi=300, bbox_inches='tight')
    logging.info("Saved figure_NMSEdraft.png")
    plt.close(fig)

# %%