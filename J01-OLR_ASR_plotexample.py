# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# %%
def plot_radiative_imbalance(
    olr_da,
    asr_da,
    ax=None,
    plot_kwargs=None,
    fontsize=14,
    cmap=mpl.colormaps['viridis'],
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if plot_kwargs is None:
        plot_kwargs = {}

    # Plot OLR vs ASR with color gradient for time dimension
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
        c=olr_da['time.year']+0.083*olr_da['time.month'], #fractional year coordinate
        cmap=cmap,
        zorder=10,
        **plot_kwargs,
    )

    # Add a colorbar with discrete intervals and extend='both' keyword
    # bounds = pd.date_range(
    #     start=pd.to_datetime(olr_da["time"].min().data),
    #     end=pd.to_datetime(olr_da["time"].max().data), 
    #     periods=min(olr_da.sizes['time'],255)) 
    bounds = np.arange(
        olr_da['time.year'].min(),
        olr_da['time.year'].max(),
        max(1,(olr_da['time.year'].max()-olr_da['time.year'].min())/255))
    norm = mpl.colors.BoundaryNorm(np.array(bounds), cmap.N, extend='both')

    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, orientation='vertical',
        label="Time",
    )

    # Plot the 1:1 line
    min_val = min(olr_da.min().item(), asr_da.min().item())
    max_val = max(olr_da.max().item(), asr_da.max().item())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="grey",
        linestyle="--",
        zorder=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("OLR [Wm$^{-2}$]", fontsize=fontsize)
    ax.set_ylabel("ASR [Wm$^{-2}$]", fontsize=fontsize)

    if fig is None:
        return ax
    else:
        return fig, ax


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


# %%


def main():
    olr = np.arange(0, 10, 0.5)  # Example ASR values
    asr = 0.5 * (olr**2)
    
    fig, ax = plot_radiative_imbalance(
        olr_da=xr.DataArray(olr, dims=["time"], coords={"time": np.arange(len(olr))}),
        asr_da=xr.DataArray(asr, dims=["time"], coords={"time": np.arange(len(asr))}),
        plot_kwargs={"marker": "o", "linestyle": "solid"},
    )
    
    # Get rid of the plot edges and add gridlines through the origin
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axhline(0, color="grey", linestyle="solid", linewidth=1.5)
    ax.axvline(0, color="grey", linestyle="solid", linewidth=1.5)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
# %%

if __name__ == '__main__':
    main()

