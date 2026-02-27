# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# %%
def plot_radiative_imbalance(
    olr_ds,
    asr_ds,
    ax=None,
    plot_kwargs=None,
    fontsize=14,
    cmap=sns.color_palette("viridis", as_cmap=True),
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if plot_kwargs is None:
        plot_kwargs = {}

    # Plot OLR vs ASR with color gradient for time dimension
    ax.plot(
        olr_ds,
        asr_ds,
        color="black",
        alpha=0.5,
        zorder=5,
        linewidth=0.5,
    )
    scatter = ax.scatter(
        olr_ds,
        asr_ds,
        c=np.arange(len(olr_ds)),
        cmap=cmap,
        zorder=10,
        **plot_kwargs,
    )

    # Add a colorbar with discrete intervals and extend='both' keyword
    bounds = np.arange(olr_ds["time"].min().item(), olr_ds["time"].max().item() + 1, max(1, len(olr_ds)//10))   
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, orientation='vertical',
        label="Time",
    )

    # Plot the 1:1 line
    min_val = min(olr_ds.min().item(), asr_ds.min().item())
    max_val = max(olr_ds.max().item(), asr_ds.max().item())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="grey",
        linestyle="--",
        zorder=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("OLR (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_ylabel("ASR (Wm$^{-2}$)", fontsize=fontsize)

    if fig is None:
        return ax
    else:
        return fig, ax

# %%

olr = np.arange(0, 10, 0.5)  # Example ASR values
asr = 0.5 * (olr**2)

fig, ax = plot_radiative_imbalance(
    olr_ds=xr.DataArray(olr, dims=["time"], coords={"time": np.arange(len(olr))}),
    asr_ds=xr.DataArray(asr, dims=["time"], coords={"time": np.arange(len(asr))}),
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

