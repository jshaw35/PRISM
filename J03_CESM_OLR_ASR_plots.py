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
    curc_lme_outpath = "data/RadInt_procdata/CESM_LME/"
    curc_cesm2_245_outpath = "data/RadInt_procdata/CESM2_WACCM_SSP2-4.5/"
    curc_ariseSAI_outpath = "data/RadInt_procdata/ARISE_SAI/"

    lme_case_str = "BLMTRC5CN.f19_g16.003"
    cesm2_cesm2_245_case_str = "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001"
    ariseSAI_case_str = "1p5K-SAI.001"

    case_labels = ["CESM-LME", "CESM2-SSP2-4.5", "ARISE-SAI"]

    asr_var = "FSNT" #"FSNTOA"
    olr_var = "FLNT" #"FLNT"
    ts_var = "TS"

    data_dict = {}
    for datapath, case_str, case_label in [(curc_lme_outpath, lme_case_str, case_labels[0]), (curc_cesm2_245_outpath, cesm2_cesm2_245_case_str, case_labels[1]), (curc_ariseSAI_outpath, ariseSAI_case_str, case_labels[2])]:

        asr_files = crawl_and_list_glob(datapath, f"**/*{case_str}*.{asr_var}.*nc")
        olr_files = crawl_and_list_glob(datapath, f"**/*{case_str}*.{olr_var}.*nc")
        ts_files = crawl_and_list_glob(datapath, f"**/*{case_str}*.{ts_var}.*nc")

        asr_ds = xr.open_mfdataset(asr_files)
        olr_ds = xr.open_mfdataset(olr_files)
        ts_ds = xr.open_mfdataset(ts_files)

        data_dict[case_label] = {
            "asr_ds": asr_ds,
            "olr_ds": olr_ds,
            "ts_ds": ts_ds,
        }

    for case_label in case_labels:
        logging.info(f"Plotting case: {case_label}")
        olr_ds = data_dict[case_label]["olr_ds"]
        asr_ds = data_dict[case_label]["asr_ds"]
        ts_ds = data_dict[case_label]["ts_ds"]

        # Handle the CESM time coordinate issue and challenges with cftime.DatetimeNoLeap
        asr_ds = asr_ds.assign_coords(
            time=shift_noleap_time_back_one_month(asr_ds["time"].values)
        )
        olr_ds = olr_ds.assign_coords(
            time=shift_noleap_time_back_one_month(olr_ds["time"].values)
        )
        ts_ds = ts_ds.assign_coords(
            time=shift_noleap_time_back_one_month(ts_ds["time"].values)
        )

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        plot_radiative_imbalance(
            olr_ds[olr_var].sel(spatial="G").compute(),
            asr_ds[asr_var].sel(spatial="G").compute(),
            ax=axs[0],
        )
        plot_radiative_imbalance(
            olr_ds[olr_var].sel(spatial="SH").compute(),
            asr_ds[asr_var].sel(spatial="SH").compute(),
            ax=axs[1],
        )
        plot_radiative_imbalance(
            olr_ds[olr_var].sel(spatial="NH").compute(),
            asr_ds[asr_var].sel(spatial="NH").compute(),
            ax=axs[2],
        )
        break
        fig1.savefig("figures/testfig.png")
        #
        break
    # %%
fontsize = 14
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="G").compute(),
    asr_ds[asr_var].sel(spatial="G").compute(),
    ax=axs[0],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="SH").compute(),
    asr_ds[asr_var].sel(spatial="SH").compute(),
    ax=axs[1],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="NH").compute(),
    asr_ds[asr_var].sel(spatial="NH").compute(),
    ax=axs[2],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
axs[0].set_title("Global", fontsize=fontsize)
axs[1].set_title("SH", fontsize=fontsize)
axs[2].set_title("NH", fontsize=fontsize)
axs[1].set_ylabel("")
axs[2].set_ylabel("")
# %%
fontsize = 14
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
plot_radiative_imbalance_annual(
    olr_ds[olr_var].sel(spatial="G").groupby("time.year").mean(),
    asr_ds[asr_var].sel(spatial="G").groupby("time.year").mean(),
    ax=axs[0],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance_annual(
    olr_ds[olr_var].sel(spatial="SH").groupby("time.year").mean(),
    asr_ds[asr_var].sel(spatial="SH").groupby("time.year").mean(),
    ax=axs[1],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance_annual(
    olr_ds[olr_var].sel(spatial="NH").groupby("time.year").mean(),
    asr_ds[asr_var].sel(spatial="NH").groupby("time.year").mean(),
    ax=axs[2],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
axs[0].set_title("Global", fontsize=fontsize)
axs[1].set_title("SH", fontsize=fontsize)
axs[2].set_title("NH", fontsize=fontsize)
axs[1].set_ylabel("")
axs[2].set_ylabel("")
# %%
datapath = curc_cesm2_245_outpath

asr_files = crawl_and_list_glob(datapath, f"**/*.{asr_var}.*")
olr_files = crawl_and_list_glob(datapath, f"**/*.{olr_var}.*")
#
asr_ds = xr.open_mfdataset(asr_files)
olr_ds = xr.open_mfdataset(olr_files)

fontsize = 14
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="G").compute(),
    asr_ds[asr_var].sel(spatial="G").compute(),
    ax=axs[0],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="SH").compute(),
    asr_ds[asr_var].sel(spatial="SH").compute(),
    ax=axs[1],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="NH").compute(),
    asr_ds[asr_var].sel(spatial="NH").compute(),
    ax=axs[2],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
axs[0].set_title("Global", fontsize=fontsize)
axs[1].set_title("SH", fontsize=fontsize)
axs[2].set_title("NH", fontsize=fontsize)
axs[1].set_ylabel("")
axs[2].set_ylabel("")
# %%
datapath = curc_ariseSAI_outpath

asr_files = crawl_and_list(datapath, f"1p5K-SAI.001.cam.h0.{asr_var}.")
olr_files = crawl_and_list(datapath, f"1p5K-SAI.001.cam.h0.{olr_var}.")
#
asr_ds = xr.open_mfdataset(asr_files)
olr_ds = xr.open_mfdataset(olr_files)

fontsize = 14
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="G").compute(),
    asr_ds[asr_var].sel(spatial="G").compute(),
    ax=axs[0],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="SH").compute(),
    asr_ds[asr_var].sel(spatial="SH").compute(),
    ax=axs[1],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
plot_radiative_imbalance(
    olr_ds[olr_var].sel(spatial="NH").compute(),
    asr_ds[asr_var].sel(spatial="NH").compute(),
    ax=axs[2],
    cax=cax,
    plot_kwargs={"s": 5},
    connected=False,
)
axs[0].set_title("Global", fontsize=fontsize)
axs[1].set_title("SH", fontsize=fontsize)
axs[2].set_title("NH", fontsize=fontsize)
axs[1].set_ylabel("")
axs[2].set_ylabel("")
# %%
olr_ds["FLNT"].sel(spatial="G").groupby("time.year").mean().plot()
asr_ds["FSNTOA"].sel(spatial="G").groupby("time.year").mean().plot()
