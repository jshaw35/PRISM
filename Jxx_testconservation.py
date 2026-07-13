# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

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
    ieei_ds = np.cumsum(eei_ds * weights)

    return ieei_ds


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


# %%

test_new_asr = xr.open_dataset("/home/josh2250/projects/PRISM/data/RadInt_procdata_old/iEEI_test/d651058/CESM-CAM5-LME/atm/proc/tseries/monthly/FSNT/b.e11.BLMTRC5CN.f19_g16.002.cam.h0.FSNT.085001-184912.nc")
test_old_asr = xr.open_dataset("/home/josh2250/projects/PRISM/data/RadInt_procdata_old/CESM_LME/d651058/CESM-CAM5-LME/atm/proc/tseries/monthly/FSNT/b.e11.BLMTRC5CN.f19_g16.002.cam.h0.FSNT.085001-184912.nc")

test_new_olr = xr.open_dataset("/home/josh2250/projects/PRISM/data/RadInt_procdata_old/iEEI_test/d651058/CESM-CAM5-LME/atm/proc/tseries/monthly/FLNT/b.e11.BLMTRC5CN.f19_g16.002.cam.h0.FLNT.085001-184912.nc")
test_old_olr = xr.open_dataset("/home/josh2250/projects/PRISM/data/RadInt_procdata_old/CESM_LME/d651058/CESM-CAM5-LME/atm/proc/tseries/monthly/FLNT/b.e11.BLMTRC5CN.f19_g16.002.cam.h0.FLNT.085001-184912.nc")

# %%
# Adjust time
test_new_asr = test_new_asr.assign_coords(
    time=shift_noleap_time_back_one_month(test_new_asr["time"].values)
)
test_old_asr = test_old_asr.assign_coords(
    time=shift_noleap_time_back_one_month(test_old_asr["time"].values)
)
test_new_olr = test_new_olr.assign_coords(
    time=shift_noleap_time_back_one_month(test_new_olr["time"].values)
)
test_old_olr = test_old_olr.assign_coords(
    time=shift_noleap_time_back_one_month(test_old_olr["time"].values)
)

# %%

old_ieei = compute_iEEI(test_old_olr["FLNT"], test_old_asr["FSNT"])
new_ieei = compute_iEEI(test_new_olr["FLNT"], test_new_asr["FSNT"])

# %%

old_ieei.sel(spatial="G")[:100].plot()
new_ieei.sel(spatial="G")[:100].plot()

# %%
# Old way of getting surface area
# earth_radius = 6371e3 # meters
# earth_SA = 4 * np.pi * earth_radius**2

# Value from pop. Total: 504921015096543.6 , Ocean: 360511672426927.94
