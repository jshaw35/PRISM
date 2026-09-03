"""
Using the decomposition of Medeiros (2023) and Simpson et al. (2020).
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA002918
https://doi.org/10.1029/2020JD032835

Essentially, this breaks the error into a components from mean shifts and from spatial pattern errors (i.e., the Taylor diagram components of variance error and spatial correlation error). For my purpose, I think I may be able to combine the last two, but I'm not entirely sure.

NMSE = (X_m - X_o)^2 / (X_o - X_o_mean)^2 [RMSE normalized by the variance of the observations]

U = (X_m_mean - X_o_mean)^2 / (X_o - X_o_mean)^2 [Unconditional Error: mean bias normalized by the variance of the observations]

C = (r_mo - \frac{sigma_m}{sigma_o})^2 [Conditional Error: correlation error minus the variance ratio, squared]

P = (1 - r_mo^2) [Phase Error: the error from the spatial pattern]

"""
# %%
import pandas as pd
import xarray as xr
import numpy as np
import os
import logging
from pathlib import Path
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%
varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO"]


def compute_error_decomposition(
    test_path: str,
    control_ds: xr.Dataset,
    var_detect_str: str = "h0",
    time_fcn: callable = lambda da: da.groupby("time.year").mean("time"),
    spatial_dims: list[str] = None,
    weights: xr.DataArray = None,
    file_pattern: str = None,
):
    """Compute the error decomposition per Medeiros (2023) and Simpson et al. (2020) weighting by np.cos(np.deg2rad(ds["lat"])).

    NMSE = (X_m - X_o)^2 / (X_o - X_o_mean)^2 [RMSE normalized by the variance of the observations]

    U = (X_m_mean - X_o_mean)^2 / (X_o - X_o_mean)^2 [Unconditional Error: mean bias normalized by the variance of the observations]

    C = (r_mo - \frac{sigma_m}{sigma_o})^2 [Conditional Error: correlation error minus the variance ratio, squared]

    P = (1 - r_mo^2) [Phase Error: the error from the spatial pattern]

    Args:
        test_path (str): Path to the test dataset.
        control_ds (xr.Dataset): Control dataset.
    """
    # Parse the variable name from the test path, assuming it is in the format of "case/atm/proc/tseries/month_1/case.cam.h0.VAR.nc"
    filename = os.path.splitext(os.path.basename(test_path))[0]
    name_parts = filename.split(".")
    # Filter files that don't match the expected pattern
    if file_pattern is not None and not glob.fnmatch.fnmatch(test_path, f"*/{file_pattern}"):
        return None
    # Filter files without the detection string
    if var_detect_str not in name_parts:
        return None
    marker_idx = name_parts.index(var_detect_str)
    test_var = name_parts[marker_idx + 1]
    if test_var not in control_ds.data_vars: # Don't open datasets without a relevant variable
        return None

    logging.info(f"Got {test_var} from {test_path}")
    test_da = xr.open_dataset(test_path)[test_var]
    # Handle the CESM time coordinate issue and challenges with cftime.DatetimeNoLeap
    if test_da["time"][0]["time.month"] == 2:
        test_da = test_da.assign_coords(
            time=shift_noleap_time_back_one_month(test_da["time"].values)
        )
    test_da = time_fcn(test_da) # Take annual mean to match the control dataset
    control_da = control_ds[test_var]

    # Compute decomposition per time step over spatial dimensions with area weighting.
    if spatial_dims is None:
        spatial_dims = [d for d in test_da.dims if d not in ["time", "year"]]
    else:
        logging.info(f"Using user-supplied spatial_dims: {spatial_dims}")

    if weights is None:
        lat_dim = "lat" if "lat" in test_da.dims else ("latitude" if "latitude" in test_da.dims else None)
        if lat_dim is None:
            raise ValueError("Expected a latitude dimension named 'lat' or 'latitude'.")
        weights = np.cos(np.deg2rad(test_da[lat_dim]))
    else:
        logging.info(f"Using user-supplied weights over dims: {weights.dims}.")

    valid = np.isfinite(test_da) & np.isfinite(control_da)
    m = test_da.where(valid) # test is the "model", m 
    o = control_da.where(valid) # control is the "observations", o

    m_mean = m.weighted(weights).mean(dim=spatial_dims)
    o_mean = o.weighted(weights).mean(dim=spatial_dims)
    mean_bias = m_mean - o_mean

    m_anom = m - m_mean
    o_anom = o - o_mean

    obs_variance = (o_anom ** 2).weighted(weights).mean(dim=spatial_dims)
    sigma_m = np.sqrt((m_anom ** 2).weighted(weights).mean(dim=spatial_dims))
    sigma_o = np.sqrt(obs_variance)
    cov_mo = (m_anom * o_anom).weighted(weights).mean(dim=spatial_dims)
    r_mo = cov_mo / (sigma_m * sigma_o)

    # Compute the NMSE to test closure of the decomposition
    NMSE = ((m - o) ** 2 / obs_variance).weighted(weights).mean(dim=spatial_dims)

    U = (mean_bias ** 2) / obs_variance
    C = (r_mo - (sigma_m / sigma_o)) ** 2
    P = 1 - r_mo ** 2

    # Combine into a single dataarray with a new "error_component" dimension
    error_components = xr.concat([NMSE, U, C, P], dim="error_component")
    error_components = error_components.assign_coords(error_component=["NMSE", "U", "C", "P"])

    return error_components


def crawl_and_process2(input_dir, output_dir, process_fn, **fn_args):
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        out_root = output_dir if rel_root == "." else os.path.join(output_dir, rel_root)
        for name in files:
            src = os.path.join(root, name)
            dst = os.path.join(out_root, name)
            if os.path.exists(dst):
                logging.info(f"{dst} already exists")
                continue
            # logging.info(f"Processing {src}")
            data = process_fn(src, **fn_args)
            if data is None:
                # logging.error(f"Failed to process {src}")
                continue
            logging.info(f"Writing {dst}")
            # Create the encompassing directory if it doesn't exist
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            data.to_netcdf(dst)


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


# %%
if __name__ == "__main__":

    # Glade paths
    control_loadpath_atm = Path("/glade/work/jonahshaw/PRISM_data/control_baselines_atm/")
    save_path_atm = Path("/glade/work/jonahshaw/PRISM_data/error_relativetobaseline_atm/")

    # Path to variables that have been derived from CESM.
    derivedpath_atm_root = "/glade/work/jonahshaw/PRISM_data/derived_vars/"
    derivedpath_ohc_root = "/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/"
    derivedpath_ohf_root = "/glade/work/jonahshaw/PRISM_data/spatial_oceanflux_data/"

    # Keys are the controls and the values are lists of cases to compare to that control.
    compare_cases = {
        "CESM2_WACCM_1850control_0050_0075": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB", "ARISE-1.0"],
        "CESM2_WACCM_1850control_0100_0499": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB", "ARISE-1.0"],
        "CESM2_WACCM_HIST_1850_1864": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB", "ARISE-1.0"],
        "CESM2_WACCM_HIST_2000_2014": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB", "ARISE-1.0"],
        "CESM2_WACCM_HIST_2015_2034": ["CESM2_WACCM_1850control", "CESM2_WACCM_HIST", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB", "ARISE-1.0"],
    }

    # Configure settings for each case to be loaded
    case_dict = {
        "CESM2_WACCM_1850control": {
            "sources": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", f"{derivedpath_atm_root}/CESM2_WACCM_1850control/"],
            "subsources": ["b.e21.BW1850.f09_g17.CMIP6-piControl.001"],
            "file_pattern": "atm/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0.*.nc",
        },
        "CESM2_WACCM_HIST": {
            "sources": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", f"{derivedpath_atm_root}/CESM2_WACCM_HIST/"],
            "subsources": ["b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.002", "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.003"],
            "file_pattern": "atm/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.cam.h0.*.nc",
        },
        "ARISE_SAI": {
            "sources": ["/gdex/data/d651059/ARISE-SAI-1.5/", f"{derivedpath_atm_root}/ARISE_SAI/"],
            "file_pattern": "*/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245*.cam.h0.*.nc",
        },
        "ARISE-1.0": {
            "sources": ["/glade/work/jonahshaw/PRISM_data/ARISE-1.0/", f"{derivedpath_atm_root}/ARISE_SAI/"],
            "file_pattern": "*/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245*.cam.h0.*.nc",
        },
        "CESM2_WACCM_SSP2-4.5": {
            "sources": ["/gdex/data/d651045/CESM2-WACCM-SSP245/", f"{derivedpath_atm_root}/CESM2_WACCM_SSP2-4.5/"],
            "file_pattern": "*/atm/proc/tseries/month_1/b.e21.BWSSP245cmip6.f09_g17.*.nc", 
        },
        "CESM2_WACCM_SSP2-4.5_MCB": {
            "sources": ["/gdex/data/d314006/", f"{derivedpath_atm_root}/CESM2_WACCM_SSP2-4.5_MCB/"],
            "file_pattern": "*/atm/month_1/b.e21.*.nc",
        },
    }

    for control_case in compare_cases:
        logging.info(f"Processing case: {control_case}")
        control_paths = glob.glob(f"{str(control_loadpath_atm)}/{control_case}/{control_case}*.nc")
        control_ds = xr.open_mfdataset(control_paths)
        for compare_case in compare_cases[control_case]:
            logging.info(f"Comparing to case: {compare_case}")
            for source in case_dict[compare_case]["sources"]:
                if "subsources" in case_dict[compare_case].keys():
                    for subs in case_dict[compare_case]["subsources"]:
                        input_dir = os.path.join(source, subs)
                        logging.info(f"Processing input_dir: {input_dir}")
                        crawl_and_process2(
                            input_dir,
                            save_path_atm / control_case / compare_case / subs,
                            compute_error_decomposition,
                            control_ds=control_ds,
                            time_fcn=lambda da: da.groupby("time.year").mean("time"), # Annual averages
                            file_pattern=case_dict[compare_case]["file_pattern"],
                        )
                else:
                    logging.info(f"Processing input_dir: {source}")
                    crawl_and_process2(
                        source,
                        save_path_atm / control_case / compare_case,
                        compute_error_decomposition,
                        control_ds=control_ds,
                        time_fcn=lambda da: da.groupby("time.year").mean("time"), # Annual averages
                        file_pattern=case_dict[compare_case]["file_pattern"],
                    )

    # %%
# import matplotlib.pyplot as plt
# test_path = "data/error_relativetobaseline/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.cam.h0.CLDTOT.170001-174912.nc"
# test_var = "CLDTOT"
# test_path = "data/error_relativetobaseline/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.cam.h0.FLNT.115001-119912.nc"
# test_var = "FLNT"
# test_path = "data/error_relativetobaseline/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002.cam.h0.FSNS.110001-114912.nc"
# test_var = "FSNS"
# test_ds = xr.open_dataset(test_path)

# fig = plt.figure()
# fig.suptitle("NMSE and components for variable: " + test_var)
# test_ds[test_var].sel(error_component="NMSE").plot(label="NMSE")
# test_ds[test_var].sel(error_component="U").plot(label="U")
# test_ds[test_var].sel(error_component="C").plot(label="C")
# test_ds[test_var].sel(error_component="P").plot(label="P")
# plt.legend()

# fig = plt.figure()
# fig.suptitle("Testing closure of the decomposition: NMSE versus the sum of U, C, and P")
# test_ds[test_var].drop_sel(error_component="NMSE").sum("error_component").plot()
# test_ds[test_var].sel(error_component="NMSE").plot()

# fig = plt.figure()
# fig.suptitle("Testing closure of the decomposition: NMSE minus the sum of U, C, and P")
# (test_ds[test_var].sel(error_component="NMSE") - test_ds[test_var].drop_sel(error_component="NMSE").sum("error_component")).plot()
# %%
