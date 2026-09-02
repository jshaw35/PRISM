"""
Compute the mean of variables of interest to be treated as controls when assessing change.
Config specifies where to look for model output, how to identify the correct files, and what time slice to average over
Results will be maps of these variables.

"""
# %%
from pathlib import Path
import os
import xarray as xr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%
atm_vars = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FLUTC', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL", "PRECIP_THERMO", "FNNT"]
ocn_vars = ["OHC", "OHF"]

def compute_background_state(
    rawdata_root,
    save_path: Path,
    case_name: str,
    glob_str: str,
    mask=list[int] | None, # not sure what this was intended for
    tslice=None,
):
    os.makedirs(save_path / case_name, exist_ok=True)
    filepaths = list(rawdata_root.glob(glob_str))
    if len(filepaths) < 1:
        logging.info(f"No files found for root '{str(rawdata_root)}' and glob string '{glob_str}'")
        return

    for _var in atm_vars:
        file_savepath = save_path / case_name / f"{case_name}_{_var}.nc"
        if file_savepath.exists():
            logging.info(f"{file_savepath} already exists, skipping computation for case {case_name}")
            continue
        subset_filepaths = [fp for fp in filepaths if f".{_var}." in fp.name]
        subset_filepaths.sort()  # Ensure files are in a consistent order
        if not subset_filepaths:
            logging.warning(f"No files found for variable {_var} in case {case_name} with pattern {rawdata_root}/{glob_str}")
            continue
        logging.info(f"Processing variable: {_var} with {len(subset_filepaths)} files for case {case_name}")
        tmean_ds_list = []
        for i,fp in enumerate(subset_filepaths):
            ds = xr.open_dataset(fp)[_var]
            if tslice is not None:
                ds = ds.sel(time=tslice)
            # Get the time length and add it to weight the average by the number of time steps in each file later
            time_len = ds.sizes["time"]
            ds_tmean = ds.mean(dim="time")
            ds_tmean = ds_tmean.assign_coords(index_t=i).expand_dims("index_t")
            # Add a variable indexed by the new index_t dimension that contains the time length of the original dataset for weighting later
            ds_tmean = ds_tmean.assign_coords(time_len=("index_t", [time_len]))
            tmean_ds_list.append(ds_tmean)
        # Compute the weighted average of the time means from each file
        tmean_ds = xr.concat(tmean_ds_list, dim="index_t")
        weighted_tmean = tmean_ds.weighted(tmean_ds["time_len"]).mean(dim="index_t")
        weighted_tmean.to_netcdf(file_savepath)


def compute_background_state_ohc(
    rawdata_root,
    save_path,
    case_name: str,
    glob_str: str,
    mask=list[int] | None,
    tslice=None,
):
    os.makedirs(save_path / case_name, exist_ok=True)
    filepaths = list(rawdata_root.glob(glob_str))
    if len(filepaths) < 1:
        logging.info(f"No files found for root '{str(rawdata_root)}' and glob string '{glob_str}'")
        return

    for _var in ocn_vars:
        file_savepath = save_path / case_name / f"{case_name}_{_var}.nc"
        if file_savepath.exists():
            logging.info(f"{file_savepath} already exists, skipping computation for case {case_name}")
            continue
        subset_filepaths = [fp for fp in filepaths if f".{_var}." in fp.name]
        subset_filepaths.sort()  # Ensure files are in a consistent order
        if not subset_filepaths:
            logging.warning(f"No files found for variable {_var} in case {case_name} with pattern {glob_str}")
            return
        logging.info(f"Processing variable: {_var} with {len(subset_filepaths)} files for case {case_name}")
        tmean_ds_list = []
        for i,fp in enumerate(subset_filepaths):
            ds = xr.open_dataset(fp)[_var]
            if tslice is not None:
                ds = ds.sel(time=tslice)
            # Get the time length and add it to weight the average by the number of time steps in each file later
            time_len = ds.sizes["time"]
            ds_tmean = ds.mean(dim="time")
            ds_tmean = ds_tmean.assign_coords(index_t=i).expand_dims("index_t")
            # Add a variable indexed by the new index_t dimension that contains the time length of the original dataset for weighting later
            ds_tmean = ds_tmean.assign_coords(time_len=("index_t", [time_len]))
            tmean_ds_list.append(ds_tmean)
        # Compute the weighted average of the time means from each file
        tmean_ds = xr.concat(tmean_ds_list, dim="index_t")
        weighted_tmean = tmean_ds.weighted(tmean_ds["time_len"]).mean(dim="index_t")
        weighted_tmean.to_netcdf(file_savepath)


# %%
if __name__ == "__main__":

    # If on glade
    save_path_atm = Path("/glade/work/jonahshaw/PRISM_data/control_baselines_atm/")
    save_path_ocn = Path("/glade/work/jonahshaw/PRISM_data/control_baselines_ocn/")

    # Path to variables that have been derived from CESM.
    derivedpath_atm_root = "/glade/work/jonahshaw/PRISM_data/derived_vars/"
    derivedpath_ohc_root = "/glade/work/jonahshaw/PRISM_data/spatial_OHC_data/"
    derivedpath_ohf_root = "/glade/work/jonahshaw/PRISM_data/spatial_oceanflux_data/"

    # Configure settings
    case_dict = {
        "CESM2_WACCM_HIST_1850_1864": {
            "sources": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", f"{derivedpath_atm_root}/CESM2_WACCM_HIST/"],
            "file_pattern": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/atm/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.cam.h0.*.nc",
            "tslice": slice("1850", "1864"),
        },
        "CESM2_WACCM_HIST_2000_2014": {
            "sources": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", f"{derivedpath_atm_root}/CESM2_WACCM_HIST/"],
            "file_pattern": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/atm/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.cam.h0.*.nc",
            "tslice": slice("2000", "2014"),
        },
        "CESM2_WACCM_HIST_2015_2034": {
            "sources": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", f"{derivedpath_atm_root}/CESM2_WACCM_HIST/"],
            "file_pattern": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/atm/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.cam.h0.*.nc",
            "tslice": slice("2015", "2034"),
        },
        "CESM2_WACCM_1850control_0100_0499": {
            "sources": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", f"{derivedpath_atm_root}/CESM2_WACCM_1850control/"],
            "file_pattern": "b.e21.BW1850.f09_g17.CMIP6-piControl.001/atm/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0.*.nc",
            "tslice": slice("0100", None),
        },
        "CESM2_WACCM_1850control_0050_0075": {
            "sources": ["/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/", f"{derivedpath_atm_root}/CESM2_WACCM_1850control/"],
            "file_pattern": "b.e21.BW1850.f09_g17.CMIP6-piControl.001/atm/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0.*.nc",
            "tslice": slice("0050", "0075"),
        },
    }

    ohc_case_dict = {
        "CESM2_WACCM_HIST_1850_1864": {
            "sources": [f"{derivedpath_ohc_root}/CESM2_WACCM_HIST/", f"{derivedpath_ohf_root}/CESM2_WACCM_HIST/"],
            "file_pattern": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/ocn/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.pop.h.*.nc",
            "tslice": slice("1850", "1864"),
        },
        "CESM2_WACCM_HIST_2000_2014": {
            "sources": [f"{derivedpath_ohc_root}/CESM2_WACCM_HIST/", f"{derivedpath_ohf_root}/CESM2_WACCM_HIST/"],
            "file_pattern": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/ocn/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.pop.h.*.nc",
            "tslice": slice("2000", "2014"),
        },
        "CESM2_WACCM_HIST_2015_2034": {
            "sources": [f"{derivedpath_ohc_root}/CESM2_WACCM_HIST/", f"{derivedpath_ohf_root}/CESM2_WACCM_HIST/"],
            "file_pattern": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?/ocn/proc/tseries/month_1/b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.00?.pop.h.*.nc",
            "tslice": slice("2015", "2034"),
        },
        "CESM2_WACCM_1850control_0100_0499": {
            "sources": [f"{derivedpath_ohc_root}/CESM2_WACCM_1850control/", f"{derivedpath_ohf_root}/CESM2_WACCM_1850control/"],
            "file_pattern": "ocn/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h.*.nc",
            "tslice": slice("0100", None),
        },
        "CESM2_WACCM_1850control_0050_0075": {
            "sources": [f"{derivedpath_ohc_root}/CESM2_WACCM_1850control/", f"{derivedpath_ohf_root}/CESM2_WACCM_1850control/"],
            "file_pattern": "ocn/proc/tseries/month_1/b.e21.BW1850.f09_g17.CMIP6-piControl.001.pop.h.*.nc",
            "tslice": slice("0050", "0075"),
        },
    }

    for case, subdict in case_dict.items():
        logging.info("Processing: %s, pattern: %s" % (case, subdict["file_pattern"]))
        for _source in subdict["sources"]:
            _ = compute_background_state(
                Path(_source),
                save_path_atm,
                case,
                subdict["file_pattern"],
                tslice=subdict["tslice"],
            )

    for case, subdict in ohc_case_dict.items():
        logging.info("Processing: %s, pattern: %s" % (case, subdict["file_pattern"]))
        for _source in subdict["sources"]:
            _ = compute_background_state_ohc(
                Path(_source),
                save_path_ocn,
                case,
                subdict["file_pattern"],
                tslice=subdict["tslice"],
            )

    # %%
    # Testing code
    # import matplotlib.pyplot as plt

    # testlist = ["/glade/work/jonahshaw/PRISM_data/control_baselines_ocn/CESM2_WACCM_HIST_1850_1864/CESM2_WACCM_HIST_1850_1864_OHC.nc", "/glade/work/jonahshaw/PRISM_data/control_baselines_ocn/CESM2_WACCM_HIST_1850_1864/CESM2_WACCM_HIST_1850_1864_OHF.nc"]
    # testlist2 = ["/glade/work/jonahshaw/PRISM_data/control_baselines_ocn/CESM2_WACCM_HIST_2000_2014/CESM2_WACCM_HIST_2000_2014_OHC.nc", "/glade/work/jonahshaw/PRISM_data/control_baselines_ocn/CESM2_WACCM_HIST_2000_2014/CESM2_WACCM_HIST_2000_2014_OHF.nc"]

    # test_ds = xr.open_mfdataset(testlist)
    # test_ds2 = xr.open_mfdataset(testlist2)

    # fig,axs = plt.subplots(2,3, figsize=(15,7))
    # fig.subplots_adjust(wspace=0.4, hspace=0.35)
    # (test_ds["OHF"]).plot(ax=axs[0,0])
    # (test_ds2["OHF"]).plot(ax=axs[0,1])
    # (test_ds2["OHF"] - test_ds["OHF"]).plot(ax=axs[0,2])
    # (test_ds["OHC"].isel(ohc_depth=0)).plot(ax=axs[1,0])
    # (test_ds2["OHC"].isel(ohc_depth=0)).plot(ax=axs[1,1])
    # (test_ds2["OHC"] - test_ds["OHC"]).isel(ohc_depth=0).plot(ax=axs[1,2])

    # testlist3 = ["/glade/work/jonahshaw/PRISM_data/control_baselines_atm/CESM2_WACCM_HIST_1850_1864/CESM2_WACCM_HIST_1850_1864_FLNT.nc", "/glade/work/jonahshaw/PRISM_data/control_baselines_atm/CESM2_WACCM_HIST_1850_1864/CESM2_WACCM_HIST_1850_1864_FNNT.nc"]
    # testlist4 = ["/glade/work/jonahshaw/PRISM_data/control_baselines_atm/CESM2_WACCM_HIST_2000_2014/CESM2_WACCM_HIST_2000_2014_FLNT.nc", "/glade/work/jonahshaw/PRISM_data/control_baselines_atm/CESM2_WACCM_HIST_2000_2014/CESM2_WACCM_HIST_2000_2014_FNNT.nc"]
    # atm_test = xr.open_mfdataset(testlist3)
    # atm_test2 = xr.open_mfdataset(testlist4)

    # fig,axs = plt.subplots(2,3, figsize=(15,7))
    # fig.subplots_adjust(wspace=0.4)
    # (atm_test["FLNT"]).plot(ax=axs[0,0])
    # (atm_test2["FLNT"]).plot(ax=axs[0,1])
    # (atm_test2["FLNT"] - atm_test["FLNT"]).plot(ax=axs[0,2])
    # (atm_test["FNNT"]).plot(ax=axs[1,0])
    # (atm_test2["FNNT"]).plot(ax=axs[1,1])
    # (atm_test2["FNNT"] - atm_test["FNNT"]).plot(ax=axs[1,2])

    # %%