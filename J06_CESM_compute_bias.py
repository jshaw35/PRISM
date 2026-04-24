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
import xarray as xr
import numpy as np
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%
varlist = ['CLDTOT', 'FLNR', 'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLNTCLR', 'FLUT', 'FSNR', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA', 'FSNTOAC', 'LHFLX', 'SHFLX', 'TS', "PRECT", "PRECC", "PRECL"]


def compute_thermoprecip(
    ds,
):
    """Compute the thermodynamically-driven precipitation per O'Gorman 2012.
    L del P = del R_TOA - del R_SFC - del SHFLX
    P = (R_TOA - R_SFC - SHFLX) / L

    Where the convention is that positive value are upwards.

    Args:
        ds (_type_): _description_
    """
    L = 2.5e6  # J/kg, latent heat of vaporization or 2.257e3 J/kg or 2.45e6 J/kg
    # 2.5e6 per this lecture slide: https://ethz.ch/content/dam/ethz/special-interest/usys/iac/iac-dam/documents/edu/courses/climatological_and_hydrological_field_work/radiation_2025.pdf
    vars = ["FLNT", "FSNT", "FLNS", "FSNS", "SHFLX"]
    assert set(vars).issubset(set(ds.data_vars)), "Not all variables in varlist are in the dataset."

    R_LW = ds["FLNT"] - ds["FLNS"] # Positive upwards convention for LW
    R_SW = -1 * (ds["FSNT"] - ds["FSNS"]) # The convention is positive downwards for SW, so invert
    R_ATM = R_LW + R_SW # Net radiation emitted/lost by the atmosphere
    SHFLX = ds["SHFLX"]
    # Compute in units of kg m^-2 s^-1
    P = (R_ATM - SHFLX) / L

    # R_TOA = ds["FLNT"] - ds["FSNT"] # Positive upwards convention for LW
    # R_SFC = (ds["FLNS"] - ds["FSNS"]) # The convention is positive downwards for SW, so invert
    # SHFLX = ds["SHFLX"]
    # # Compute in units of kg m^-2 s^-1
    # P = (R_TOA - R_SFC - SHFLX) / L

    # Convert to mm/day: 1000 mm / m, 86400 s / day, 1000 kg / m^3 for water density (last two cancel out)
    P = P * 86400
    P.attrs["long_name"] = "Thermodynamically-driven precipitation"
    P.attrs["units"] = "mm/day"
    P.name = "PRECIP_THERMO"
    return P


def compute_error_decomposition(
    test_path: str,
    control_ds: xr.Dataset,
    var_detect_str: str = "h0",
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
    marker_idx = name_parts.index(var_detect_str)
    test_var = name_parts[marker_idx + 1]

    test_da = xr.open_dataset(test_path)[test_var]
    control_da = control_ds[test_var]

    # Compute decomposition per time step over spatial dimensions with area weighting.
    spatial_dims = [d for d in test_da.dims if d != "time"]
    lat_dim = "lat" if "lat" in test_da.dims else ("latitude" if "latitude" in test_da.dims else None)
    if lat_dim is None:
        raise ValueError("Expected a latitude dimension named 'lat' or 'latitude'.")

    weights = np.cos(np.deg2rad(test_da[lat_dim]))
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
        os.makedirs(out_root, exist_ok=True)
        for name in files:
            src = os.path.join(root, name)
            dst = os.path.join(out_root, name)
            if os.path.exists(dst):
                logging.info(f"{dst} already exists")
                continue
            logging.info(f"Processing {src}")
            data = process_fn(src, **fn_args)
            if data is None:
                logging.error(f"Failed to process {src}")
                continue
            logging.info(f"Writing {dst}")
            data.to_netcdf(dst)


# %%
if __name__ == "__main__":
    # CURC paths
    # control_loadpath = Path("/home/josh2250/projects/PRISM/data/control_baselines/")
    # rawdata_loadpath = Path("/home/josh2250/kaydata/jshaw/RadInt_rawdata/")
    # savepath = Path("/home/josh2250/projects/PRISM/data/error_relativetobaseline/")

    # case_dict = {
    #     "CESM2_LME_control": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008","CESM2_LME/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/atm/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.cam.h0.*"],
    #     "CESM2_1850control": ["b.e21.B1850.f09_g17.CMIP6-piControl.001", "CESM2_1850control/b.e21.B1850.f09_g17.CMIP6-piControl.001/atm/proc/tseries/month_1/b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.*.nc"],
    # }

    # Glade test paths
    control_loadpath = Path("/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/control_baselines/")
    rawdata_loadpath = Path("/gdex/data/")
    save_path = Path("/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/error_relativetobaseline/")

    case_dict = {
        "CESM2_LME_control": ["b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008","d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/atm/proc/tseries/month_1/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008.cam.h0.*"],
    }

    rawdata_loadpath = Path("/home/josh2250/kaydata/jshaw/RadInt_rawdata/")
    savepath = Path("/home/josh2250/projects/PRISM/data/error_relativetobaseline/")


    compare_cases = {
        "CESM2_LME_control": ["CESM2_LME"],
        "CESM2_1850control": ["CESM2_1850control", "CESM2_LE", "ARISE_SAI", "CESM2_WACCM_SSP2-4.5", "CESM2_WACCM_SSP2-4.5_MCB"]
    }
    # Find the control files in the control load path where the filename matches the pattern in case_dict + nc
    for case, caselist in case_dict.items():
        logging.info(f"Processing case: {case}")
        for compare_case in caselist:
            logging.info(f"Comparing to case: {compare_case}")
            input_dir = rawdata_loadpath / compare_case
            control_path = str(control_loadpath) + "/" + case + ".nc"
            control_ds = xr.open_dataset(control_path)
            
            crawl_and_process2(input_dir, savepath, compute_error_decomposition, control_ds=control_ds)
            break
        break
