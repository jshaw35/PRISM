from pathlib import Path
import os
import xarray as xr
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def average_spatially(
    datapath,
    verbose=False,
    average_vars = ["CLDTOT", "FLNS", "FLNSC", "FLNT", "FLNTC", "FLNR", "FSNR", "FSNT", "FSNS", "FSNSC", "FSNTOA", "FSNTC", "FLNTCLR", "FSNTOAC", "LHFLX", "SHFLX",],
):
    ds = xr.open_dataset(datapath)

    varlist = [i for i in list(ds.data_vars) if i in average_vars]

    lon_average = ds[varlist].mean(dim="lon")
    T_average = lon_average.weighted(np.cos(np.deg2rad(ds.lat))).mean(dim="lat")
    T_average_SH = lon_average.sel(lat=slice(-90,0)).weighted(np.cos(np.deg2rad(ds.lat))).mean(dim="lat")
    T_average_NH = lon_average.sel(lat=slice(0,90)).weighted(np.cos(np.deg2rad(ds.lat))).mean(dim="lat")
    
    T_average = T_average.assign_coords(spatial="G").expand_dims("spatial")
    T_average_SH = T_average_SH.assign_coords(spatial="SH").expand_dims("spatial")
    T_average_NH = T_average_NH.assign_coords(spatial="NH").expand_dims("spatial")

    out = xr.combine_by_coords([T_average, T_average_SH, T_average_NH])
    return out


def crawl_and_process(input_dir, output_dir, process_fn):
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
            data = process_fn(src)
            logging.info(f"Writing {dst}")
            data.to_netcdf(dst)
            # mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
            # with open(dst, mode) as f:
                # f.write(data)

if __name__ == "__main__":
    curc_lme_datapath = "/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM_LME/"
    curc_lme_outpath = "/home/josh2250/kaydata/jshaw/RadInt_procdata/CESM_LME/"

    curc_cesm2_245_datapath = "/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_WACCM_SSP2-4.5/"
    curc_cesm2_245_outpath = "/home/josh2250/kaydata/jshaw/RadInt_procdata/CESM2_WACCM_SSP2-4.5/"

    curc_ariseSAI_datapath = "/home/josh2250/kaydata/jshaw/RadInt_rawdata/ARISE_SAI/"
    curc_ariseSAI_outpath = "/home/josh2250/kaydata/jshaw/RadInt_procdata/ARISE_SAI/"

    crawl_and_process(curc_lme_datapath, curc_lme_outpath, average_spatially)
    crawl_and_process(curc_cesm2_245_datapath, curc_cesm2_245_outpath, average_spatially)
    crawl_and_process(curc_ariseSAI_datapath, curc_ariseSAI_outpath, average_spatially)
