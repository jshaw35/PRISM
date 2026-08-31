"""
Compute the background radiation and precipitation proxy states from CESM control simulations.
Results will be maps of these variables.

"""
# %%
import os
import xarray as xr
import numpy as np
import logging


# import functions from J14_CESM_compute_THERMO_PRECIP.py
from J14_CESM_compute_newvars_parallel import (
    compute_thermoprecip,
    compute_thermoprecip_wrapper,
    compute_toaimbalance_wrapper,
    crawl_and_process2,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# %%
if __name__ == "__main__":
    # Set output path
    machine = "glade"
    if machine == "glade":
        savepath_root = "/glade/work/jonahshaw/PRISM_data/derived_vars/"
    # Identify files with the FLNT variable:
    match_var = "FLNT"
    # Make sure there are two arguments: input file path and output file path
    import sys
    if len(sys.argv) != 3:
        print("Usage: python J13_OHC_calculation_single.py <load_path> <save_path>")
        sys.exit(1)
    load_path = sys.argv[1]
    save_path = sys.argv[2]

    logging.info(f"Processing case: {load_path} to {save_path}")
    crawl_and_process2(
        input_dir=load_path,
        output_dir=save_path,
        process_fn=compute_thermoprecip_wrapper,
        match_var=match_var,
    )

    crawl_and_process2(
        input_dir=load_path,
        output_dir=save_path,
        process_fn=compute_toaimbalance_wrapper,
        match_var=match_var,
    )

# %%