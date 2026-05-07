#!/usr/bin/env python3
"""
Script for computing ocean heat content (OHC) from a single input file. Called by job_scripts/compute_ohc_single.sh.
"""
# %%
import os
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# import functions from J09_OHC_calculation_parallel.py
from J09_OHC_calculation_parallel import (
    compute_ohc_single_file,
    compute_OHC,
    shift_noleap_time_back_one_month,
)


# %%
if __name__ == "__main__":
    # Make sure there are two arguments: input file path and output file path
    import sys
    if len(sys.argv) != 3:
        print("Usage: python J09_OHC_calculation_single.py <input_file_path> <output_file_path>")
        sys.exit(1)
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    compute_ohc_single_file(input_file_path, output_file_path)

    # %%
