"""Configuration and NCAR Glade paths for the OHC/iEEI rebuild.

Fill in GLADE_USERNAME and PBS_PROJECT_ACCOUNT (and, if needed,
OCN_GRID_FILE) before running anything on Glade/Casper. See README.md for
the full list of decisions this build makes.
"""

# --- Glade account details --------------------------------------------------
# PBS_PROJECT_ACCOUNT is taken from this user's other active job scripts
# (e.g. era5_atm/cdo_daily.sh) — confirm it is still the right account to
# charge before submitting a long-running job.
GLADE_USERNAME = "wkamp"
PBS_PROJECT_ACCOUNT = "UCUB0144"

# Where output NetCDF files are written.
OUTPUT_ROOT = f"/glade/work/{GLADE_USERNAME}/GEO/PRISM/PRISM/ohc_ieei_rebuild/"

# --- 1850 piControl case ---------------------------------------------------
# NCAR CMIP6 timeseries archive location (shared community data, not
# user-specific — confirm it still resolves before running, campaign-storage
# paths can move).
CASE_STR = "b.e21.BW1850.f09_g17.CMIP6-piControl.001"
CASE_ROOT = f"/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/{CASE_STR}/"
OCN_TSERIES_DIR = CASE_ROOT + "ocn/proc/tseries/month_1/"
ATM_TSERIES_DIR = CASE_ROOT + "atm/proc/tseries/month_1/"

TEMP_FILE_GLOB = f"{CASE_STR}.pop.h.TEMP.*.nc"
FSNT_FILE_GLOB = f"{CASE_STR}.cam.h0.FSNT.*.nc"
FLNT_FILE_GLOB = f"{CASE_STR}.cam.h0.FLNT.*.nc"

# If the TEMP files above don't already carry TAREA/KMT/dz/z_w_bot (some
# archive layouts strip static grid variables out of single-variable-per-file
# time series), point this at any POP2 history file for this case that does
# contain them.
OCN_GRID_FILE = None

# CAM variables used for iEEI = ASR - OLR (New_OHC_iEEI.md sec. 2.3).
ASR_VAR = "FSNT"
OLR_VAR = "FLNT"
