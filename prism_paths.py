"""
Central path configuration for PRISM.

Every script imports its directories from here instead of hardcoding them, so the
same code runs unchanged on the laptop and on NCAR Casper (OnDemand Jupyter or PBS).

Which machine are we on?
    1. If the environment variable PRISM_SYSTEM is set to "casper" or "laptop", use it.
    2. Otherwise, if /glade exists we are on an NCAR system (Casper/Derecho share GLADE).
    3. Otherwise, assume the laptop.

Three kinds of directories are defined below:

    * Shared, read-only input collections on GLADE (CMIP6 time series, GDEX). These
      are the same for every NCAR user and only exist on Casper.
    * WORK_ROOT: where this project's derived products are written and read back
      (spatial averages, derived variables, control baselines, OHC, NMSE, ...).
      Casper: /glade/work/$USER/PRISM_data  (not purged; per-user quota)
      Laptop: <repo>/data                   (small, unzipped subset for testing)
      Override on either machine with the environment variable PRISM_WORK_ROOT.
    * REPO_ROOT: this git checkout (code, job_scripts/, figures/, job_logs/).

Product sub-directories keep the names Jonah used under /glade/work/jonahshaw/PRISM_data,
with ONE exception: the spatially averaged data is called "spatial_averages_data" on
GLADE but "RadInt_procdata" in the unzipped laptop copy (that is the folder name inside
procdata.zip / zipped_data). SPATIAL_AVERAGES_DIR hides that difference.

PBS_ACCOUNT (the project code jobs are charged to) is chosen per user, like WORK_ROOT:
PBS_ACCOUNT_CASPER / PBS_ACCOUNT from the environment, else a per-user default.

Run `python prism_paths.py` to print the resolved paths for the current machine, or
`python prism_paths.py --account` to print only the project code.
"""

import getpass
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Machine detection
# ----------------------------------------------------------------------------
_VALID_SYSTEMS = ("casper", "laptop")


def _detect_system() -> str:
    """Return "casper" or "laptop" (see module docstring for the rules)."""
    override = os.environ.get("PRISM_SYSTEM", "").strip().lower()
    if override:
        if override not in _VALID_SYSTEMS:
            raise ValueError(
                f"PRISM_SYSTEM={override!r} is not one of {_VALID_SYSTEMS}"
            )
        return override
    # /glade is mounted on every NCAR HPC node (login, compute, OnDemand Jupyter)
    return "casper" if Path("/glade").is_dir() else "laptop"


SYSTEM = _detect_system()
ON_CASPER = SYSTEM == "casper"
USER = getpass.getuser()

# Directory that contains this file = the root of the PRISM code checkout,
# wherever it was cloned (laptop monorepo subfolder or a clone on GLADE).
REPO_ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Shared, read-only input collections on GLADE (identical for all NCAR users)
# ----------------------------------------------------------------------------
# CESM2 / CESM2-WACCM CMIP6 single-variable time series (piControl, historical, ...)
CMIP6_TIMESERIES_DIR = Path("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6")
# Geoscience Data Exchange (formerly the RDA; dataset IDs d651059, d651078, ...)
GDEX_DIR = Path("/gdex/data")

# ----------------------------------------------------------------------------
# Machine-specific roots
# ----------------------------------------------------------------------------
if ON_CASPER:
    _default_work_root = Path(f"/glade/work/{USER}/PRISM_data")
    # Derecho scratch is mounted on Casper; 30 TB, files purged after 180 days.
    # Used only for throwaway files (Dask spill), never for results.
    SCRATCH_DIR = Path(f"/glade/derecho/scratch/{USER}")
    _spatial_averages_name = "spatial_averages_data"
else:
    _default_work_root = REPO_ROOT / "data"
    SCRATCH_DIR = REPO_ROOT / "data" / "scratch"
    _spatial_averages_name = "RadInt_procdata"

WORK_ROOT = Path(os.environ.get("PRISM_WORK_ROOT", _default_work_root))

# ----------------------------------------------------------------------------
# Derived-product directories (written by the processing scripts, read by later ones)
# ----------------------------------------------------------------------------
SPATIAL_AVERAGES_DIR = WORK_ROOT / _spatial_averages_name  # J02 output; J03/J04/J16 input
DERIVED_VARS_DIR = WORK_ROOT / "derived_vars"  # J14 output (new analysis variables)
ARISE_1_0_DIR = WORK_ROOT / "ARISE-1.0"  # J13 download target (from AWS S3)
CONTROL_BASELINES_ATM_DIR = WORK_ROOT / "control_baselines_atm"  # J05 output
CONTROL_BASELINES_OCN_DIR = WORK_ROOT / "control_baselines_ocn"  # J05 output
SPATIAL_OHC_DIR = WORK_ROOT / "spatial_OHC_data"  # J09/J10 OHC output
OHC_ANCILLARY_DIR = SPATIAL_OHC_DIR / "ancillary_files"  # grid/area files for J09
SPATIAL_OCEANFLUX_DIR = WORK_ROOT / "spatial_oceanflux_data"  # J09/J10 OHF output
ERROR_ATM_DIR = WORK_ROOT / "error_relativetobaseline_atm"  # J06 output; J08 input
SPATIAL_MAPS_ATM_DIR = WORK_ROOT / "spatial_maps_atm"  # J11 output; J12 input

# ----------------------------------------------------------------------------
# Repo-relative directories
# ----------------------------------------------------------------------------
JOB_SCRIPTS_DIR = REPO_ROOT / "job_scripts"
JOB_LOGS_DIR = REPO_ROOT / "job_logs"
FIGURES_DIR = REPO_ROOT / "figures"
# Older spatial-map files kept inside the repo's data/ folder (used by J12_plot_spatialmaps.py)
REPO_SPATIAL_MAPS_DIR = REPO_ROOT / "data" / "spatial_maps"
# Placeholder single-run OHC series used by J04 until real per-case OHC files exist
PLACEHOLDER_OHC_FILE = REPO_ROOT / "data" / "OHC_data" / "cesm2_ohc_global_timeseries.nc"
# CO2 record for Will Kamp's simple model (simple_model_kamp.py); small CSVs tracked in git
SIMPLE_MODEL_DATA_DIR = REPO_ROOT / "data" / "simple_model"
# Dask worker spill directory (J06). Scratch on Casper keeps it out of the 50 GB home quota.
DASK_LOCAL_DIR = SCRATCH_DIR / "PRISM_dask_workers"

# ----------------------------------------------------------------------------
# PBS project code (the allocation that NCAR charges for each job)
# ----------------------------------------------------------------------------
# Chosen by WHO is running, the same way WORK_ROOT is chosen from $USER: each person's
# jobs must bill their own allocation, and PBS rejects a project code the user is not on.
# Order of precedence:
#   1. PBS_ACCOUNT_CASPER, then PBS_ACCOUNT (the variables NCAR's qcmd/qinteractive
#      already read, so one line in ~/.bashrc covers PRISM too), see
#      https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/
#   2. The per-user default below.
# Add your own code here if you prefer it in the repo over ~/.bashrc.
_DEFAULT_PBS_ACCOUNTS = {
    "jonahshaw": "UCUC0007",  # Jonah Shaw (CU Boulder allocation)
}
PBS_ACCOUNT = (
    os.environ.get("PBS_ACCOUNT_CASPER")
    or os.environ.get("PBS_ACCOUNT")
    or _DEFAULT_PBS_ACCOUNTS.get(USER)
)


def require_pbs_account() -> str:
    """Return PBS_ACCOUNT, or fail loudly before any job is submitted without one."""
    if not PBS_ACCOUNT:
        raise RuntimeError(
            f"No PBS project code for user {USER!r}. Add `export PBS_ACCOUNT=<code>` to "
            "~/.bashrc, or add the user to _DEFAULT_PBS_ACCOUNTS in prism_paths.py."
        )
    return PBS_ACCOUNT


def summary() -> str:
    """Human-readable listing of every resolved path, for sanity checks."""
    names = [n for n in globals() if n.isupper() and not n.startswith("_")]
    width = max(len(n) for n in names)
    lines = []
    for n in names:
        value = globals()[n]
        flag = ""
        if isinstance(value, Path):
            flag = "" if value.exists() else "   (does not exist yet)"
        lines.append(f"{n:<{width}} = {value}{flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # `python prism_paths.py --account` prints only the project code, for shell scripts
    if "--account" in sys.argv[1:]:
        print(require_pbs_account())
    else:
        print(summary())
