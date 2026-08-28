"""
Download ARISE1.0 data from AWS S3 bucket
"""
# %%
import numpy as np
import s3fs
from pathlib import Path

# %%
# Index files by variable name (the token between '.h0.' and the date range)
def var_from_filename(path, separator):
    name = path.split("/")[-1]
    # e.g. ....cam.h0.CLDTOT.206001-209912.nc -> CLDTOT
    parts = name.split(".")
    try:
        h0_idx = parts.index(separator)
        return parts[h0_idx + 1]
    except (ValueError, IndexError):
        return None


# %%
if __name__ == "__main__":

    fs = s3fs.S3FileSystem(anon=True)

    # See top-level layout
    print("Top of bucket:")
    for item in fs.ls("ncar-cesm2-arise/raw/"):
        print(" ", item)

    # Look for ARISE-1.0 experiments — case and naming may differ
    # Common naming patterns to try:
    candidates = [
        # "ncar-cesm2-arise/ARISE-SAI-1.0/",
        # "ncar-cesm2-arise/ARISE-SAI-1.0-EXTENDED/",
        "ncar-cesm2-arise/raw/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.001",
        "ncar-cesm2-arise/raw/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.001",
        # "ncar-cesm2-arise/raw/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.*",
        # "ncar-cesm2-arise/raw/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.*",
    ]
    for c in candidates:
        try:
            print(f"\nContents of {c}:")
            for item in fs.ls(c):
                print(" ", item)
        except FileNotFoundError:
            print(f"  (not found)")

    # %%

    fs = s3fs.S3FileSystem(anon=True)

    # ---- CONFIGURE ----
    BUCKET = "ncar-cesm2-arise"
    EXPERIMENT = "raw"
    CASE = "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.001"
    SUBDIR = "atm/proc/tseries/month_1"
    OCN_SUBDIR = "ocn/proc/tseries/month_1"
    LOCAL_DIR = Path("/glade/work/jonahshaw/PRISM_data/ARISE-1.0/") # JKS adjust

    VARIABLES = [
        "CLDTOT", "FLNR", "FLNS", "FLNSC", "FLNT", "FLNTC", "FLNTCLR",
        "FLUT", "FSNS", "FSNSC", "FSNT", "FSNTOA", "FSNTOAC",
        "LHFLX", "PRECT", "SHFLX", "TS",
    ]
    OCN_VARIABLES = ["TEMP", "RHO", "QFLUX", "SHF", "SSH"]
    # -------------------

    # %%
    # Set up cases to pull from.
    cases = \
        [f"b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.{member:03d}" for member in range(1, 11)] + \
        [f"b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.{member:03d}" for member in range(1, 11)]

    # %%
    prefix = f"{BUCKET}/{EXPERIMENT}/{CASE}/{SUBDIR}/"
    print(f"Listing: {prefix}")
    all_files = fs.ls(prefix)
    print(f"Found {len(all_files)} files in directory\n")

    available = {}
    for f in all_files:
        v = var_from_filename(f, "h0")
        if v:
            available.setdefault(v, []).append(f)

    # Report
    print("=== Verification report ===")
    found, missing = [], []
    for v in VARIABLES:
        if v in available:
            for f in available[v]:
                print(f"  [OK]      {v:15s} -> {f.split('/')[-1]}")
                found.append(f)
        else:
            print(f"  [MISSING] {v}")
            missing.append(v)

    extras = sorted(set(available) - set(VARIABLES))
    if extras:
        print(f"\nOther variables present in directory (not requested): {extras}")

    print(f"\nSummary: {len(found)} files to download, {len(missing)} variables missing")

    # %%
    prefix = f"{BUCKET}/{EXPERIMENT}/{CASE}/{OCN_SUBDIR}/"
    print(f"Listing: {prefix}")
    all_files = fs.ls(prefix)
    print(f"Found {len(all_files)} files in directory\n")

    available = {}
    for f in all_files:
        v = var_from_filename(f, "h")
        if v:
            available.setdefault(v, []).append(f)

    # Report
    print("=== Verification report ===")
    found, missing = [], []
    for v in OCN_VARIABLES:
        if v in available:
            for f in available[v]:
                print(f"  [OK]      {v:15s} -> {f.split('/')[-1]}")
                found.append(f)
        else:
            print(f"  [MISSING] {v}")
            missing.append(v)

    extras = sorted(set(available) - set(VARIABLES))
    if extras:
        print(f"\nOther variables present in directory (not requested): {extras}")

    print(f"\nSummary: {len(found)} files to download, {len(missing)} variables missing")

    # %%
    # # Download atm variables
    for case in cases:
        # Use the case name pattern from cell 3
        prefix = f"{BUCKET}/{EXPERIMENT}/{case}/{SUBDIR}/"

        print(f"\n--- Processing atm variables case {case} ---")

        try:
            all_files = fs.ls(prefix)
        except FileNotFoundError:
            print(f"Directory not found for case {case}, skipping.")
            continue

        # Identify files for this member
        available = {}
        for f in all_files:
            v = var_from_filename(f, "h0")
            if v:
                available.setdefault(v, []).append(f)

        # Collect exactly the requested variables
        found = []
        for v in VARIABLES:
            if v in available:
                for f in available[v]:
                    found.append(f)
        
        print(f"Found {len(found)} files to download for case {case}")
        
        # Download files
        for i, remote in enumerate(found, 1):
            local = LOCAL_DIR / Path(remote).relative_to("ncar-cesm2-arise/raw/")
            if local.exists():
                print(f"[{i}/{len(found)}] skip (exists): {local.name}")
                continue
            size_mb = fs.size(remote) / 1e6
            print(f"[{i}/{len(found)}] downloading {local.name} ({size_mb:.1f} MB)")
            fs.get(remote, str(local))

    # %%
    # Download ocn variables
    for case in cases: 
        # Use the case name pattern from cell 3
        ocn_prefix = f"{BUCKET}/{EXPERIMENT}/{case}/{OCN_SUBDIR}/"

        print(f"\n--- Processing ocn variables member {case} ---")

        try:
            all_files = fs.ls(ocn_prefix)
        except FileNotFoundError:
            print(f"Directory not found for case {case}, skipping.")
            continue

        # Identify files for this member
        available = {}
        for f in all_files:
            v = var_from_filename(f, "h")
            if v:
                available.setdefault(v, []).append(f)

        # Collect exactly the requested variables
        found = []
        for v in OCN_VARIABLES:
            if v in available:
                for f in available[v]:
                    found.append(f)
        
        print(f"Found {len(found)} files to download for case {case}")
        
        # Download files
        for i, remote in enumerate(found, 1):
            local = LOCAL_DIR / Path(remote).relative_to("ncar-cesm2-arise/raw/")
            if local.exists():
                print(f"[{i}/{len(found)}] skip (exists): {local.name}")
                continue
            size_mb = fs.size(remote) / 1e6
            print(f"[{i}/{len(found)}] downloading {local.name} ({size_mb:.1f} MB)")
            fs.get(remote, str(local))

    # %%