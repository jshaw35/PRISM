#!/bin/bash

# Simple script to find files matching a pattern and rsync them to a remote server
# Usage: ./sync_files.sh [--dry-run] <pattern> <source_dir> <remote_spec>
# Example: ./sync_files.sh "*month_1*.TS.*" /gdex/data/d651059/ARISE-SAI-1.5 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/ARISE_SAI/d651059/ARISE-SAI-1.5/
# ^For correct directory level matching, make sure the source and target paths match up to the point where the pattern starts (e.g. /gdex/data/d651059/ARISE-SAI-1.5/).

# Can look for files using find with -wholename to match the full path before executing this scripts.
# e.g. find /gdex/data/d651045/CESM2-WACCM-SSP245/ -wholename "*month_1/**.TS.*"

dry_run=false

# Check for --dry-run flag
if [[ "$1" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

# Validate arguments
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 [--dry-run] <pattern> <source_dir> <remote_spec>"
    echo "Example: $0 \"*month_1*.TS.*\" /gdex/data/d651059/ARISE-SAI-1.5 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/ARISE-SAI-1.5/"
    exit 1
fi

pattern="$1"
source_dir="$2"
remote_spec="$3"

# Validate source directory exists
if [[ ! -d "$source_dir" ]]; then
    echo "Error: Source directory does not exist: $source_dir"
    exit 1
fi

# Build rsync options
rsync_opts="-avz"
if [[ "$dry_run" == true ]]; then
    rsync_opts="$rsync_opts --dry-run"
fi

# Find matching files and rsync them in a single batch (login once)
files=$(find "$source_dir" -wholename "$pattern")
if [[ -z "$files" ]]; then
    echo "No files found matching pattern: $pattern"
    exit 0
fi

echo "Found $(echo "$files" | wc -l) files. Starting batch transfer..."

# Generate file list with relative paths and pass to rsync via stdin
if echo "$files" | sed "s|^$source_dir/||" | rsync $rsync_opts --files-from=- "$source_dir/" "$remote_spec"; then
    echo ""
    echo "Summary: Batch transfer completed successfully"
else
    echo ""
    echo "Summary: Batch transfer failed"
    exit 1
fi
