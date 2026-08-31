#!/bin/bash -l
#PBS -N ohc_compute
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=0
#PBS -l walltime=0:40:00
#PBS -q casper
#PBS -o /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/
#PBS -e /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel

echo $PBS_JOBID

if [[ -z "$FILE_PATH" ]] || [[ -z "$SAVE_PATH" ]]; then
    echo "ERROR: missing variables FILE_PATH or SAVE_PATH"
    echo ""
    echo "Usage: qsub -v FILE_PATH=/path/to/input,SAVE_PATH=/path/to/output $0"
    exit 2
fi

echo "Job started at $(date)"
mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J14_CESM_compute_newvars_single.py "$FILE_PATH" "$SAVE_PATH"

echo "Job completed successfully at $(date)"