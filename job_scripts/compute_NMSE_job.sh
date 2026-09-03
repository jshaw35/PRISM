#!/bin/bash -l
#PBS -N compute_error
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=0
#PBS -l walltime=1:59:59
#PBS -q casper
#PBS -o /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/
#PBS -e /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel
cd /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/daskworkers
echo "Job started at $(date)"
mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J06_CESM_compute_NMSE_parallel.py

echo "Job completed successfully at $(date)"