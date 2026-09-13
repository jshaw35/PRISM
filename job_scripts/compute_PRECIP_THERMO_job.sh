#!/bin/bash -l
#PBS -N compute_precip_thermo
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=0
#PBS -l walltime=2:59:59
#PBS -q casper
#PBS -o /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/
#PBS -e /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel
cd /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs
echo "Job started at $(date)"

mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J07_CESM_compute_THERMO_PRECIP_parallel.py

echo "Job completed successfully at $(date)"