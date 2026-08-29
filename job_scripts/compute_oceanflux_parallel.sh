#!/bin/bash -l
#PBS -N ohf_compute
#PBS -A UCUC0007
#PBS -l select=1:ncpus=1:mem=8GB:ngpus=0
#PBS -l walltime=0:15:00
#PBS -q casper
#PBS -o /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/
#PBS -e /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel
cd /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs
echo "Job started at $(date)"

mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J09_oceanflux_calculation_parallel.py

echo "Job completed successfully at $(date)"