#!/bin/bash -l
#PBS -N ohc_compute
#PBS -A UCUC0007
#PBS -l select=1:ncpus=1:mem=8GB:ngpus=0
#PBS -l walltime=0:05:00
#PBS -q casper
#PBS -j oe
# #PBS -o /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel
cd /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs
conda activate /glade/work/jonahshaw/conda-envs/hackathon_extended
python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J09_OHC_calculation_parallel.py
