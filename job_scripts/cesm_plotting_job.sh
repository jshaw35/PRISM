#!/bin/bash -l
#PBS -N cesm_plotting
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=8GB:ngpus=0
#PBS -l walltime=0:59:59
#PBS -q casper
#PBS -j oe

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel
cd /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs
mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J03_CESM_OLR_ASR_plots_multiensemble.py

mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J04_CESM_EEI_timeseries_multiensemble.py