#!/bin/bash -l
#PBS -N spatial_averaging
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=0
#PBS -l walltime=1:59:59
#PBS -q casper
#PBS -j oe

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel
cd /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs
mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J13_download_ARISE1.0.py
