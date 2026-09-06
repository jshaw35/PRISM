#!/bin/bash -l
#PBS -N preprocess_spatialmaps
#PBS -A UCUC0007
#PBS -l select=1:ncpus=4:mem=128GB:ngpus=0
#PBS -l walltime=1:59:59
#PBS -q casper
#PBS -o /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/
#PBS -e /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/job_logs/

# Kill all casper jobs with: qselect -u jonahshaw | xargs qdel
echo "Job started at $(date)"
mamba run -p /glade/work/jonahshaw/conda-envs/hackathon_extended \
    python /glade/u/home/jonahshaw/Scripts/git_repos/PRISM/J11_preprocess_spatialmaps.py

echo "Job completed successfully at $(date)"