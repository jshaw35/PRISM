#!/bin/bash

# Because this is a job array, each job will request resources independently
# This means each job will request N ntasks, on N nodes with N cpus-per-task

# %A: Job ID
# %a: Array Task ID
# ----------------------------------------------------------
# #SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=00:59:59   # Request 23 hours and 59 minutes for longer computation
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=spatial_average_CESM_data
#SBATCH --output=spatial_average_CESM_data_%A_%a.out
#SBATCH --error=spatial_average_CESM_data_%A_%a.err
#SBATCH --mail-user=josh2250@colorado.edu
#SBATCH --mail-type=ALL
# #SBATCH --array=101-173    # 73 measurements from the ensemble_profiles to process

ml anaconda
conda activate /curc/sw/anaconda3/2023.09/envs/ATOC_NWP
python /projects/josh2250/PRISM/J02_CESM_spatialaveraging.py