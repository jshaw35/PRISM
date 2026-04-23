#!/bin/bash

# This means each job will request N ntasks, on N nodes with N cpus-per-task

# %A: Job ID
# %a: Array Task ID
# ----------------------------------------------------------
# #SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=00:59:59
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=control_background_CESM
#SBATCH --output=/projects/josh2250/PRISM/job_logs/control_background_%A_%a.out
#SBATCH --error=/projects/josh2250/PRISM/job_logs/control_background_%A_%a.err
#SBATCH --mail-user=josh2250@colorado.edu
#SBATCH --mail-type=ALL
# #SBATCH --array=101-173    # 73 measurements from the ensemble_profiles to process

ml anaconda
conda activate /curc/sw/anaconda3/2023.09/envs/ATOC_NWP
python /projects/josh2250/PRISM/J05_CESM_compute_background.py

# Submit with sbatch