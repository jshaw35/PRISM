#!/bin/bash

# %A: Job ID
# %a: Array Task ID
# ----------------------------------------------------------
# #SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=00:14:59
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=compress_data
#SBATCH --output=/projects/josh2250/PRISM/job_logs/compress_data_%A_%a.out
#SBATCH --error=/projects/josh2250/PRISM/job_logs/compress_data_%A_%a.err
#SBATCH --mail-user=josh2250@colorado.edu
#SBATCH --mail-type=ALL
# #SBATCH --array=101-173    # 73 measurements from the ensemble_profiles to process

cd /home/josh2250/projects/PRISM/data

# Global+hemispheric means
FILE=../zipped_data/CESM2_SSP245_ARISE_data.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/ARISE_SAI
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM_LME_data.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM_LME
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_LE_data.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM2_LE
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_LME_data.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM2_LME
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_SSP245_data.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM2_WACCM_SSP2-4.5
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_SSP245_MCB_data.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM2_WACCM_SSP2-4.5_MCB
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_1850control_data.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM2_1850control
else
    echo "${FILE} already exists"
fi

# Baseline fields
FILE=../zipped_data/control_baselines.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} control_baselines
else
    echo "${FILE} already exists"
fi

# NMSE components
FILE=../zipped_data/CESM2_1850control_error_components.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} error_relativetobaseline/CESM2_1850control
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_LE_2000_2009_cmip6_error_components.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} error_relativetobaseline/CESM2_LE_2000_2009_cmip6
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_LE_2000_2009_smbb_error_components.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} error_relativetobaseline/CESM2_LE_2000_2009_smbb
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_LME_control_error_components.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} error_relativetobaseline/CESM2_LME_control
else
    echo "${FILE} already exists"
fi