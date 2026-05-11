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

FILE=../zipped_data/CESM2_WACCM_1850control.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM2_WACCM_1850control
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_HIST.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} RadInt_procdata/CESM2_WACCM_HIST
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

FILE=../zipped_data/control_baselines_ohc.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} control_baselines_ohc
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

FILE=../zipped_data/CESM2_WACCM_1850control_error_components.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} error_relativetobaseline/CESM2_WACCM_1850control
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_HIST_1850_1864_error_components.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} error_relativetobaseline/CESM2_WACCM_HIST_1850_1864
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_HIST_2000_2014_error_components.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} error_relativetobaseline/CESM2_WACCM_HIST_2000_2014
else
    echo "${FILE} already exists"
fi

# Spatial map data
# FILE=../zipped_data/CESM2_WACCM_1850control_spatialdata.zip
# if [ ! -e ${FILE} ]; then 
#     zip -r ${FILE} spatial_maps/CESM2_WACCM_1850control
# else
#     echo "${FILE} already exists"
# fi

FILE=../zipped_data/CESM2_WACCM_1850control_spatialdata1.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps/CESM2_WACCM_1850control/*.FL*.*
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_1850control_spatialdata3.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps/CESM2_WACCM_1850control/*.FS*.*
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_1850control_spatialdata2.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps/CESM2_WACCM_1850control/b.e21.BW1850.f09_g17.CMIP6-piControl.001.CLDTOT.spatial_uncertainty.nc spatial_maps/CESM2_WACCM_1850control/b.e21.BW1850.f09_g17.CMIP6-piControl.001.LHFLX.spatial_uncertainty.nc spatial_maps/CESM2_WACCM_1850control/b.e21.BW1850.f09_g17.CMIP6-piControl.001.PRECIP_THERMO.spatial_uncertainty.nc spatial_maps/CESM2_WACCM_1850control/b.e21.BW1850.f09_g17.CMIP6-piControl.001.PRECT.spatial_uncertainty.nc spatial_maps/CESM2_WACCM_1850control/b.e21.BW1850.f09_g17.CMIP6-piControl.001.SHFLX.spatial_uncertainty.nc spatial_maps/CESM2_WACCM_1850control/b.e21.BW1850.f09_g17.CMIP6-piControl.001.TS.spatial_uncertainty.nc
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_1850control_spatialdata_ohc.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps_ohc/CESM2_WACCM_1850control
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_SSP2-4.5_spatialdata.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps/CESM2_WACCM_SSP2-4.5
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_SSP2-4.5_spatialdata_ohc.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps_ohc/CESM2_WACCM_SSP2-4.5
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/ARISE-SAI_spatialdata.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps/ARISE-SAI
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/ARISE-SAI_spatialdata_ohc.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps_ohc/ARISE-SAI
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_SSP2-4.5_MCB_spatialdata.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps/CESM2_WACCM_SSP2-4.5_MCB
else
    echo "${FILE} already exists"
fi

FILE=../zipped_data/CESM2_WACCM_SSP2-4.5_MCB_spatialdata_ohc.zip
if [ ! -e ${FILE} ]; then 
    zip -r ${FILE} spatial_maps_ohc/CESM2_WACCM_SSP2-4.5_MCB
else
    echo "${FILE} already exists"
fi