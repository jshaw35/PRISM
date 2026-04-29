#!/bin/bash

cd /glade/u/home/jonahshaw/Scripts/git_repos/PRISM
# Process global/hemispheric means
unzip -n zipped_data/CESM_LME_data.zip -d data/
unzip -n zipped_data/CESM2_LME_data.zip -d data/
unzip -n zipped_data/CESM2_LE_data.zip -d data/
unzip -n zipped_data/CESM2_SSP245_data.zip -d data/
unzip -n zipped_data/CESM2_SSP245_ARISE_data.zip -d data/
unzip -n zipped_data/CESM2_SSP245_MCB_data.zip -d data/
unzip -n zipped_data/CESM2_1850control_data.zip -d data/

# Control baselines
unzip -n zipped_data/control_baselines.zip -d data

# NMSE error components
unzip -n zipped_data/CESM2_LME_control_error_components.zip -d data/
unzip -n zipped_data/CESM2_1850control_error_components.zip -d data/
unzip -n zipped_data/CESM2_LE_2000_2009_smbb_error_components.zip -d data/
unzip -n zipped_data/CESM2_LE_2000_2009_cmip6_error_components.zip -d data/
