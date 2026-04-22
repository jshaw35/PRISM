#!/bin/bash

cd /home/josh2250/projects/PRISM/data

zip -r ../zipped_data/CESM2_SSP245_ARISE_data.zip RadInt_procdata/ARISE_SAI

zip -r ../zipped_data/CESM_LME_data.zip RadInt_procdata/CESM_LME

zip -r ../zipped_data/CESM2_LE_data.zip RadInt_procdata/CESM2_LE

zip -r ../zipped_data/CESM2_LME_data.zip RadInt_procdata/CESM2_LME

zip -r ../zipped_data/CESM2_SSP245_data.zip RadInt_procdata/CESM2_WACCM_SSP2-4.5

zip -r ../zipped_data/CESM2_SSP245_MCB_data.zip RadInt_procdata/CESM2_WACCM_SSP2-4.5_MCB

zip -r ../zipped_data/CESM2_1850control_data.zip RadInt_procdata/CESM2_1850control
