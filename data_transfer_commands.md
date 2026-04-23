List of variable strings used in the data transfer commands.
.*HFLX. (gets both LHFLX and SHFLX)
.TS.
.CLDTOT.
.FLN*.
.FSN*.
.FLUT*.

If I open a connection with -M it should allow me to reuse the SSH connection for multiple file transfers, so I can run multiple commands in a row without having to re-enter my password each time. For example:
casp
ssh -M josh2250@login.rc.colorado.edu

List of command line entries used to move data between glade/gdex and CURC.

__ARISE-SAI-1.5 examples:__
find /gdex/data/d651059/ARISE-SAI-1.5 -wholename "*month_1**.FLNR.*"

./job_scripts/gdex_file_transfer.sh "*month_1**.FLNR.*" /gdex/data/d651059/ARISE-SAI-1.5 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/ARISE_SAI/d651059/ARISE-SAI-1.5/

./job_scripts/gdex_file_transfer.sh "*month_1*.TS.*" /gdex/data/d651059/ARISE-SAI-1.5 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/ARISE_SAI/d651059/ARISE-SAI-1.5/

__CESM-CAM5-LME examples:__

find /gdex/data/d651058/CESM-CAM5-LME -wholename "*monthly**BLMTRC5CN.f19_g16.002**.TS.*"

./job_scripts/gdex_file_transfer.sh "*monthly**BLMTRC5CN.f19_g16.002*.TS.*" /gdex/data/d651058/CESM-CAM5-LME josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM_LME/d651058/CESM-CAM5-LME/

*Volcanic forcing only run:*
./job_scripts/gdex_file_transfer.sh "*monthly**BLMTRC5CN.f19_g16.VOLC_GRA.001.cam*.CLDTOT.*" /gdex/data/d651058/CESM-CAM5-LME josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM_LME/d651058/CESM-CAM5-LME/

*GHG forcing only run:*
find /gdex/data/d651058/CESM-CAM5-LME -wholename "*monthly**B1850C5CN.f19_g16.LME.GHG.001**.TS.*"

./job_scripts/gdex_file_transfer.sh "*monthly**B1850C5CN.f19_g16.LME.GHG.001*.TS.*" /gdex/data/d651058/CESM-CAM5-LME josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM_LME/d651058/CESM-CAM5-LME/


*Aerosol + Ozone forcing only run:*
find /gdex/data/d651058/CESM-CAM5-LME -wholename "*monthly**LME.O3AER.001**.TS.*"

./job_scripts/gdex_file_transfer.sh "*monthly**B1850C5CN.f19_g16.LME.O3AER.001*.TS.*" /gdex/data/d651058/CESM-CAM5-LME josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM_LME/d651058/CESM-CAM5-LME/

__SSP245 examples:__
find /gdex/data/d651045/CESM2-WACCM-SSP245 -wholename "*month_1/**.TS.*"

./job_scripts/gdex_file_transfer.sh "*month_1/**.TS.*" /gdex/data/d651045/CESM2-WACCM-SSP245 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_WACCM_SSP2-4.5/d651045/CESM2-WACCM-SSP245/

__MCB SSP245 examples:__
find /gdex/data/d314006 -wholename "*month_1/**.TS.*"

./job_scripts/gdex_file_transfer.sh "*month_1/**.TS.*" /gdex/data/d314006 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_WACCM_SSP2-4.5_MCB/d314006/

__CESM2_LME examples:__
*Initially just the past1000 run*
find /gdex/data/d651078/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/ -wholename "*month_1/**.TS.*"

./job_scripts/gdex_file_transfer.sh "*month_1/**.TS.*" /gdex/data/d651078/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_LME/d651078/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002/

*Now including the control run:*
find /gdex/data/d651078/ -wholename "*month_1/**.TS.*"

./job_scripts/gdex_file_transfer.sh "*month_1/**.TS.*" /gdex/data/d651078 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_LME/d651078/

find /gdex/data/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/ -wholename "*month_1/**.TS.*"

./job_scripts/gdex_file_transfer.sh "*month_1/**.TS.*" /gdex/data/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_LME/d651078/b.e21.BWma1850.f19_g17.PMIP4-PaleoStrat.850CEcontrol.008/

__CESM2-LE examples:__
*I will try to get a few ensemble members from both the original and SMBB simulations before branching to SSP.*
*This gets too many because the BHISTcmip6 members have different start years that reset the ensemble member index.
find /gdex/data/d651056/CESM2-LE -wholename "*month_1/**BHIST*00?.cam.h0.TS.*"

Just get the BHISTsmbb members for now:
find /gdex/data/d651056/CESM2-LE -wholename "*month_1/**BHISTsmbb*00?.cam.h0.TS.*"

Just get the BHISTcmip6 members branched from 1301 for now:
find /gdex/data/d651056/CESM2-LE -wholename "*month_1/**BHISTcmip6.f09_g17.LE2-1301*00?.cam.h0.TS.*"
BHISTcmip6.f09_g17.LE2-1301

*All simulations with member index 00?:*
./job_scripts/gdex_file_transfer.sh "*month_1/**BHIST*00?.cam.h0.TS.*" /gdex/data/d651056/CESM2-LE josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_LE/d651056/CESM2-LE/

*Just the BHISTsmbb members with member index 00?:*
./job_scripts/gdex_file_transfer.sh "*month_1/**BHISTsmbb*00?.cam.h0.TS.*" /gdex/data/d651056/CESM2-LE josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_LE/d651056/CESM2-LE/

*Just the BHISTcmip6 members with member index 00? branched from 1301:*
./job_scripts/gdex_file_transfer.sh "*month_1/**BHISTcmip6.f09_g17.LE2-1301*00?.cam.h0.TS.*" /gdex/data/d651056/CESM2-LE josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_LE/d651056/CESM2-LE/

__CESM2 1850control examples:__

find /glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.B1850.f09_g17.CMIP6-piControl.001 -wholename "*month_1/**b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.TS.*"

./job_scripts/gdex_file_transfer.sh "*month_1/**b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.TS.*" /glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.B1850.f09_g17.CMIP6-piControl.001 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_1850control/b.e21.B1850.f09_g17.CMIP6-piControl.001/


