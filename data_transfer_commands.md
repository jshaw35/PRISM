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


__SSP245 examples:__
find /gdex/data/d651045/CESM2-WACCM-SSP245 -wholename "*month_1/**.TS.*"

./job_scripts/gdex_file_transfer.sh "*month_1/**.TS.*" /gdex/data/d651045/CESM2-WACCM-SSP245 josh2250@login.rc.colorado.edu:/home/josh2250/kaydata/jshaw/RadInt_rawdata/CESM2_WACCM_SSP2-4.5/d651045/CESM2-WACCM-SSP245/