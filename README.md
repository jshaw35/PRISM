# PRISM
A radiative perspective on climate intervention safety (ARPOCIS/PRISM)

**2026/09/13**
Started to consolidate commonly used functions to avoid inconsistent practices
Also starting to update J08_CESM_bias_decomposition. Files are now loading, but need to think about the figure format with the additional data complexity.
- Currently the time series of the NMSE show the departure from piControl conditions, did we want to compare with the pre-geoengineering period instead?
- Each experiment has multiple realizations, which would make showing the departure from "precedent" messy. Do we instead show the mean value and scatter over the period 2060 - 2069 relative to whatever our control(s) are? So each panel could be a different variable, where the x-axis indexes over the ~12 experiments, and the y-axis shows both the null hypothesis range and the results from individual members and the ensemble mean. (use similar graphics with dot size, alpha, etc from the ASR-OLR plots)
- But this would hide the delta-function forcing of the MCB experiments (but we could leave this in the supplement and a future paper)

**2026/09/10**
Reworked time series figure with all CESM2 experiments is fully drafted in J04_CESM_EEI_timeseries_multiensemble.py.

**2026/09/07**
Created a new file J03_CESM_OLR_ASR_plots_multiensemble.py to handle all of the data dimensions more completely.

**2026/09/06**
Finished updating J11_preprocess_spatialmaps.py and pushed.

**2026/09/03**
- Updating J11_preprocess_spatialmaps.py. This works quite well already, but it can only compare with the PIcontrol. I am adding functionality to also compare with other periods. e.g. 2014 - 2034. Not done yet.


**2026/09/02**
- Updating J06_CESM_compute_bias.py. This is now working but a parallelized approach would be way way faster. I parallelized it and now it is quite fast!

**2026/09/01**
- Corrected spatial averaging script to read files from multiple paths (to handle the derived variables)
- Started updating j05_compute_background to also read from multiple paths. Now done. These files will need to be saved separately unless I shift to zarr, thus requiring later scripts to be updated to use the new file conventions.

**2026/08/31**
- Parallelized J07 which is now replaced with J14 files.

**2026/08/29**
- Ran into issues running the OHC and OHF scripts on ARISE1.0 data. Because the TAREA is masked in land regions instead of being assigned to every point. To avoid this, I do not produce a global surface area variable when nans are present in this field. The values themselves look good. Raises more questions about energy closure.
- Ported the spatial averaging code to glade. But I think that this should be parallelized over datasets along with the derived variables code J07. Otherwise they take hours to run because it is done sequentially. The spatial averaging actually runs in just ~35 minutes but deriving new variables takes hours.

**2026/08/28**
Resuming porting to glade.
- figured out how to get log files to specific directories (-o and -e PBS settings)
- Tested and commited ARISE1.0 download script on glade

- Currently running compute_PRECIP_THERMO_job.sh on glade (check job 5746007). **This port will be useful for the later scripts because the file was initially written for CURC.**
- Currently running compute_ohc_parallel.sh and compute_oceanflux_parallel.sh on glade (jobs 5746043.casper-pbs and 5746044.casper-pbs). Check on return. These appear to have died from wallclock so I will increase the time. Fixed on 08/29

**2026/07/13:**
Porting code fully to NCAR HPC (Casper) and providing more detailed documentation.

### Order of operations:
**Processing**
1. Download ARISE-1.0 data using download_ARISE_1.0_job.sh (J13_download_ARISE1.0.py). If you are operating on a system other than Casper/Derecho, you will need to transfer other data to your platform. The script "gdex_file_transfer.sh" can be modified and used for this purpose.
2. Create new analysis variables by running job_scripts/compute_PRECIP_THERMO_job.sh to execute J07_compute_THERMO_PRECIP.py. This has now been superceded by job_scripts/compute_newvar_parallel.sh.
3. Compute processed ocean fields (Ocean Heat Content and integrated Ocean Heat Flux) using job_scripts/compute_ohc_parallel.sh and job_scripts/compute_oceanflux_parallel.sh (J09_OHC_calculation_parallel.py, J09_oceanflux_calculation_parallel.py, J10_OHC_calculation_single.py, J10_oceanflux_calculation_single.py)
4. Compute global and hemispheric area-weighted averages of the atmosphere fields using job_scripts/spatial_averaging_job.sh (J02_CESM_spatialaveraging.py)
5. Compute the background states (used to construct confidence intervals) of energetic variables from CESM control simulations using control_background_job.sh (J05_CESM_compute_background.py)
6. Compute the NMSE relative to the background and decompose its components using compute_error_job.sh (J06_CESM_compute_bias.py). Replaced by J06_CESM_compute_NMSE_parallel.py.
7. Compute significance of the "spatial" changes with respect to the piControl simulation (J11_process_spatialmaps.py)

**Plotting:**
1. Produce ASR-OLR plots showing the evolution and variability of EEI using cesm_plotting_job.sh (J03_CESM_OLR_ASR_plots.py)
2. Produce time series plots of OLR, ASR, EEI, iEEI, OHC, OHF with J04_CESM_EEI_timeseries3.py
3. Produce NMSE time series plots with J08_CESM_bias_decomposition.py (this may need to be cleaned up a lot)
4. Produce spatial maps of change with significance shading (J12_plot_spatialmaps.py)
5. Additional tests of energy conservation in Jxx_testconservation.py

List of python scripts and bash scripts that trigger them (so I can put them in order later):
J02_CESM_spatialaveraging.py: spatial_averaging_job.sh
J03_CESM_OLR_ASR_plots.py: cesm_plotting_job.sh
J04_CESM_EEI_timeseries3.py: no bash script but running slow recently
J05_CESM_compute_background.py: control_background_job.sh
J06_CESM_compute_bias.py: compute_error_job.sh (replaced by J06_CESM_compute_NMSE_parallel.py)
J07_compute_THERMO_PRECIP.py: compute_PRECIP_THERMO_job.sh (replaced by J13_CESM_compute_newvars_parallel.py)
J11_preprocess_spatialmaps.py: preprocess_spatialmaps.sh

If on glade and moving data to a new location, use: job_scripts/gdex_file_transfer.sh

If on CURC and processing data, run: sh job_scripts/spatial_averaging_job.sh
If on CURC and compressing processed data, run: sh job_scripts/compress_data.sh

If trying to unpack processed data, use: job_scripts/unpack_data.sh

