# PRISM
A radiative perspective on climate intervention safety (ARPOCIS/PRISM)

**2026/07/13:**
Porting code fully to NCAR HPC (Casper) and providing more detailed documentation.

### Order of operations:
**Processing**
1. Download ARISE-1.0 data using J13_download_ARISE1.0.py. If you are operating on a system other than Casper/Derecho, you will need to transfer other data to your platform. The script "gdex_file_transfer.sh" can be modified and used for this purpose.
2. Create new analysis variables by running job_scripts/compute_PRECIP_THERMO_job.sh to execute J07_compute_THERMO_PRECIP.py
3. Compute processed ocean fields (Ocean Heat Content and integrated Ocean Heat Flux) using job_scripts/compute_ohc_parallel.sh and job_scripts/compute_oceanflux_parallel.sh (J09_OHC_calculation_parallel.py, J09_oceanflux_calculation_parallel.py, J10_OHC_calculation_single.py, J10_oceanflux_calculation_single.py)
4. Compute global and hemispheric area-weighted averages of the atmosphere fields using job_scripts/spatial_averaging_job.sh (J02_CESM_spatialaveraging.py)
5. Compute the background states (used to construct confidence intervals) of energetic variables from CESM control simulations using control_background_job.sh (J05_CESM_compute_background.py)
6. Compute the NMSE relative to the background and decompose its components using compute_error_job.sh (J06_CESM_compute_bias.py)
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
J06_CESM_compute_bias.py: compute_error_job.sh
J07_compute_THERMO_PRECIP.py: compute_PRECIP_THERMO_job.sh
J11_preprocess_spatialmaps.py: preprocess_spatialmaps.sh

If on glade and moving data to a new location, use: job_scripts/gdex_file_transfer.sh

If on CURC and processing data, run: sh job_scripts/spatial_averaging_job.sh
If on CURC and compressing processed data, run: sh job_scripts/compress_data.sh

If trying to unpack processed data, use: job_scripts/unpack_data.sh

