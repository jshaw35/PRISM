#!/usr/bin/env python3
"""
CESM2 Ocean Heat Content (OHC) Calculator

Computes global ocean heat content from CESM2/POP2 monthly output using full
equation of state (pre-calculated density from model).

Author: OpenCode
Date: 2026-04-30
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


class CESM2OHCCalculator:
    """
    Calculate Ocean Heat Content from CESM2 POP2 output.
    
    Processes monthly data in chunks to minimize memory usage on login nodes.
    Uses pre-calculated RHO fields for accurate equation of state treatment.
    """
    
    # Physical constants
    CP_SEAWATER = 3850.0  # J/(kg·K) - specific heat capacity of seawater
    
    def __init__(self, temp_file: str, rho_file: str = None, verbose: bool = True):
        """
        Initialize calculator with CESM2 temperature and density files.
        
        Parameters
        ----------
        temp_file : str
            Path to CESM2 monthly TEMP file (contains TEMP, dz, TAREA, KMT)
        rho_file : str, optional
            Path to CESM2 RHO file. If None, infer from temp_file path.
        verbose : bool
            Print progress information
        """
        self.temp_file = Path(temp_file)
        self.verbose = verbose
        self._validate_file_exists()
        
        # Infer RHO file if not provided
        if rho_file is None:
            # Replace TEMP with RHO in filename
            self.rho_file = Path(str(self.temp_file).replace('.TEMP.', '.RHO.'))
        else:
            self.rho_file = Path(rho_file)
        
        if not self.rho_file.exists():
            raise FileNotFoundError(f"RHO file not found: {self.rho_file}")
        if self.verbose:
            print(f"✓ RHO file found: {self.rho_file.name}")
        
        # Will be loaded on demand
        self._grid_loaded = False
        self.dz = None
        self.tarea = None
        self.kmt = None
        
    def _validate_file_exists(self):
        """Check that input file exists."""
        if not self.temp_file.exists():
            raise FileNotFoundError(f"Temperature file not found: {self.temp_file}")
        if self.verbose:
            print(f"✓ Input file found: {self.temp_file.name}")
    
    def _load_grid_info(self):
        """Load time-independent grid information from TEMP file."""
        if self._grid_loaded:
            return
        
        if self.verbose:
            print("Loading grid information...")
        
        with xr.open_dataset(self.temp_file, decode_times=True) as ds:
            # Convert dz from cm to m
            self.dz = np.array(ds['dz'].values) / 100.0  # [m]
            
            # Convert TAREA from cm² to m²
            self.tarea = np.array(ds['TAREA'].values) / 1e4  # [m²]
            
            # Ocean mask (0=land, >0=ocean)
            self.kmt = np.array(ds['KMT'].values)
            
            # Store grid dimensions
            self.nlat = ds.dims['nlat']
            self.nlon = ds.dims['nlon']
            self.nz = ds.dims['z_t']
            self.ntime = ds.dims['time']
        
        self._grid_loaded = True
        if self.verbose:
            print(f"  Grid shape: {self.nlat} × {self.nlon} × {self.nz}")
            print(f"  Total timesteps: {self.ntime}")
    
    def _get_reference_climatology(self, chunk_size: int = 12) -> np.ndarray:
        """
        Compute mean temperature from first 10 years (months 0-119).
        
        Loads reference period in chunks to minimize memory.
        
        Parameters
        ----------
        chunk_size : int
            Number of months per chunk (default 12 = 1 year)
            
        Returns
        -------
        temp_ref : ndarray
            Mean temperature over reference period [K], shape (nz, nlat, nlon)
        """
        if self.verbose:
            print("Computing reference climatology (first 10 years)...")
        
        ref_end = 120  # First 10 years = 120 months
        temp_sum = None
        
        for start_idx in range(0, ref_end, chunk_size):
            end_idx = min(start_idx + chunk_size, ref_end)
            
            with xr.open_dataset(self.temp_file, decode_times=True) as ds:
                chunk = ds['TEMP'].isel(time=slice(start_idx, end_idx)).values
                # Shape: (chunk_months, nz, nlat, nlon)
                
                if temp_sum is None:
                    temp_sum = np.sum(chunk, axis=0)
                else:
                    temp_sum += np.sum(chunk, axis=0)
        
        temp_ref = temp_sum / ref_end  # Average
        
        if self.verbose:
            print(f"  Reference mean temp: {np.nanmean(temp_ref):.2f}°C")
        
        return temp_ref
    
    def compute_global_ohc_timeseries(
        self,
        output_file: Optional[str] = None,
        chunk_size: int = 12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute monthly global ocean heat content time series.
        
        OHC formula: OHC = Σ(ρ × c_p × ΔT × dz)
        
        Integrates over:
        - All vertical levels (full ocean depth)
        - All ocean grid cells
        
        Parameters
        ----------
        output_file : str, optional
            Path to save output NetCDF. If None, only returns arrays.
        chunk_size : int
            Number of months to process per iteration (default 12)
            
        Returns
        -------
        time : ndarray
            Time array [days since 0000-01-01]
        ohc_global : ndarray
            Global mean OHC time series [J/m²], shape (ntime,)
        """
        self._load_grid_info()
        
        # Compute reference climatology
        temp_ref = self._get_reference_climatology(chunk_size=chunk_size)
        
        if self.verbose:
            print(f"\nProcessing {self.ntime} months in chunks of {chunk_size}...")
        
        ohc_timeseries = []
        time_coords = None
        
        # Process full time series in chunks
        for start_idx in range(0, self.ntime, chunk_size):
            end_idx = min(start_idx + chunk_size, self.ntime)
            chunk_len = end_idx - start_idx
            
            # Load TEMP and time from TEMP file
            with xr.open_dataset(self.temp_file, decode_times=True) as ds:
                temp_chunk = ds['TEMP'].isel(time=slice(start_idx, end_idx)).values
                time_chunk = ds['time'].isel(time=slice(start_idx, end_idx)).values
            
            # Load RHO from RHO file
            with xr.open_dataset(self.rho_file, decode_times=True) as ds:
                rho_chunk = ds['RHO'].isel(time=slice(start_idx, end_idx)).values
            
            # Store time coordinates from first chunk
            if time_coords is None:
                time_coords = time_chunk
            else:
                time_coords = np.concatenate([time_coords, time_chunk])
            
            # Process each month in chunk
            for i_month in range(chunk_len):
                # Extract single month
                temp = temp_chunk[i_month, :, :, :]  # (nz, nlat, nlon)
                rho = rho_chunk[i_month, :, :, :]    # (nz, nlat, nlon) [g/cm³]
                
                # Calculate temperature anomaly
                delta_t = temp - temp_ref  # [K]
                
                # Convert RHO from g/cm³ to kg/m³
                rho_kg_m3 = rho * 1000.0  # [kg/m³]
                
                # Calculate OHC per unit volume: ρ × c_p × ΔT
                # Shape: (nz, nlat, nlon) [J/(m³·K)]
                ohc_per_vol = rho_kg_m3 * self.CP_SEAWATER * delta_t
                
                # Vertical integration: multiply by dz and sum
                # dz shape: (nz,), expand to (nz, nlat, nlon)
                dz_expanded = self.dz[:, np.newaxis, np.newaxis]
                ohc_per_area = (ohc_per_vol * dz_expanded).sum(axis=0)  # [J/m²]
                
                # Apply ocean mask (KMT > 0 = ocean)
                ohc_masked = ohc_per_area.copy()
                ohc_masked[self.kmt == 0] = np.nan
                
                # Global mean
                ohc_global_month = np.nanmean(ohc_masked)
                ohc_timeseries.append(ohc_global_month)
            
            if self.verbose:
                pct_done = (end_idx / self.ntime) * 100
                print(f"  Processed {end_idx}/{self.ntime} months ({pct_done:.1f}%)")
        
        ohc_timeseries = np.array(ohc_timeseries)
        
        if self.verbose:
            print(f"\n✓ OHC time series computed!")
            print(f"  Mean OHC: {np.mean(ohc_timeseries):.2e} J/m²")
            print(f"  OHC range: [{np.min(ohc_timeseries):.2e}, {np.max(ohc_timeseries):.2e}] J/m²")
        
        # Save to NetCDF if requested
        if output_file is not None:
            self._save_output(time_coords, ohc_timeseries, output_file)
        
        return time_coords, ohc_timeseries
    
    def _save_output(
        self,
        time: np.ndarray,
        ohc_global: np.ndarray,
        output_file: str
    ):
        """
        Save OHC time series to NetCDF file.
        
        Parameters
        ----------
        time : ndarray
            Time coordinates [days since 0000-01-01]
        ohc_global : ndarray
            Global OHC time series [J/m²]
        output_file : str
            Output file path
        """
        if self.verbose:
            print(f"\nWriting output to {Path(output_file).name}...")
        
        # Create output dataset
        ds_out = xr.Dataset(
            {
                'OHC_global': (['time'], ohc_global),
            },
            coords={
                'time': (['time'], time),
            }
        )
        
        # Add metadata
        ds_out.attrs['title'] = 'CESM2 Global Ocean Heat Content'
        ds_out.attrs['method'] = 'OHC = Σ(ρ × c_p × ΔT × dz)'
        ds_out.attrs['reference_period'] = 'First 10 years (months 0-119)'
        ds_out.attrs['depth_integration'] = 'Full ocean depth (60 levels)'
        ds_out.attrs['equation_of_state'] = 'Full (using model RHO field)'
        ds_out.attrs['source_file'] = str(self.temp_file)
        
        ds_out['OHC_global'].attrs['long_name'] = 'Global mean ocean heat content'
        ds_out['OHC_global'].attrs['units'] = 'J/m²'
        ds_out['OHC_global'].attrs['description'] = 'Temperature anomaly integrated over full ocean depth'
        
        # Save with compression
        ds_out.to_netcdf(output_file, encoding={'OHC_global': {'zlib': True, 'complevel': 4}})
        
        if self.verbose:
            print(f"✓ Output saved: {output_file}")
    
    def compute_spatial_ohc_field(
        self,
        time_index: int
    ) -> np.ndarray:
        """
        Compute spatial OHC field for a single time step.
        
        This function is provided as a utility for generating spatial fields
        if needed, but is not called by default (requires ~1-2 GB per field).
        
        Parameters
        ----------
        time_index : int
            Time index to compute (0 to ntime-1)
            
        Returns
        -------
        ohc_spatial : ndarray
            2D OHC field [J/m²], shape (nlat, nlon)
            
        Example
        -------
        >>> calc = CESM2OHCCalculator(temp_file)
        >>> ohc_field = calc.compute_spatial_ohc_field(0)  # First month
        >>> ohc_field.shape  # (384, 320)
        """
        if not self._grid_loaded:
            self._load_grid_info()
        
        # Get reference climatology
        temp_ref = self._get_reference_climatology()
        
        with xr.open_dataset(self.temp_file, decode_times=True) as ds:
            temp = ds['TEMP'].isel(time=time_index).values  # (nz, nlat, nlon)
        
        with xr.open_dataset(self.rho_file, decode_times=True) as ds:
            rho = ds['RHO'].isel(time=time_index).values     # (nz, nlat, nlon)
        
        # Calculate anomaly
        delta_t = temp - temp_ref
        
        # Convert RHO
        rho_kg_m3 = rho * 1000.0
        
        # Vertical integration
        ohc_per_vol = rho_kg_m3 * self.CP_SEAWATER * delta_t
        dz_expanded = self.dz[:, np.newaxis, np.newaxis]
        ohc_spatial = (ohc_per_vol * dz_expanded).sum(axis=0)
        
        # Apply mask
        ohc_spatial[self.kmt == 0] = np.nan
        
        return ohc_spatial


def create_validation_plots(time, ohc_global, output_dir):
    """
    Create validation plots for OHC time series.
    
    Parameters
    ----------
    time : ndarray
        Time coordinates (can be numeric or datetime objects)
    ohc_global : ndarray
        Global OHC time series [J/m²]
    output_dir : str or Path
        Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert time to numeric if needed
    # If time is datetime-like, convert to days since first value
    if hasattr(time[0], 'year'):  # cftime or datetime object
        import cftime
        if isinstance(time[0], (cftime.datetime, cftime.DatetimeNoLeap)):
            # Convert cftime to days since start
            start_time = time[0]
            time_numeric = np.array([(t - start_time).days + (t - start_time).seconds / 86400.0 
                                     for t in time])
            time_label = 'Time (days from start)'
        else:
            time_numeric = np.arange(len(time))
            time_label = 'Time (months)'
    else:
        time_numeric = time / 365.0  # Convert days to years
        time_label = 'Time (years)'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Full time series
    ax = axes[0, 0]
    ax.plot(time_numeric, ohc_global, 'b-', linewidth=1.5, label='OHC')
    ax.set_xlabel(time_label)
    ax.set_ylabel('OHC (J/m²)')
    ax.set_title('Global Ocean Heat Content Time Series')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: OHC anomaly (deviation from mean)
    ax = axes[0, 1]
    ohc_anom = ohc_global - np.mean(ohc_global)
    ax.plot(time_numeric, ohc_anom, 'r-', linewidth=1.5)
    ax.set_xlabel(time_label)
    ax.set_ylabel('OHC Anomaly (J/m²)')
    ax.set_title('OHC Anomaly (deviation from mean)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: Monthly change (time derivative)
    ax = axes[1, 0]
    delta_ohc = np.diff(ohc_global)
    delta_time = np.diff(time_numeric)
    # Normalize to per-month rate
    ohc_rate = delta_ohc / delta_time * 12.0 if 'months' in time_label else delta_ohc / delta_time
    ax.plot(time_numeric[1:], ohc_rate, 'g-', linewidth=1, alpha=0.7)
    ax.set_xlabel(time_label)
    ax.set_ylabel('OHC Rate of Change (J/m²/month)' if 'month' in time_label else 'OHC Rate of Change (J/m²)')
    ax.set_title('Ocean Heat Content Change Rate')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    CESM2 Ocean Heat Content Statistics
    
    Time Coverage: {len(ohc_global)} months
    
    OHC Statistics:
      Mean:     {np.mean(ohc_global):.2e} J/m²
      Std Dev:  {np.std(ohc_global):.2e} J/m²
      Min:      {np.min(ohc_global):.2e} J/m²
      Max:      {np.max(ohc_global):.2e} J/m²
      Range:    {np.max(ohc_global) - np.min(ohc_global):.2e} J/m²
    
    Rate of Change:
      Mean rate: {np.mean(ohc_rate):.2e} J/m²/month
      Std Dev:   {np.std(ohc_rate):.2e} J/m²/month
    
    Methodology:
      Reference: First 10 years
      Integration: Full ocean depth (60 levels)
      EOS: Full (using model ρ)
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'ohc_validation_plots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Validation plots saved: {output_file}")
    
    return fig


def main():
    """Main execution function."""
    
    # Paths
    temp_file = '/gdex/data/d651059/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI.001/ocn/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-G6-1p5K-SAI.001.pop.h.TEMP.203501-208412.nc'
    
    output_dir = Path('/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/data/OHC_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'cesm2_ohc_global_timeseries.nc'
    
    print("="*70)
    print("CESM2 OCEAN HEAT CONTENT CALCULATOR")
    print("="*70)
    
    # Initialize calculator
    calc = CESM2OHCCalculator(temp_file, verbose=True)
    
    # Compute OHC time series
    time, ohc_global = calc.compute_global_ohc_timeseries(
        output_file=str(output_file),
        chunk_size=12  # Process 1 year at a time
    )
    
    # Create validation plots
    print("\nGenerating validation plots...")
    create_validation_plots(time, ohc_global, output_dir)
    
    print("\n" + "="*70)
    print("CALCULATION COMPLETE")
    print("="*70)
    print(f"Output file: {output_file}")
    print(f"Plots saved in: {output_dir}")


if __name__ == '__main__':
    main()
