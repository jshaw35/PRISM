# %%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from copy import deepcopy

from J15_shared_functions import (
    weighted_annualmean,
    get_weights_by_month2,
    compute_decadal2,
    load_data_with_configs,
    title_dict,
    CASE_CONFIGS_TEMPLATE,
)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# %%

if __name__ == "__main__":
    data_root = "/glade/work/jonahshaw/PRISM_data/spatial_averages_data/"
    # data_root_sfens = "/glade/u/home/jonahshaw/Scripts/git_repos/PRISM"

    # CASE_CONFIGS_ATM = deepcopy(CASE_CONFIGS_TEMPLATE)

    # Albedo cases
    test_cases = ["CESM2-LM", "CESM2_WACCM_1850control", "CESM2_WACCM_SSP2-4.5"]
    CASE_CONFIGS_ALBEDO = {}
    for i in test_cases:
        CASE_CONFIGS_ALBEDO[i] = CASE_CONFIGS_TEMPLATE[i]

    for config in CASE_CONFIGS_ALBEDO.values():
        config["path"] = str(Path(data_root) / config.pop("path"))

    # %%
    # Load the data using the generalized loading function
    data_varlist = ['SOLIN', 'FSNT', 'TS']
    year_dim = "time"

    # Load data using the generalized function (requires 4 nodes on CURC)
    data_dict = {}
    for var in data_varlist:
        data_dict[var] = load_data_with_configs(CASE_CONFIGS_ALBEDO, [var], year_dim=year_dim, load_into_memory=False)

    # %%
    # For each case in CASE_CONFIGS_ALBEDO, compute the planetary albedo
    albedo_dict = {}
    for case_label, case_config in CASE_CONFIGS_ALBEDO.items():
        logging.info(f"Computing planetary albedo for case: {case_label}")
        sub_dict = {}
        for case_str, ds in data_dict["SOLIN"][case_label].items():
            data_list = []
            data_list.append(data_dict["SOLIN"][case_label][case_str].sel(spatial="G").squeeze())
            data_list.append(data_dict["FSNT"][case_label][case_str].sel(spatial="G").squeeze())
            data_list.append(data_dict["TS"][case_label][case_str].sel(spatial="G").squeeze())
            data_ds = xr.merge(data_list)
            albedo = (1 - (data_ds["FSNT"] / data_ds["SOLIN"])).rename("albedo")
            data_ds["A"] = albedo
            sub_dict[case_str] = data_ds
        albedo_dict[case_label] = sub_dict

    # %%
    # Regression analysis for albedo vs. surface temperature
    regression_dict = {}
    for case_label, case_config in albedo_dict.items():
    # for case_label in ["CESM2_WACCM_SSP2-4.5"]:
        logging.info(f"Computing planetary albedo regression for case: {case_label}")
        sub_dict = {}
        for case_str, ds in albedo_dict[case_label].items():
            ds = weighted_annualmean(ds) # Annual mean
            # Regress albedo (A) against surface temperature (TS), ignoring
            # missing values and retaining one slope/intercept for each case.
            valid = ds["A"].notnull() & ds["TS"].notnull()
            temperature = ds["TS"].where(valid).values.ravel()
            albedo = ds["A"].where(valid).values.ravel()
            valid_values = np.isfinite(temperature) & np.isfinite(albedo)
            temperature = temperature[valid_values]
            albedo = albedo[valid_values]

            if temperature.size < 2 or np.ptp(temperature) == 0:
                logging.warning("Insufficient TS variation for regression: %s/%s", case_label, case_str)
                sub_dict[case_str] = {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan}
                continue

            slope, intercept = np.polyfit(temperature, albedo, 1)
            predicted = slope * temperature + intercept
            ss_res = np.sum((albedo - predicted) ** 2)
            ss_tot = np.sum((albedo - albedo.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan
            sub_dict[case_str] = {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
            }
            # Scatter plot of albedo vs. surface temperature with regression line
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                temperature,
                albedo,
                s=24,
                alpha=0.65,
                edgecolors="none",
                label="Annual means",
            )
            temperature_range = np.linspace(temperature.min(), temperature.max(), 200)
            ax.plot(
                temperature_range,
                slope * temperature_range + intercept,
                color="tab:red",
                linewidth=2,
                label=f"Fit: intercept={intercept:.3g}, slope={slope:.3g}, $R^2$={r_squared:.3f}",
            )
            ax.set_xlabel("Surface temperature (K)")
            ax.set_ylabel("Planetary albedo")
            ax.set_title(f"Albedo vs. surface temperature: {case_label}\n{case_str}")
            ax.grid(True, alpha=0.3)
            ax.legend()

            output_dir = Path("/glade/u/home/jonahshaw/Scripts/git_repos/PRISM/figures")

            fig.savefig(
                output_dir
                / f"albedo_temperature_regression_{case_label}_{case_str}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
        regression_dict[case_label] = sub_dict

    # %%
