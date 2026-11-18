"""
Branch figure for the J04 ..._with_simpleModel_verB / _verC scripts.

Each column shows one "trunk" run and the runs that branch off it, on the same axes, so a stabilized
run and its unstabilized trunk can be compared directly (request from the 11 Nov 2026 meeting):

    column 1  CESM2-WACCM6   trunk: historical + SSP2-4.5          branches: ARISE-SAI runs
    column 2  CESM2 (CAM6)   trunk: CMIP6 SSP2-4.5 baseline         branches: CMIP6 MCB runs
    column 3  three-layer model  trunk: no SAI                      branches: SAI controllers

Colors identify the variable (same colors as the calling script); line styles identify the run, with
a legend under each column. Only ensemble means are drawn (decadal running means for CESM2; annual
means for the model, which has no internal variability).

Rows: 2 rows with twinned right-hand axes (verB style: ASR/OLR + EEI, iEEI/OHC + surface temperature)
or 4 rows, one per variable group (verC style).

x-axis: "continuous" (one axis from x_start, default CONTINUOUS_START_YEAR = 1950 as in Jonah Shaw's
slide 30 of 12 Nov 2026, to 2100) or "broken" (1850-2015 compressed on the left, 2015-2100 on the right).

Accumulated quantities (iEEI, OHC), current J04 convention:
    The historical run is integrated from 1850. Every 2015-onward run (SSP2-4.5, ARISE, and the CAM6
    baseline and MCB runs) is integrated from its own start in 2015 and then shifted up by the
    WACCM6 historical run's final-year (2014) annual-mean iEEI and OHC, as J04 does with
    "preceding_case". For the CAM6 column this borrows the WACCM6 historical accumulation, because
    there is no CAM6 historical run in our data. The figure says so under the CAM6 column.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from J04_CESM_EEI_timeseries_multiensemble import compute_ieei_with_start_year
from J15_shared_functions import compute_decadal2, weighted_annualmean

CONTINUOUS_START_YEAR = 1950
BROKEN_SEGMENTS = [(1850, 2015), (2015, 2100)]
# Width of each segment relative to the others: the historical period is compressed
BROKEN_WIDTH_RATIOS = [1, 2]
# Trunk first, then branches in order
BRANCH_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]

# (variable, axes role, index into the six-color list ASR, OLR, EEI, iEEI, surface temperature, OHC)
VARIABLES = [
    ("asr", "rad", 0),
    ("olr", "rad", 1),
    ("eei", "eei", 2),
    ("ieei", "heat", 3),
    ("ts", "ts", 4),
    ("ohc", "heat", 5),
]


def default_ylims(four_rows: bool) -> dict:
    """
    Shared y-limits per axes role. ASR/OLR reach down to 232 W m-2 because the model's zero-imbalance
    controller drives them to ~233.5 W m-2 by 2099. With twinned axes (2 rows) the EEI range keeps the
    same 13-unit span as ASR/OLR, so one tick on the left equals one tick on the right.
    """
    if four_rows:
        return {"rad": (232, 245), "eei": (-1, 3), "heat": (-0.1e9, 6e9), "ts": (287, 292)}
    return {"rad": (232, 245), "eei": (-9, 4), "heat": (-0.1e9, 6e9), "ts": (286, 293)}


# ------------------------------------------------------------------------------------------------
# Series
# ------------------------------------------------------------------------------------------------
def cesm2_monthly_means(data_dict: dict, ohc_dict: dict, case_label: str, case_str: str, earth_sa_m2: float) -> dict:
    """
    Global, ensemble-mean monthly series of one CESM2 case.

    Returns dict of 1-D DataArrays on the case's time axis: asr, olr [W m-2], ts [K], ohc [J m-2 of
    Earth surface, anomaly relative to the case's first month], and n_ens (number of members).
    """

    def global_series(var):
        return data_dict[var][case_label][case_str][var].sel(spatial="G").squeeze().compute()

    asr = global_series("FSNT")
    olr = global_series("FLNT")
    ts = global_series("TS")
    n_ens = int(asr.sizes["ens"]) if "ens" in asr.dims else 1

    # Full-depth (ohc_depth=-1) global-mean OHC [J m-2 of ocean] -> [J m-2 of Earth surface], anomaly per
    # member relative to the first month (as in J04)
    ohc_case = ohc_dict[case_label][case_str].squeeze()
    ohc = ohc_case["OHC_global_mean"].sel(ohc_depth=-1).compute() * ohc_case.attrs["ocean_area_m2"] / earth_sa_m2
    ohc = ohc - ohc.isel(time=0)

    out = {}
    for name, da in (("asr", asr), ("olr", olr), ("ts", ts), ("ohc", ohc)):
        if "ens" in da.dims:
            da = da.mean("ens")
        out[name] = da.reset_coords(drop=True)
    out["n_ens"] = n_ens
    return out


def _time_only(da: xr.DataArray) -> xr.DataArray:
    """Drop every coordinate except time (e.g. the "month" left by the iEEI month-length weights), so
    series from different runs can be concatenated."""
    return da.drop_vars([c for c in da.coords if c != "time"])


def chain_with_history(hist: dict, run: dict) -> dict:
    """
    Join the historical monthly series and a 2015-onward run, then return decadal means of every
    variable as (years, values) numpy pairs, plus the run's first year.

    iEEI and OHC follow the current J04 convention (see module docstring): the run's own accumulation
    is shifted by the historical run's final-year annual mean.
    """
    run_start = int(run["asr"]["time.year"].values[0])
    h = {k: v.where(v["time.year"] < run_start, drop=True) for k, v in hist.items() if k != "n_ens"}

    # iEEI [J m-2]: historical integrated from 1850; the run from its own start, shifted by the
    # historical final-year annual mean (J04's case_end_ohc_ieei_offsets)
    ieei_hist = compute_ieei_with_start_year(h["asr"], h["olr"], 1850)
    ieei_run = compute_ieei_with_start_year(run["asr"], run["olr"], run_start)
    ieei_run = ieei_run + weighted_annualmean(ieei_hist).isel(year=-1)
    # OHC [J m-2]: the run's anomaly shifted by the historical final-year annual mean, as in J04
    ohc_run = run["ohc"] + weighted_annualmean(h["ohc"]).isel(year=-1)

    monthly = {
        name: xr.concat([_time_only(a), _time_only(b)], dim="time")
        for name, (a, b) in {
            "asr": (h["asr"], run["asr"]),
            "olr": (h["olr"], run["olr"]),
            "ts": (h["ts"], run["ts"]),
            "ieei": (ieei_hist, ieei_run),
            "ohc": (h["ohc"], ohc_run),
        }.items()
    }
    monthly["eei"] = monthly["asr"] - monthly["olr"]

    series = {}
    for name, da in monthly.items():
        dec = compute_decadal2(da)  # 10-year centred running mean of annual means, NaN within 5 years of the ends
        series[name] = (dec["year"].values, dec.values)
    series["run_start"] = run_start
    return series


def model_series(fields: dict) -> dict:
    """Simple-model annual means (from simple_model_annual_fields) as (years, values) pairs."""
    keys = {"asr": "asr_W_m2", "olr": "olr_W_m2", "eei": "eei_W_m2", "ieei": "ieei_J_m2", "ohc": "ohc_J_m2", "ts": "ts_K"}
    return {name: (fields["year"], fields[key]) for name, key in keys.items()}


def build_column_series(columns: list, data_sources: dict, ohc_sources: dict, hist_case: tuple, earth_sa_m2: float, model_runs: dict) -> list:
    """
    Series for every run of every column. Each run dict gains "series" and "label_full" (label with N).

    data_sources / ohc_sources map a run's "data" key ("main", "cam6", ...) to its data_dict / ohc_dict.
    hist_case is (case_label, case_str) of the WACCM6 historical run, read from data_sources["main"].
    model_runs maps an SAI controller (None = no SAI) to simple_model_annual_fields output.
    """
    hist = cesm2_monthly_means(data_sources["main"], ohc_sources["main"], *hist_case, earth_sa_m2)
    out = []
    for col in columns:
        runs = []
        for run in col["runs"]:
            run = dict(run)
            if run["kind"] == "model":
                run["series"] = model_series(model_runs[run["controller"]])
                run["label_full"] = run["label"]
            else:
                src = run.get("data", "main")
                m = cesm2_monthly_means(data_sources[src], ohc_sources[src], run["case_label"], run["case_str"], earth_sa_m2)
                run["series"] = chain_with_history(hist, m)
                n = f"N={m['n_ens']}"
                if run.get("with_history_label"):
                    n = f"N={hist['n_ens']}, {m['n_ens']}"
                run["label_full"] = f"{run['label']} ({n})"
            runs.append(run)
        out.append(dict(col, runs=runs))
    return out


# ------------------------------------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------------------------------------
def panel_letter(i: int) -> str:
    """a, b, ..., z, then aa, ab, ..."""
    return chr(97 + i) if i < 26 else chr(97 + i // 26 - 1) + chr(97 + i % 26)


def _left_title(ax, parts, fontsize, offset_pts=48):
    """
    y-axis title drawn as text a fixed distance (points) left of the axes, so it lines up across rows and
    segments of different widths. parts: [(text, color)], one part (centred) or two (stacked, meeting
    at mid-height, as in J04's two-colored "ASR, OLR" title).
    """
    trans = offset_copy(ax.transAxes, fig=ax.figure, x=-offset_pts, y=0, units="points")
    if len(parts) == 1:
        ax.text(0, 0.5, parts[0][0], transform=trans, rotation=90, ha="center", va="center", fontsize=fontsize, color=parts[0][1])
    else:
        (top, top_color), (bottom, bottom_color) = parts
        ax.text(0, 0.42, top, transform=trans, rotation=90, ha="center", va="top", fontsize=fontsize, color=top_color)
        ax.text(0, 0.43, bottom, transform=trans, rotation=90, ha="center", va="bottom", fontsize=fontsize, color=bottom_color)


def _break_marks(ax_left, ax_right):
    """Diagonal '//' marks where a broken x-axis jumps between segments (matplotlib broken-axis recipe)."""
    kw = dict(marker=[(-1, -1), (1, 1)], markersize=10, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kw)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kw)


def plot_branch_figure(
    columns: list,
    colors: list,
    four_rows: bool,
    x_mode: str,
    ylims: dict = None,
    fontsize: int = 14,
    banner: str = None,
    x_start: int = None,
):
    """
    Draw the branch figure. columns: output of build_column_series. colors: six colors ordered ASR, OLR,
    EEI, iEEI, surface temperature, OHC. x_mode: "continuous" or "broken". x_start: first year of the
    continuous axis (default CONTINUOUS_START_YEAR); ignored for "broken". Returns the figure.
    """
    if x_mode not in ("continuous", "broken"):
        raise ValueError(f"x_mode={x_mode!r} must be 'continuous' or 'broken'")
    ylims = default_ylims(four_rows) if ylims is None else ylims
    x_start = CONTINUOUS_START_YEAR if x_start is None else x_start
    segments = [(x_start, 2100)] if x_mode == "continuous" else BROKEN_SEGMENTS
    width_ratios = [1] if x_mode == "continuous" else BROKEN_WIDTH_RATIOS
    n_cols, n_seg = len(columns), len(segments)
    n_rows = 4 if four_rows else 2

    fig = plt.figure(figsize=(5.5 * n_cols, 4.5 * n_rows + 1))
    outer = fig.add_gridspec(1, n_cols, wspace=0.28 if four_rows else 0.32, top=0.93 if four_rows else 0.88)

    letter = 0
    for c, col in enumerate(columns):
        inner = outer[0, c].subgridspec(n_rows, n_seg, width_ratios=width_ratios, wspace=0.03, hspace=0.25)
        cells = []
        for s in range(n_seg):
            if four_rows:
                rad = fig.add_subplot(inner[0, s])
                cell = {"rad": rad}
                for r, role in ((1, "eei"), (2, "heat"), (3, "ts")):
                    cell[role] = fig.add_subplot(inner[r, s], sharex=rad)
            else:
                rad = fig.add_subplot(inner[0, s])
                heat = fig.add_subplot(inner[1, s], sharex=rad)
                cell = {"rad": rad, "eei": rad.twinx(), "heat": heat, "ts": heat.twinx()}
            cells.append(cell)

        for s, (cell, (x0, x1)) in enumerate(zip(cells, segments)):
            # -- curves: one line style per run, one color per variable ----------------------------
            for run in col["runs"]:
                for var, role, ci in VARIABLES:
                    years, values = run["series"][var]
                    keep = years >= run["draw_from"]
                    cell[role].plot(years[keep], values[keep], color=colors[ci], linestyle=run["linestyle"], linewidth=2.5)
            # A column without data in this segment (CAM6 before 2015) says so
            if all(run["draw_from"] >= x1 for run in col["runs"]):
                cell["rad"].text(0.5, 0.5, "no CAM6\nhistorical run", transform=cell["rad"].transAxes,
                                 ha="center", va="center", fontsize=fontsize - 3, color="0.4")

            # -- axes formatting -------------------------------------------------------------------
            base_roles = ("rad", "eei", "heat", "ts") if four_rows else ("rad", "heat")
            twin_roles = () if four_rows else ("eei", "ts")
            for role, ax in cell.items():
                ax.set_ylim(*ylims[role])
            for role in base_roles:
                ax = cell[role]
                ax.set_xlim(x0, x1)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor("whitesmoke")
            cell["eei"].axhline(0, color=colors[2], linestyle="--", linewidth=1)
            cell["eei"].tick_params(axis="y", labelcolor=colors[2])
            cell["ts"].tick_params(axis="y", labelcolor=colors[4])
            # Broken axis: fewer year ticks, kept away from the break so labels do not run together
            if x_mode == "broken":
                cell["rad"].set_xticks([1850, 1950] if s == 0 else [2040, 2060, 2080, 2100])

            # Tick labels: left axes on the first segment of each column, right twins on the last
            first, last = s == 0, s == n_seg - 1
            for role in base_roles:
                cell[role].tick_params(axis="y", left=first, labelleft=first)
                if not first:
                    cell[role].spines["left"].set_visible(False)
                if not last:
                    cell[role].spines["right"].set_visible(False)
            for role in twin_roles:
                cell[role].tick_params(axis="y", right=last, labelright=last)
                if not first:
                    cell[role].spines["left"].set_visible(False)
                if not last:
                    cell[role].spines["right"].set_visible(False)

            # "Year" only under the bottom row; x tick labels on every row
            bottom = cell["ts"] if four_rows else cell["heat"]
            bottom.set_xlabel("Year" if s == n_seg - 1 or n_seg == 1 else "", fontsize=fontsize)

        # Break marks between segments, on every row
        for s in range(n_seg - 1):
            for role in (("rad", "eei", "heat", "ts") if four_rows else ("rad", "heat")):
                _break_marks(cells[s][role], cells[s + 1][role])

        # -- y-axis titles: left titles on the first column; in 2-row mode right titles on the last ----
        if c == 0:
            _left_title(cells[0]["rad"], [("ASR,", colors[0]), ("OLR [Wm$^{-2}$]", colors[1])], fontsize)
            _left_title(cells[0]["heat"], [("iEEI,", colors[3]), ("OHC [Jm$^{-2}$]", colors[5])], fontsize)
            if four_rows:
                _left_title(cells[0]["eei"], [("EEI [Wm$^{-2}$]", colors[2])], fontsize)
                _left_title(cells[0]["ts"], [("Surface Temperature [K]", colors[4])], fontsize)
        if not four_rows and c == n_cols - 1:
            cells[-1]["eei"].set_ylabel("EEI [Wm$^{-2}$]", fontsize=fontsize, color=colors[2])
            cells[-1]["ts"].set_ylabel("Surface Temperature [K]", fontsize=fontsize, color=colors[4])

        # -- panel letters, one per row of the column (on its first segment) ------------------------
        for role in (("rad", "eei", "heat", "ts") if four_rows else ("rad", "heat")):
            cells[0][role].text(0.03, 0.97, f"{panel_letter(letter)}.", transform=cells[0][role].transAxes,
                                fontsize=fontsize, fontweight="bold", va="top")
            letter += 1

        # -- column title and run legend: on an invisible axes spanning the whole column -------------
        holder = fig.add_subplot(outer[0, c], frameon=False)
        holder.set_axis_off()
        holder.set_title(col["title"], fontsize=fontsize, pad=8)
        handles = [Line2D([], [], color="black", linestyle=r["linestyle"], linewidth=2.5, label=r["label_full"]) for r in col["runs"]]
        legend = holder.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False,
                               fontsize=fontsize - 2, title=col.get("note"), title_fontsize=fontsize - 4)
        legend._legend_box.align = "left"

    if banner:
        fig.text(0.5, 0.995, banner, ha="center", va="top", color="red", fontsize=fontsize, fontweight="bold")
    logging.info(f"Branch figure drawn: {n_rows} rows, x_mode={x_mode}")
    return fig
