"""Discover, load and plot the PRISM ocean-heat-content (OHC) time series.

The OHC time series live under PRISM_data/spatial_OHC_data/<family>/.../*.pop.h.OHC.*.nc, one or more files per
model case. Every file holds OHC_global_mean (J per m^2 of ocean, by depth bin) and an ocean_area_m2 attribute, so
the global total heat content is OHC_global_mean * ocean_area_m2 [J]. (The other OHC files in PRISM_data --
control_baselines_ocn/ and spatial_maps_ohc/ -- are time-mean maps with no time axis.)

Time coordinates are checked, not assumed: file names are not trusted (some names give a different date range than
the file holds), files of one case are combined on their decoded time coordinate whatever their encoding, and every
case must form an unbroken run of first-of-month noleap months with no duplicates.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

PRISM_DATA = "/glade/work/jonahshaw/PRISM_data/"
OHC_TSERIES_ROOT = PRISM_DATA + "spatial_OHC_data/"
ZJ = 1e21

CASE_RE = re.compile(r"^(?P<case>.+)\.pop\.h\.OHC\..+\.nc$")
MEMBER_RE = re.compile(r"^(?P<experiment>.+)\.(?P<member>e?\d{2,3})$")  # trailing ensemble number, e.g. .001 or .e11
GRID_RE = re.compile(r"\.f\d\d_g\d\d\.")


def discover_cases(root=OHC_TSERIES_ROOT):
    """One row per model case: family (top-level directory), experiment (case name without its ensemble number),
    member, case and its files."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*.nc"), recursive=True)):
        m = CASE_RE.match(os.path.basename(path))
        if m is None:
            continue  # e.g. ancillary_files/ohc_ancillary_data.nc
        case = m.group("case")
        family = os.path.relpath(path, root).split(os.sep)[0]
        rows.setdefault((family, case), []).append(path)
    records = []
    for (family, case), files in rows.items():
        mm = MEMBER_RE.match(case)
        experiment, member = (mm.group("experiment"), mm.group("member")) if mm else (case, "000")
        records.append(dict(family=family, experiment=experiment, member=member, case=case, files=files))
    return pd.DataFrame(records).sort_values(["family", "experiment", "member"]).reset_index(drop=True)


def experiment_label(experiment):
    """Short name: the part of the case name after the grid (e.g. 'SSP245-TSMLT-GAUSS-DEFAULT')."""
    parts = GRID_RE.split(experiment)
    return parts[-1] if len(parts) > 1 else experiment


def _check_monthly_axis(case, time):
    t = time.values
    if not all(type(d).__name__ == "DatetimeNoLeap" for d in t):
        raise ValueError(f"{case}: time is not decoded to a noleap calendar")
    if not all(d.day == 1 for d in t):
        raise ValueError(f"{case}: time labels are not first-of-month")
    ym = np.array([d.year * 12 + d.month - 1 for d in t])
    step = np.diff(ym)
    if np.any(step == 0):
        raise ValueError(f"{case}: duplicate months across files")
    if np.any(step != 1):
        raise ValueError(f"{case}: time axis has gaps or is out of order")


def load_case_total(files, case=""):
    """Global total OHC [J] for one case, dims (ohc_depth, time), with its files combined on time."""
    ds = xr.open_mfdataset(
        sorted(files), combine="by_coords", data_vars="minimal", coords="minimal", compat="override",
        drop_variables=["OHC"], use_cftime=True, chunks={},
    )
    if "ocean_area_m2" not in ds.attrs:
        raise ValueError(f"{case}: no ocean_area_m2 attribute; can't convert OHC_global_mean to a total")
    total = (ds["OHC_global_mean"] * ds.attrs["ocean_area_m2"]).load()
    ds.close()
    _check_monthly_axis(case, total["time"])
    total.attrs = dict(units="J", long_name="Global total ocean heat content")
    return total


def annual_mean(total):
    """Annual means over complete (12-month) years only, dims (ohc_depth, year)."""
    counts = total["time"].dt.year.to_series().value_counts()
    full_years = counts[counts == 12].index
    annual = total.groupby("time.year").mean()
    return annual.sel(year=sorted(full_years))


def load_experiments(cases=None, depth=-1, verbose=True):
    """Annual total OHC change since each member's first year [J] for every experiment.

    Returns {(family, experiment): DataArray (member, year)}. Members whose data are identical to another member's
    are dropped (the same run stored twice under two case names).
    """
    cases = discover_cases() if cases is None else cases
    out = {}
    for (family, experiment), grp in cases.groupby(["family", "experiment"], sort=False):
        members = []
        for row in grp.itertuples():
            annual = annual_mean(load_case_total(row.files, row.case)).sel(ohc_depth=depth)
            change = (annual - annual.isel(year=0)).expand_dims(member=[row.member])
            duplicate = next((m for m in members if m.sizes["year"] == change.sizes["year"]
                              and np.array_equal(m["year"], change["year"])
                              and np.array_equal(m.values, change.values)), None)
            if duplicate is not None:
                if verbose:
                    print(f"  dropped {row.case}: identical to member {duplicate['member'].item()}")
                continue
            members.append(change)
        da = xr.concat(members, dim="member", join="outer")
        da.attrs = dict(units="J", long_name="Annual total OHC change since first year", ohc_depth=depth)
        out[(family, experiment)] = da
    return out


def summary_table(experiments):
    rows = []
    for (family, experiment), da in experiments.items():
        count = da.notnull().sum("member")
        full = count == da.sizes["member"]
        rows.append(dict(
            family=family, experiment=experiment_label(experiment), members=da.sizes["member"],
            years=f"{int(da.year[0])}–{int(da.year[-1])}",
            all_members_through=int(da.year[full][-1]),
            change_ZJ=float(da.mean("member").dropna("year")[-1]) / ZJ,
        ))
    return pd.DataFrame(rows).set_index(["family", "experiment"])


# Reference categorical palette (fixed slot order) and chart ink, from the dataviz method.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "lines.linewidth": 2, "font.size": 10, "legend.frameon": False, "legend.labelcolor": INK2, "figure.dpi": 110,
}


def _end_labels(ax, ends, lines, min_gap_frac=0.045, label_width_frac=0.1):
    """Direct value labels at line ends, nudged apart vertically; a label is skipped when another line runs
    through the space it would occupy (the legend and summary table still carry that value)."""
    if not ends:
        return
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    gap, width = min_gap_frac * (ymax - ymin), label_width_frac * (xmax - xmin)
    placed = []
    for x, y, text, own in sorted(ends, key=lambda e: e[1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        blocked = any(np.any((xs > x) & (xs <= x + width) & (np.abs(ys - y_text) < gap))
                      for i, (xs, ys) in enumerate(lines) if i != own)
        if blocked:
            continue
        placed.append(y_text)
        ax.annotate(text, xy=(x, y), xytext=(x, y_text), textcoords="data", va="center", ha="left",
                    color=INK2, fontsize=8.5, annotation_clip=False)


def plot_family(ax, experiments, family, per_year=False, reference_year=None):
    """Plot every experiment of one family on `ax`: single runs as lines; ensembles as the member mean with a
    min-max band over the years all members cover, continued as a thinner mean where only some members run on.

    per_year: plot the year-to-year change (ZJ/yr) instead of the change since the first year.
    reference_year: re-reference every run to this year instead of its own first year (runs that start later
    are left out), so experiments with different start years share a zero point.
    """
    items = [(exp, da) for (fam, exp), da in experiments.items() if fam == family]
    if reference_year is not None:
        items = [(exp, da - da.sel(year=reference_year)) for exp, da in items if reference_year in da["year"]]
    ends, lines = [], []
    for slot, (experiment, da) in enumerate(items):
        color = PALETTE[slot % len(PALETTE)]
        if per_year:
            da = da.diff("year")
        da = da / ZJ
        year = da["year"].values
        n = da.sizes["member"]
        count = da.notnull().sum("member").values
        mean = da.mean("member").values
        name = experiment_label(experiment)
        if n == 1:
            ax.plot(year, mean, color=color, lw=1.8, label=f"{name} (1 run)")
        else:
            full = count == n
            ax.fill_between(year[full], da.min("member").values[full], da.max("member").values[full],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(year[full], mean[full], color=color, lw=2, label=f"{name} (mean of {n}, min–max)")
            tail = np.flatnonzero(~full & (count > 0))
            if tail.size:
                seg = np.r_[np.flatnonzero(full)[-1], tail]  # start the tail at the last full-membership year
                ax.plot(year[seg], mean[seg], color=color, lw=1, label=f"  {name}: mean of remaining members")
        last = np.flatnonzero(np.isfinite(mean))[-1]
        lines.append((year, mean))
        ends.append((year[last], mean[last], f" {mean[last]:+.0f}" + (" ZJ/yr" if per_year else " ZJ")
                     + (f" (n={count[last]})" if n > 1 and count[last] < n else ""), len(lines) - 1))
    ax.axhline(0, color=AXIS, lw=0.8)
    x_lo, x_hi = ax.get_xlim()
    ax.set_xlim(x_lo, x_hi + 0.12 * (x_hi - x_lo))  # room for end labels
    if len(items) <= 4:
        _end_labels(ax, ends, lines)
    ax.legend(loc="upper left", fontsize=8.5)
    since = f"since {reference_year}" if reference_year is not None else "since first year"
    ax.set(title=family, xlabel="Year" if not family.endswith(("1850control", "LME")) else "Model year",
           ylabel="OHC change per year [ZJ/yr]" if per_year else f"OHC change {since} [ZJ]")
    return ax
