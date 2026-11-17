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
    """Month-length-weighted annual means over complete (12-month) years only (the `time` dim becomes `year`)."""
    counts = total["time"].dt.year.to_series().value_counts()
    full_years = counts[counts == 12].index
    days = total["time"].dt.days_in_month
    annual = (total * days).groupby("time.year").sum() / days.groupby("time.year").sum()
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
            annual = annual_mean(load_case_total(row.files, row.case))
            if depth is not None:  # depth=None keeps every depth bin
                annual = annual.sel(ohc_depth=depth)
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
        da.attrs = dict(units="J", long_name="Annual total OHC change since first year")
        out[(family, experiment)] = da
    return out


LAYERS = ["0–300 m", "300–700 m", "700–2000 m", "below 2000 m"]


def layer_changes(da):
    """Split the cumulative depth bins (0–300, 0–700, 0–2000 m, full column = -1) into the four layers between
    them. Works on anything with an ohc_depth dimension; returns the same dims with ohc_depth replaced by layer."""
    b = {int(d): da.sel(ohc_depth=d, drop=True) for d in da["ohc_depth"].values}
    layers = [b[300], b[700] - b[300], b[2000] - b[700], b[-1] - b[2000]]
    return xr.concat(layers, dim=pd.Index(LAYERS, name="layer"))


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
# Ordinal blue ramp for depth layers, shallow (light) to deep (dark); validated with --ordinal.
LAYER_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
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


def plot_layers(ax, da, family, experiment, title=None, legend=True, pad_xlim=True):
    """Total OHC change of one experiment with the change in each depth layer shaded beneath it.

    da: (member, ohc_depth, year), as returned by load_experiments(depth=None). Ensembles show the member mean
    (total and layers); the members' min-max range of the total is drawn as thin lines over the years every member
    covers, and the mean continues thinner where only some members run on. Layers gaining heat stack upward from
    zero and layers losing heat stack downward, so the total equals the shaded area above zero minus that below.

    title: replaces the default "family: experiment (n members)" title. legend=False and pad_xlim=False are for
    panels in a shared-axis grid (one legend for the figure; x-limits set once by the caller).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n = da.sizes["member"]
    year = da["year"].values
    total = da.sel(ohc_depth=-1) / ZJ
    count = total.notnull().sum("member").values
    full = count == n
    layers = layer_changes(da).mean("member") / ZJ

    up, down = np.zeros(year.size), np.zeros(year.size)
    for layer, color in zip(LAYERS, LAYER_COLORS):
        v = np.nan_to_num(layers.sel(layer=layer).values)
        pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.fill_between(year, up, up + pos, color=color, edgecolor=SURFACE, linewidth=0.4)
        ax.fill_between(year, down, down + neg, color=color, edgecolor=SURFACE, linewidth=0.4)
        up, down = up + pos, down + neg

    mean_total = total.mean("member").values
    ax.plot(year[full], mean_total[full], color=INK, lw=2.2)
    handles = [Line2D([], [], color=INK, lw=2.2, label="total" if n == 1 else f"total (mean of {n})")]
    if n > 1:
        for stat in ("min", "max"):
            ax.plot(year[full], getattr(total, stat)("member").values[full], color=MUTED, lw=0.8)
        handles.append(Line2D([], [], color=MUTED, lw=0.8, label="total, member min–max"))
    tail = np.flatnonzero(~full & (count > 0))
    if tail.size:
        ax.plot(year[np.r_[np.flatnonzero(full)[-1], tail]], mean_total[np.r_[np.flatnonzero(full)[-1], tail]],
                color=INK, lw=1)
        ax.axvline(year[tail[0]] - 0.5, color=AXIS, lw=0.8)
    handles += [Patch(color=c, label=l) for l, c in zip(LAYERS, LAYER_COLORS)]

    ax.axhline(0, color=AXIS, lw=0.8)
    if pad_xlim:
        x_lo, x_hi = ax.get_xlim()
        ax.set_xlim(x_lo, x_hi + 0.1 * (x_hi - x_lo))
    if tail.size:
        k_first, k_last = count[tail[0]], count[tail[-1]]
        members = f"{k_first}" if k_first == k_last else f"{k_first}, then {k_last},"
        x_div = year[tail[0]] - 0.5
        x_lo, x_hi = ax.get_xlim()
        right_side = (x_div - x_lo) / (x_hi - x_lo) < 0.8  # near the right edge, put the note left of the divider
        ax.annotate(f" mean of {members} members after {int(year[tail[0] - 1])} " , xy=(x_div, 1),
                    xycoords=("data", "axes fraction"), va="top", ha="left" if right_side else "right",
                    color=MUTED, fontsize=8.5)
    last = np.flatnonzero(np.isfinite(mean_total))[-1]
    ax.annotate(f" {mean_total[last]:+.0f} ZJ", xy=(year[last], mean_total[last]), va="center", color=INK2, fontsize=9)
    if legend:
        ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    runs = "1 run" if n == 1 else f"{n} members"
    ax.set(title=title or f"{family}: {experiment_label(experiment)} ({runs})",
           xlabel="Year" if not family.endswith(("1850control", "LME")) else "Model year",
           ylabel=f"OHC change since {int(year[0])} [ZJ]")
    return handles


# --- Spatial maps of OHC change ----------------------------------------------------------------------------------

MAP_LAYERS = ["total", *LAYERS]
GJ = 1e9
# Diverging blue <-> red with a neutral gray midpoint (dataviz reference diverging pair).
DIVERGING_COLORS = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4aeac", "#e34948", "#8a1f1f"]
LAND_COLOR = "#e1e0d9"


def _window_mean(da, years, case=""):
    """Month-length-weighted mean of `da` over the given calendar years (all 12 months of each must be present)."""
    sub = da.isel(time=da["time"].dt.year.isin(years).values)
    if sub.sizes["time"] != 12 * len(years):
        raise ValueError(f"{case}: expected {12 * len(years)} months in {years[0]}–{years[-1]}, found {sub.sizes['time']}")
    return sub.weighted(sub["time"].dt.days_in_month).mean("time")


def change_maps(cases, experiments, n_years=5, verbose=True):
    """Map of the change in OHC per column area [J m-2] between the mean of the last and the first `n_years` years,
    for every experiment, on the native POP grid: {(family, experiment): DataArray (ohc_depth, nlat, nlon)}.

    Ensembles use the members kept by load_experiments (duplicates already dropped) and the first/last years that
    every member covers, and average the members' change maps. Each map also records the change in the members'
    mean OHC_global_mean over the same windows (attr global_mean_change) as an independent check on the map.
    """
    out = {}
    for (family, experiment), da in experiments.items():
        if "ohc_depth" in da.dims:  # accept load_experiments output with or without the depth dimension
            da = da.sel(ohc_depth=-1)
        n = da.sizes["member"]
        full_years = da["year"].values[(da.notnull().sum("member") == n).values]
        if full_years.size < 2 * n_years:
            raise ValueError(f"{experiment}: only {full_years.size} years with all members; need {2 * n_years}")
        first, last = list(full_years[:n_years]), list(full_years[-n_years:])
        rows = cases[(cases.family == family) & (cases.experiment == experiment) & cases.member.isin(da["member"].values)]
        member_maps, member_gm = [], []
        for row in rows.itertuples():
            ds = xr.open_mfdataset(sorted(row.files), combine="by_coords", data_vars="minimal", coords="minimal",
                                   compat="override", use_cftime=True, chunks={})
            _check_monthly_axis(row.case, ds["time"])
            ohc = ds["OHC"]
            member_maps.append((_window_mean(ohc, last, row.case) - _window_mean(ohc, first, row.case)).load())
            gm = ds["OHC_global_mean"].sel(ohc_depth=-1)
            member_gm.append(float(_window_mean(gm, last, row.case) - _window_mean(gm, first, row.case)))
            ds.close()
        change = xr.concat(member_maps, dim="member").mean("member")
        change.attrs = dict(units="J m-2", first_years=f"{first[0]}–{first[-1]}", last_years=f"{last[0]}–{last[-1]}",
                            n_members=n, global_mean_change=float(np.mean(member_gm)))
        out[(family, experiment)] = change
        if verbose:
            print(f"  {family:26s} {experiment_label(experiment):34s} {n:2d} member(s): "
                  f"{change.attrs['last_years']} minus {change.attrs['first_years']}")
    return out


def ocean_cell_area(root=OHC_TSERIES_ROOT):
    """POP T-cell area (any unit; only used as a weight), NaN over land, from the ancillary file."""
    anc = xr.open_dataset(os.path.join(root, "ancillary_files", "ohc_ancillary_data.nc"))
    return anc["TAREA"].where(anc["KMT"] > 0)


def check_change_maps(maps, area):
    """Ocean-area-weighted mean of each total-OHC change map vs the change in OHC_global_mean over the same years."""
    rows = []
    for (family, experiment), change in maps.items():
        total = change.sel(ohc_depth=-1)
        map_mean = float((total * area).sum() / area.where(total.notnull()).sum())
        gm = change.attrs["global_mean_change"]
        rows.append(dict(family=family, experiment=experiment_label(experiment),
                         map_mean_GJ_m2=map_mean / GJ, global_mean_change_GJ_m2=gm / GJ,
                         relative_difference=(map_mean - gm) / abs(gm)))
    return pd.DataFrame(rows).set_index(["family", "experiment"])


def select_map_layer(change, layer="total"):
    """'total' (full column) or one of LAYERS; hyphens are accepted in place of en dashes ('0-300 m')."""
    layer = layer.replace("-", "–")
    if layer not in MAP_LAYERS:
        raise ValueError(f"layer must be one of {MAP_LAYERS}, got {layer!r}")
    if layer == "total":
        return change.sel(ohc_depth=-1, drop=True)
    return layer_changes(change).sel(layer=layer, drop=True)


def make_regridder(template):
    """Bilinear regridder from the POP T-grid (TLAT/TLONG of `template`) to a regular 1-degree global grid. The
    curvilinear, displaced-pole POP grid can't be contoured directly without seams; a regular grid can."""
    import xesmf as xe

    grid_in = xr.Dataset({"lat": (("nlat", "nlon"), template["TLAT"].values),
                          "lon": (("nlat", "nlon"), template["TLONG"].values)})
    return xe.Regridder(grid_in, xe.util.grid_global(1, 1), "bilinear", periodic=True,
                        ignore_degenerate=True, unmapped_to_nan=True)


def _nice_ceiling(x):
    exponent = np.floor(np.log10(x))
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * 10**exponent >= x:
            return step * 10**exponent


def _contour_on_map(ax, field, regridder, levels):
    """Regrid a native-grid field to 1 degree and draw it as filled contours with land and coastlines."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    from matplotlib.colors import LinearSegmentedColormap

    regular = regridder(field, skipna=True, na_thres=0.5)
    lon, lat = regular["lon"].values[0, :], regular["lat"].values[:, 0]
    data, lon_cyclic = add_cyclic_point(regular.values, coord=lon)
    cmap = LinearSegmentedColormap.from_list("ohc_diverging", DIVERGING_COLORS)
    # transform_first: project the grid points, then contour in map space. Contouring in lon/lat and reprojecting
    # the polygons leaves large blank or mis-filled areas when one contour level covers much of the ocean.
    lon2d, lat2d = np.meshgrid(lon_cyclic, lat)
    filled = ax.contourf(lon2d, lat2d, data, levels=levels, cmap=cmap, extend="both",
                         transform=ccrs.PlateCarree(), transform_first=True)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=2)
    ax.coastlines(resolution="110m", color=AXIS, linewidth=0.5, zorder=3)
    ax.set_global()
    return filled


def _style_colorbar(cbar, label):
    cbar.set_label(label, color=INK2)
    cbar.ax.tick_params(colors=INK2, labelsize=8.5)
    cbar.outline.set_edgecolor(AXIS)


def plot_change_map(ax, change, regridder, family, experiment, layer="total", vmax=None, n_levels=21):
    """Filled-contour global map (cartopy) of one experiment's OHC change, last minus first n years, in GJ m-2.
    `ax` must be a cartopy GeoAxes. The color scale is symmetric about zero; by default it spans the 98th
    percentile of |change| for this map (values beyond it fall in the arrow-shaped end bins)."""
    import matplotlib.pyplot as plt

    layer = layer.replace("-", "–")
    field = select_map_layer(change, layer) / GJ
    vmax = vmax or _nice_ceiling(float(np.nanpercentile(np.abs(field), 98)))
    filled = _contour_on_map(ax, field, regridder, np.linspace(-vmax, vmax, n_levels))
    cbar = plt.colorbar(filled, ax=ax, orientation="horizontal", shrink=0.65, pad=0.04, aspect=40,
                        ticks=np.linspace(-vmax, vmax, 5))
    _style_colorbar(cbar, f"OHC change, {layer} [GJ m⁻²]")
    a = change.attrs
    runs = "1 run" if a["n_members"] == 1 else f"mean of {a['n_members']} members"
    ax.set_title(f"{family}: {experiment_label(experiment)}\n{layer}, {a['last_years']} minus {a['first_years']} ({runs})",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    return filled


# --- Scenario period vs. a baseline period from another experiment -----------------------------------------------

def find_experiment(experiments, family, label):
    """The (family, experiment) key whose short label (experiment_label) is `label`."""
    matches = [key for key in experiments if key[0] == family and experiment_label(key[1]) == label]
    if len(matches) != 1:
        raise KeyError(f"expected one experiment {label!r} in {family!r}, found {len(matches)}")
    return matches[0]


def ensemble_period_mean(cases, experiments, key, years):
    """Member-mean OHC per column area [J m-2] over the calendar `years` (every member must cover them all),
    (ohc_depth, nlat, nlon); attr global_mean is the members' mean OHC_global_mean (full column) over the same
    months, for checking the map."""
    family, experiment = key
    members = experiments[key]["member"].values
    rows = cases[(cases.family == family) & (cases.experiment == experiment) & cases.member.isin(members)]
    maps, global_means = [], []
    for row in rows.itertuples():
        ds = xr.open_mfdataset(sorted(row.files), combine="by_coords", data_vars="minimal", coords="minimal",
                               compat="override", use_cftime=True, chunks={})
        _check_monthly_axis(row.case, ds["time"])
        maps.append(_window_mean(ds["OHC"], years, row.case).load())
        global_means.append(float(_window_mean(ds["OHC_global_mean"].sel(ohc_depth=-1), years, row.case)))
        ds.close()
    mean = xr.concat(maps, dim="member").mean("member")
    mean.attrs = dict(units="J m-2", years=f"{years[0]}–{years[-1]}", n_members=len(maps),
                      global_mean=float(np.mean(global_means)))
    return mean


def difference_maps(cases, experiments, scenario_keys, scenario_years, baseline_key, baseline_years):
    """Map of (scenario mean over scenario_years) minus (baseline experiment mean over baseline_years) [J m-2] for
    each scenario: ({key: DataArray (ohc_depth, nlat, nlon)}, baseline mean). Each map carries attr
    global_mean_change, so check_change_maps can verify it."""
    base = ensemble_period_mean(cases, experiments, baseline_key, list(baseline_years))
    runs = "1 run" if base.attrs["n_members"] == 1 else f"mean of {base.attrs['n_members']}"
    base.attrs["label"] = f"{experiment_label(baseline_key[1])} {base.attrs['years']} ({runs})"
    out = {}
    for key in scenario_keys:
        scen = ensemble_period_mean(cases, experiments, key, list(scenario_years))
        diff = scen - base
        diff.attrs = dict(units="J m-2", years=scen.attrs["years"], n_members=scen.attrs["n_members"],
                          baseline=base.attrs["label"],
                          global_mean_change=scen.attrs["global_mean"] - base.attrs["global_mean"])
        out[key] = diff
    return out, base


def plot_difference_panels(fig, axes, diffs, regridder, layer="total", vmax=None, n_levels=21):
    """One cartopy map per scenario in `diffs` on the GeoAxes in `axes`, all on a single shared, zero-centered
    color scale (by default spanning the 98th percentile of |difference| over all panels) with one colorbar."""
    layer = layer.replace("-", "–")
    fields = {key: select_map_layer(diff, layer) / GJ for key, diff in diffs.items()}
    vmax = vmax or _nice_ceiling(max(float(np.nanpercentile(np.abs(f), 98)) for f in fields.values()))
    levels = np.linspace(-vmax, vmax, n_levels)
    for ax, (key, field) in zip(np.ravel(axes), fields.items()):
        filled = _contour_on_map(ax, field, regridder, levels)
        a = diffs[key].attrs
        runs = "1 run" if a["n_members"] == 1 else f"mean of {a['n_members']} members"
        ax.set_title(f"{key[0]}: {experiment_label(key[1])}\n{a['years']} ({runs})",
                     loc="left", fontsize=10.5, fontweight="bold", color=INK)
    cbar = fig.colorbar(filled, ax=np.ravel(axes).tolist(), orientation="horizontal", shrink=0.55, aspect=50,
                        ticks=np.linspace(-vmax, vmax, 5))
    baseline = next(iter(diffs.values())).attrs["baseline"]
    _style_colorbar(cbar, f"OHC change, {layer}, relative to {baseline} [GJ m⁻²]")
    return filled


# --- Surface temperature and cross-experiment summary figures -----------------------------------------------------

ATM_TSERIES_ROOT = PRISM_DATA + "spatial_averages_data/"


def _shift_back_one_month(times):
    """CESM stamps each monthly mean at the end of its averaging period (January's mean is labeled 1 February);
    relabel each stamp with the first day of the month the data describe."""
    out = []
    for t in times:
        year, month = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
        out.append(type(t)(year, month, 1))
    return np.array(out)


def load_case_ts(family, case, root=ATM_TSERIES_ROOT, var="TS", region="G"):
    """Monthly global-mean surface temperature [K] for one case from PRISM's spatial-average files.

    The files use CESM's end-of-period time stamps, so they are shifted back one month (the record must then start
    in January). The ARISE-1.0 yearly files repeat the month at each file boundary; the repeats are checked to hold
    the same values before one copy is kept.
    """
    pattern = os.path.join(root, family, "**", f"*{case}.cam.h0.{var}.*.nc")  # exact ".TS." (not e.g. "TSMLT")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no {var} files match {pattern}")
    da = xr.concat([xr.open_dataset(f, use_cftime=True)[var].sel(spatial=region, drop=True).load() for f in files],
                   dim="time")
    keys = np.array([t.year * 12 + t.month for t in da["time"].values])
    order = np.argsort(keys, kind="stable")
    da, keys = da.isel(time=order), keys[order]
    repeats = np.flatnonzero(np.diff(keys) == 0)
    if repeats.size and not np.allclose(da.values[repeats], da.values[repeats + 1]):
        raise ValueError(f"{case}: files repeat months with different {var} values")
    da = da.isel(time=np.r_[0, np.flatnonzero(np.diff(keys) != 0) + 1])
    da = da.assign_coords(time=_shift_back_one_month(da["time"].values))
    _check_monthly_axis(case, da["time"])
    if da["time"].values[0].month != 1:
        raise ValueError(f"{case}: after the end-of-period shift the record starts in month "
                         f"{da['time'].values[0].month}, not January -- check the time-stamp convention")
    return da


def load_experiment_ts(cases, experiments, key, root=ATM_TSERIES_ROOT):
    """Month-length-weighted annual global-mean surface temperature [K], (member, year), for the same members
    load_experiments kept for this experiment."""
    family, experiment = key
    members = experiments[key]["member"].values
    rows = cases[(cases.family == family) & (cases.experiment == experiment) & cases.member.isin(members)]
    annual = [annual_mean(load_case_ts(family, row.case, root)).expand_dims(member=[row.member])
              for row in rows.itertuples()]
    da = xr.concat(annual, dim="member", join="outer")
    da.attrs = dict(units="K", long_name="Annual global-mean surface temperature")
    return da


def change_since(da, start, end):
    """Years start..end of `da` (any dims including member and year), relative to each member's value in `start`,
    so every member starts at 0. Every member must cover the whole period."""
    sub = da.sel(year=slice(start, end))
    if sub.sizes["year"] == 0 or int(sub["year"][0]) != start or int(sub["year"][-1]) != end:
        raise ValueError(f"data cover {int(da.year[0])}–{int(da.year[-1])}, not all of {start}–{end}")
    if bool(sub.isnull().any()):
        raise ValueError(f"not every member covers {start}–{end}")
    return sub - sub.sel(year=start)


def plot_layer_grid(fig, axes, items, start, end):
    """Stacked-layer OHC panels (plot_layers) on shared axes, one per condition, with one legend for the figure.
    items: [(name, (family, experiment), da (member, ohc_depth, year) already relative to `start`), ...]."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    flat = np.ravel(axes)
    for ax, (name, (family, experiment), da) in zip(flat, items):
        n = da.sizes["member"]
        runs = "1 run" if n == 1 else f"mean of {n} members"
        plot_layers(ax, da, family, experiment, title=name, legend=False, pad_xlim=False)
        ax.set_title(name, pad=16)
        ax.text(0, 1.012, f"{experiment_label(experiment)} · {runs}", transform=ax.transAxes, va="bottom",
                color=INK2, fontsize=8.5)
    for ax in flat[len(items):]:
        ax.set_visible(False)
    flat[0].set_xlim(start, end + 0.16 * (end - start))  # shared x: room for the end labels
    for ax in flat[:len(items)]:
        ax.label_outer()
    handles = [Line2D([], [], color=INK, lw=2.2, label="total (ensemble mean for ensembles)")]
    if any(da.sizes["member"] > 1 for _, _, da in items):
        handles.append(Line2D([], [], color=MUTED, lw=0.8, label="total, member min–max"))
    handles += [Patch(color=c, label=l) for l, c in zip(LAYERS, LAYER_COLORS)]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=9)


def window_means(series, start, end, window=10):
    """Mean of a (year) series over start..end, and over consecutive non-overlapping `window`-year windows
    starting at `start` (a final partial window is left out): (period_mean, [((first, last), mean), ...])."""
    period = float(series.sel(year=slice(start, end)).mean())
    windows = [(y, y + window - 1) for y in range(start, end + 1, window) if y + window - 1 <= end]
    return period, [(w, float(series.sel(year=slice(*w)).mean())) for w in windows]



class _LabelSpace:
    """Free-space bookkeeping for direct labels on one axes, in points (1/72 in). Obstacles are boxes: markers,
    sampled line segments, legends, text already placed. A label is only put where its box clears every obstacle
    and stays inside the axes."""

    def __init__(self, ax):
        ax.figure.canvas.draw()  # settle limits and layout so data -> screen positions are final
        self.ax = ax
        self.to_points = 72.0 / ax.figure.dpi
        self.renderer = ax.figure.canvas.get_renderer()
        b = ax.get_window_extent(self.renderer)
        self.bounds = tuple(v * self.to_points for v in (b.x0, b.y0, b.x1, b.y1))
        self.boxes = []  # (x0, y0, width, height)
        self.texts = []  # boxes of placed labels only, for leader checks
        self.leaders = []  # ((x0, y0), (x1, y1)) leader lines already drawn

    def screen(self, x, y):
        return tuple(self.ax.transData.transform((x, y)) * self.to_points)

    def add_markers(self, xs, ys, size):
        """Scatter markers of area `size` (points^2) at data positions xs, ys."""
        half = 0.5 * np.sqrt(size) + 1.0
        for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
            px, py = self.screen(x, y)
            self.boxes.append((px - half, py - half, 2 * half, 2 * half))

    def add_line(self, xs, ys, half=1.5, step=3.0):
        """A polyline through data points xs, ys, sampled every `step` points along its screen length."""
        pts = np.array([self.screen(x, y) for x, y in zip(xs, ys)])
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step) + 1)
            for t in np.linspace(0, 1, n):
                px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                self.boxes.append((px - half, py - half, 2 * half, 2 * half))

    def add_artist(self, artist):
        b = artist.get_window_extent(self.renderer)
        self.boxes.append((b.x0 * self.to_points, b.y0 * self.to_points,
                           b.width * self.to_points, b.height * self.to_points))

    def clear(self, x0, y0, width, height):
        bx0, by0, bx1, by1 = self.bounds
        inside = bx0 <= x0 and x0 + width <= bx1 and by0 <= y0 and y0 + height <= by1
        return inside and not any(x0 < bx + bw and bx < x0 + width and y0 < by + bh and by < y0 + height
                                  for bx, by, bw, bh in self.boxes)

    def leader_clear(self, start, box):
        """True if a leader from `start` to the edge of label `box` crosses no other leader and no other label."""
        x0, y0, w, h = box
        end = (min(max(start[0], x0), x0 + w), min(max(start[1], y0), y0 + h))  # nearest point of the box
        n = max(2, int(np.hypot(end[0] - start[0], end[1] - start[1]) / 2.0))
        for t in np.linspace(0, 1, n)[1:-1]:
            px, py = start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])
            if any(bx < px < bx + bw and by < py < by + bh for bx, by, bw, bh in self.texts):
                return False

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

        for a, b in self.leaders:
            if (ccw(start, end, a) * ccw(start, end, b) < 0) and (ccw(a, b, start) * ccw(a, b, end) < 0):
                return False
        return True

    def add_label(self, box, leader_start=None):
        """Record a placed label (and its leader, if any). Returns the leader's attachment point on the label as
        a fraction of the box, for annotate's arrowprops relpos, so the drawn leader is the one that was checked."""
        self.boxes.append(box)
        self.texts.append(box)
        if leader_start is None:
            return None
        x0, y0, w, h = box
        end = (min(max(leader_start[0], x0), x0 + w), min(max(leader_start[1], y0), y0 + h))
        self.leaders.append((leader_start, end))
        return ((end[0] - x0) / w, (end[1] - y0) / h)

    @staticmethod
    def size(text, fontsize, bold=False):
        return (0.62 if bold else 0.58) * fontsize * len(text), 1.25 * fontsize


def end_periods(start, end, length=5):
    """The first and last `length` years of start..end: (base, final), e.g. ((2035, 2039), (2065, 2069))."""
    return (start, start + length - 1), (end - length + 1, end)


def period_mean(da, years):
    """Mean over the inclusive (first, last) year range of a (..., year) array."""
    return da.sel(year=slice(*years)).mean("year")


def period_difference(da, base, final):
    """Mean over the `final` years minus mean over the `base` years (inclusive (first, last) ranges)."""
    return period_mean(da, final) - period_mean(da, base)


def _place_point_labels(ax, points, fontsize=9, gap=1.5, space=None):
    """Label each (x, y, text) next to its point, clear of the obstacles in `space` (a _LabelSpace; by default
    only the labeled points, as markers of area 190). Tries up-right, down-right, up-left and down-left of the
    point, then the same four spots stepped further out, nearest first; a label stepped out gets a thin leader line
    back to its point.
    Returns the space, with the labels added as obstacles."""
    if space is None:
        space = _LabelSpace(ax)
        space.add_markers([p[0] for p in points], [p[1] for p in points], 190)
    for x, y, text in sorted(points, key=lambda p: -p[1]):
        px, py = space.screen(x, y)
        width, height = space.size(text, fontsize)
        candidates = [(k, cx, cy) for k in range(40) for step in [k * (height + gap)]
                      for cx, cy in ((9.0, 5.0 + step), (9.0, -5.0 - height - step),
                                     (-9.0 - width, 5.0 + step), (-9.0 - width, -5.0 - height - step))]
        k, dx, dy = next(((k, cx, cy) for k, cx, cy in candidates if space.clear(px + cx, py + cy, width, height)
                          and (k == 0 or space.leader_clear((px, py), (px + cx, py + cy, width, height)))),
                         candidates[0])
        relpos = space.add_label((px + dx, py + dy, width, height), (px, py) if k > 0 else None)
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", va="bottom", ha="left",
                    color=INK2, fontsize=fontsize,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=6, relpos=relpos)
                    if k > 0 else None)
    return space


def _place_line_label(ax, space, x_lo, x_hi, slope, intercept, text, fontsize=10, bold=False, others=(),
                      fractions=(0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85), min_separation=14.0):
    """Label the line y = slope * x + intercept (drawn over x_lo..x_hi) with a callout: the text sits in free space
    above-left or below-right of the line, with a thin leader to an anchor on the line. Anchors are only taken where
    every line in `others` ([(x_lo, x_hi, slope, intercept), ...]) is at least `min_separation` points away, so
    the leader can only point at this line; nearer placements are tried first."""
    width, height = space.size(text, fontsize, bold)

    def far_from_others(px, py):
        for lo, hi, m, c in others:
            xs = np.linspace(lo, hi, 60)
            pts = np.array([space.screen(xx, m * xx + c) for xx in xs])
            if np.min(np.hypot(pts[:, 0] - px, pts[:, 1] - py)) < min_separation:
                return False
        return True

    anchors = []
    for frac in fractions:
        x = x_lo + frac * (x_hi - x_lo)
        px, py = space.screen(x, slope * x + intercept)
        if far_from_others(px, py):
            anchors.append((x, px, py))
    for d in (10.0, 22.0, 36.0, 52.0, 70.0, 90.0):
        for x, px, py in anchors:
            for dx, dy in ((-d - width, d), (d, -d - height)):
                box = (px + dx, py + dy, width, height)
                if space.clear(*box) and space.leader_clear((px, py), box):
                    relpos = space.add_label(box, (px, py))
                    return ax.annotate(text, xy=(x, slope * x + intercept), xytext=(dx, dy),
                                       textcoords="offset points", ha="left", va="bottom", color=INK2,
                                       fontsize=fontsize, fontweight="bold" if bold else "normal",
                                       arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=0,
                                                       relpos=relpos))
    raise RuntimeError(f"no free spot for line label {text!r}")


def plot_uptake_vs_warming(ax, items, start, end, window=10, styles=None, members=False, window_size=38,
                           window_markers=None, base=None, final=None):
    """Ensemble-mean surface-temperature change vs. ocean heat uptake, both measured from the mean over the `base`
    years (default: the first 5 years of start..end).

    items: [(name, ohc (member, year) [J], ts (member, year) [K]), ...], on any common zero point (it cancels).
    The large point for each condition is the mean over the `final` years (default: the last 5) minus the mean
    over the `base` years; the points of area `window_size` are the means over each consecutive `window`-year
    window from `start`, minus the same base mean, joined in time order. Large points are labeled directly so identity never rests on
    color alone. styles: optional {name: {"color": ..., "marker": ...}} so a condition looks the same in every
    figure; by default conditions take palette slots in order, as circles.
    window_markers: optional marker per window, in time order (e.g. ["^", "s", "D"]), so the windows are told
    apart by shape; by default every point takes the condition's marker.
    members: also plot each ensemble member's own final-minus-base difference and window means as small, faded,
    unjoined points (area 38) in the same shape as the ensemble-mean point they belong to, and add the member
    standard deviation of both quantities to the returned table.
    """
    from matplotlib.lines import Line2D

    MEAN_SIZE, MEMBER_SIZE, MEMBER_ALPHA = 190, 38, 0.45
    default_base, default_final = end_periods(start, end)
    base, final = base or default_base, final or default_final
    summary, labels, paths, member_xy = [], [], [], []
    for slot, (name, ohc, ts) in enumerate(items):
        style = (styles or {}).get(name, {})
        color, marker = style.get("color", PALETTE[slot % len(PALETTE)]), style.get("marker", "o")
        ohc = (ohc - period_mean(ohc, base)) / ZJ  # every quantity on this plot is measured from the base mean
        ts = ts - period_mean(ts, base)
        x_all = float(period_difference(ohc.mean("member"), base, final))
        y_all = float(period_difference(ts.mean("member"), base, final))
        x_win = window_means(ohc.mean("member"), start, end, window)[1]
        y_win = window_means(ts.mean("member"), start, end, window)[1]
        periods = [w for w, _ in x_win]
        shapes = list(window_markers) if window_markers is not None else [marker] * len(periods)
        if len(shapes) < len(periods):
            raise ValueError(f"window_markers has {len(shapes)} markers for {len(periods)} windows")
        xs, ys = [v for _, v in x_win], [v for _, v in y_win]
        row = dict(condition=name, members=ohc.sizes["member"], ocean_heat_uptake_ZJ=x_all,
                   surface_temperature_change_K=y_all)
        if members:
            for (a, b), shape in [(final, marker)] + list(zip(periods, shapes)):
                mx = period_mean(ohc, (a, b)).values  # for `final` this is final minus base (ohc, ts are rebased)
                my = period_mean(ts, (a, b)).values
                ax.scatter(mx, my, s=MEMBER_SIZE, color=color, marker=shape, alpha=MEMBER_ALPHA, linewidths=0,
                           zorder=1)
                member_xy.append((mx, my))
                suffix = "" if (a, b) == final else f"_{a}-{b}"
                row.update({f"uptake_ZJ_member_sd{suffix}": float(np.std(mx, ddof=1)) if mx.size > 1 else np.nan,
                            f"dTS_K_member_sd{suffix}": float(np.std(my, ddof=1)) if my.size > 1 else np.nan})
        ax.plot(xs, ys, color=color, lw=1, zorder=2)
        for x, y, shape in zip(xs, ys, shapes):
            ax.scatter(x, y, s=window_size, color=color, marker=shape, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.scatter(x_all, y_all, s=MEAN_SIZE, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=4,
                   label=name)
        labels.append((x_all, y_all, name))
        paths.append((xs, ys))
        row.update({f"uptake_ZJ_{a}-{b}": v for (a, b), v in x_win})
        row.update({f"dTS_K_{a}-{b}": v for (a, b), v in y_win})
        summary.append(row)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    mean_word = "ensemble mean" if members else "mean"
    key = [Line2D([], [], ls="", marker="o", ms=np.sqrt(MEAN_SIZE), color=MUTED, mec=SURFACE,
                  label=f"{final[0]}–{final[1]} {mean_word} minus {base[0]}–{base[1]} {mean_word}")]
    if window_markers is not None:
        key += [Line2D([], [], ls="-", lw=1, marker=shape, ms=np.sqrt(window_size), color=MUTED, mec=SURFACE,
                       label=f"{a}–{b} {mean_word}") for (a, b), shape in zip(periods, shapes)]
    else:
        windows = ", ".join(f"{a}–{b}" for a, b in periods)
        key.append(Line2D([], [], ls="-", lw=1, marker="o", ms=np.sqrt(window_size), color=MUTED, mec=SURFACE,
                          label=f"{window}-year {mean_word}s ({windows})"))
    if members:
        key.append(Line2D([], [], ls="", marker="o", ms=np.sqrt(MEMBER_SIZE), color=MUTED, alpha=MEMBER_ALPHA,
                          mew=0, label="one ensemble member (faded; same shape as its ensemble mean)"))
    first_legend = ax.legend(loc="upper left", fontsize=9, title="Condition", title_fontsize=9)
    ax.add_artist(first_legend)
    key_legend = ax.legend(handles=key, loc="lower right", fontsize=9)
    ax.set(xlabel=f"Ocean heat uptake from the {base[0]}–{base[1]} mean [ZJ]",
           ylabel=f"Surface temperature change from the {base[0]}–{base[1]} mean [K]")

    space = _LabelSpace(ax)
    space.add_markers([p[0] for p in labels], [p[1] for p in labels], MEAN_SIZE)
    for xs, ys in paths:
        space.add_markers(xs, ys, window_size)
        space.add_line(xs, ys)
    for mx, my in member_xy:
        space.add_markers(mx, my, MEMBER_SIZE)
    space.add_artist(first_legend)
    space.add_artist(key_legend)
    _place_point_labels(ax, labels, space=space)
    return pd.DataFrame(summary).set_index("condition")


def rolling_slope(da, window=5):
    """Least-squares slope (per year) of `da` over the `window` years centered on each year (window must be odd).
    Years without a full window at either end are dropped. Needs consecutive annual years. For equally spaced
    years the OLS slope is a fixed weighted sum, sum_k (k - kbar) y_k / sum_k (k - kbar)^2, applied here as a
    rolling dot product; being linear, the slope of an ensemble mean equals the mean of the member slopes."""
    if window % 2 != 1:
        raise ValueError(f"window must be odd so it centers on a year, got {window}")
    years = da["year"].values
    if not (np.diff(years) == 1).all():
        raise ValueError("rolling_slope needs consecutive annual years")
    offsets = np.arange(window) - window // 2
    weights = xr.DataArray(offsets / (offsets ** 2).sum(), dims="window")
    windows = da.rolling(year=window, center=True).construct("window")
    half = window // 2
    return (windows * weights).sum("window", skipna=False).isel(year=slice(half, -half))


def plot_annual_paths(fig, axes, items, start, end, cmap="viridis", rate_window=None):
    """One panel per condition: annual surface-temperature change vs. ocean heat uptake, both since `start`.
    The ensemble-mean year-by-year points are colored by year (one colorbar shared by every panel); each member's
    own annual path is a thin, semi-transparent line in one neutral color behind them (no lines for a single-run
    condition, whose points are the run itself).
    items: [(name, ohc_change (member, year) [J], ts_change (member, year) [K]), ...] with change_since-style
    data (0 at `start`).
    rate_window: plot rates instead of totals: each year's least-squares slope over the `rate_window` years
    centered on it (rolling_slope), as ocean heat uptake rate [ZJ/yr] vs. surface warming rate [K/decade]; the
    first and last rate_window // 2 years have no full window and are left out. Colors keep the start..end scale,
    so a year has the same color as in the totals figure. Returns the colorbar."""
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    norm = Normalize(start, end)
    any_ensemble = False
    for ax, (name, ohc, ts) in zip(np.ravel(axes), items):
        ohc = ohc.sel(year=slice(start, end)) / ZJ
        ts = ts.sel(year=slice(start, end))
        assert (ohc["year"].values == ts["year"].values).all(), f"{name}: OHC and TS years differ"
        if rate_window:
            ohc, ts = rolling_slope(ohc, rate_window), 10 * rolling_slope(ts, rate_window)  # ZJ/yr, K/decade
        n = ohc.sizes["member"]
        any_ensemble |= n > 1
        for m in range(n if n > 1 else 0):
            ax.plot(ohc.isel(member=m), ts.isel(member=m), color=MUTED, lw=0.7, alpha=0.45, zorder=1)
        points = ax.scatter(ohc.mean("member"), ts.mean("member"), c=ohc["year"], cmap=cmap, norm=norm, s=30,
                            edgecolors=SURFACE, linewidths=0.6, zorder=3)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(f"{name} (mean of {n})" if n > 1 else name, loc="left", fontweight="bold")
        if rate_window:
            ax.set(xlabel="Ocean heat uptake rate [ZJ/yr]", ylabel="Surface warming rate [K/decade]")
        else:
            ax.set(xlabel=f"Ocean heat uptake since {start} [ZJ]",
                   ylabel=f"Surface temperature change since {start} [K]")
        ax.label_outer()
    what = f"{rate_window}-year centered trend" if rate_window else "annual value"
    sizes = {it[1].sizes["member"] for it in items}
    mean_word = "ensemble mean" if min(sizes) > 1 else ("ensemble mean (or the single run)" if max(sizes) > 1
                                                         else "single run")
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=MUTED, mec=SURFACE,
                      label=f"{mean_word}, one point per year: {what} (color = year)")]
    if any_ensemble:
        handles.append(Line2D([], [], color=MUTED, lw=0.7, alpha=0.45, label="one ensemble member, year-by-year path"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=9)
    return fig.colorbar(points, ax=axes, label="Year", pad=0.01, fraction=0.025)


def _spread(values, gap):
    """Positions for labels at `values` that are at least `gap` apart, each as close to its value as possible."""
    order = np.argsort(values)
    pos = np.asarray(values, dtype=float)[order]
    for _ in range(100):  # push overlapping neighbours apart symmetrically until none overlap
        moved = False
        for i in range(1, pos.size):
            overlap = gap - (pos[i] - pos[i - 1])
            if overlap > 1e-9:
                pos[i - 1] -= overlap / 2
                pos[i] += overlap / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


def plot_total_comparison(ax, items, start, end):
    """Ensemble-mean total OHC change since `start` for several conditions on one axes, each labeled at its line
    end (labels are spread apart where lines end close together, with a leader line back to the line).
    items: [(name, change (member, year) [J], {"color": ..., "linestyle": ...}), ...]."""
    ends = []
    for name, change, style in items:
        n = change.sizes["member"]
        mean = change.mean("member") / ZJ
        ax.plot(mean["year"], mean, color=style["color"], ls=style.get("linestyle", "-"), lw=2,
                label=name if n == 1 else f"{name} (mean of {n})")
        ends.append((float(mean[-1]), f"{name}  {float(mean[-1]):+.0f} ZJ"))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlim(start, end + 0.32 * (end - start))
    y_lo, y_hi = ax.get_ylim()
    label_y = _spread([y for y, _ in ends], gap=0.042 * (y_hi - y_lo))
    for (y, text), y_text in zip(ends, label_y):
        ax.annotate(text, xy=(end, y), xytext=(end + 0.03 * (end - start), y_text), va="center", ha="left",
                    color=INK2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=2))
    ax.set(xlabel="Year", ylabel=f"OHC change since {start} [ZJ]")


def _fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return dict(slope_K_per_ZJ=float(slope), intercept_K=float(intercept),
                r2=float(np.corrcoef(x, y)[0, 1] ** 2), n=len(x), x_min=float(np.min(x)), x_max=float(np.max(x)))


def plot_period_mean_regression(ax, items, start, end, family_markers=None, family_fits=None, base=None, final=None):
    """Each condition's ensemble-mean surface-temperature change vs. ocean heat uptake, both the mean over the
    `final` years minus the mean over the `base` years (defaults: the last and first 5 years of start..end), with a dashed least-squares line through all the points and, for each group in
    `family_fits`, a thinner dashed line through that group's points alone. Every line is labeled on the plot with
    its r² and intercept, and listed in the legend with its slope, intercept and r².
    items: [(name, ohc_change (member, year) [J], ts_change (member, year) [K], {"color", "marker"}), ...].
    family_markers: optional {legend label: marker} explaining what the marker shapes mean.
    family_fits: optional {short name: marker}; fits the points drawn with that marker separately.
    Returns (points DataFrame, fits DataFrame indexed by fit name: slope [K per ZJ], intercept [K], r2, n, x range)."""
    from matplotlib.lines import Line2D

    MEAN_SIZE = 190
    default_base, default_final = end_periods(start, end)
    base, final = base or default_base, final or default_final
    rows = []
    for name, ohc, ts, style in items:
        x = float(period_difference(ohc.mean("member"), base, final)) / ZJ
        y = float(period_difference(ts.mean("member"), base, final))
        marker = style.get("marker", "o")
        ax.scatter(x, y, s=MEAN_SIZE, color=style["color"], marker=marker, edgecolors=SURFACE, linewidths=2,
                   zorder=4)
        rows.append(dict(condition=name, marker=marker, members=ohc.sizes["member"], ocean_heat_uptake_ZJ=x,
                         surface_temperature_change_K=y))
    points = pd.DataFrame(rows).set_index("condition")
    X, Y = "ocean_heat_uptake_ZJ", "surface_temperature_change_K"

    groups = [(f"all {len(points)} conditions", points, dict(ls="--", lw=1.6))]
    thin_dashes = [(0, (2.5, 2)), (0, (8, 3))]  # short dashes for the first family, long for the second
    for (short, marker), dashes in zip((family_fits or {}).items(), thin_dashes):
        sub = points[points["marker"] == marker]
        if len(sub) < 3:
            raise ValueError(f"{short}: {len(sub)} points; need at least 3 for a meaningful fit and r²")
        groups.append((f"{short} ({len(sub)} conditions)", sub, dict(ls=dashes, lw=0.9)))

    fits, lines, handles = {}, [], []
    for label, sub, line_style in groups:
        f = _fit(sub[X].values, sub[Y].values)
        fits[label] = f
        xx = np.array([f["x_min"], f["x_max"]])
        ax.plot(xx, f["slope_K_per_ZJ"] * xx + f["intercept_K"], color=INK2, zorder=3, **line_style)
        lines.append((label, f))
        handles.append(Line2D([], [], color=INK2, **line_style,
                              label=f"fit, {label}: slope {100 * f['slope_K_per_ZJ']:+.2f} K per 100 ZJ, "
                                    f"intercept {f['intercept_K']:+.2f} K, r² = {f['r2']:.2f}"))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    shapes = [Line2D([], [], ls="", marker=m, ms=11, color=MUTED, mec=SURFACE, label=lbl)
              for lbl, m in (family_markers or {}).items()]
    legend = ax.legend(handles=shapes + handles, loc="upper left", fontsize=8.5)
    ax.set(xlabel=f"Ocean heat uptake, {final[0]}–{final[1]} minus {base[0]}–{base[1]} mean [ZJ]",
           ylabel=f"Surface temperature change, {final[0]}–{final[1]} minus {base[0]}–{base[1]} mean [K]")

    # Direct labels: point names first, then one label per fit line, all clear of markers, lines and the legend
    space = _LabelSpace(ax)
    space.add_markers(points[X].values, points[Y].values, MEAN_SIZE)
    for _, f in lines:
        xx = np.array([f["x_min"], f["x_max"]])
        space.add_line(xx, f["slope_K_per_ZJ"] * xx + f["intercept_K"])
    space.add_artist(legend)
    _place_point_labels(ax, list(zip(points[X], points[Y], points.index)), space=space)
    segments = [(f["x_min"], f["x_max"], f["slope_K_per_ZJ"], f["intercept_K"]) for _, f in lines]
    # the all-conditions fit usually lies between the family fits, so label the family fits first: their free side
    # is open, and the all-conditions callout then routes around them
    for i, (label, f) in sorted(enumerate(lines), key=lambda item: item[0] == 0):
        name = label.split(" (")[0]
        _place_line_label(ax, space, *segments[i], f"{name} fit: r² = {f['r2']:.2f}, intercept {f['intercept_K']:+.2f} K",
                          fontsize=10 if i == 0 else 9, bold=(i == 0), others=segments[:i] + segments[i + 1:])
    return points.drop(columns="marker"), pd.DataFrame(fits).T.astype({"n": int})

