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


def plot_layers(ax, da, family, experiment):
    """Total OHC change of one experiment with the change in each depth layer shaded beneath it.

    da: (member, ohc_depth, year), as returned by load_experiments(depth=None). Ensembles show the member mean
    (total and layers); the members' min-max range of the total is drawn as thin lines over the years every member
    covers, and the mean continues thinner where only some members run on. Layers gaining heat stack upward from
    zero and layers losing heat stack downward, so the total equals the shaded area above zero minus that below.
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
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    runs = "1 run" if n == 1 else f"{n} members"
    ax.set(title=f"{family}: {experiment_label(experiment)} ({runs})",
           xlabel="Year" if not family.endswith(("1850control", "LME")) else "Model year",
           ylabel="OHC change since first year [ZJ]")
    return ax


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
