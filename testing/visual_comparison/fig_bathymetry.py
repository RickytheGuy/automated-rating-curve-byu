"""Bathymetry (arc.bathymetry): banks, channel depths, the carve, the smoothing along the network, and the raster."""
from __future__ import annotations

import collections
import math

import matplotlib.pyplot as plt
import numpy as np

from vc_plot import (ACCENT, BANK, GROUND, LEGACY, NEW, WATER, Window, between_banks, figure, hillshade,
                     legacy_stations, mark_banks, new_stations, note, rounded, show_raster, water)

WALL = 9999.0
TRAPEZOID_HEIGHT = 0.2  # the sites' Bathy_Trap_H


# --- Legacy's carve, on any cross section ----------------------------------------------------------------------------


def legacy_section(elevations: np.ndarray, spacing: float, *, use_banks: bool, trapezoid_height=TRAPEZOID_HEIGHT):
    """Legacy's CrossSection holding a new cross section (left to right): its left half as side 1 and its right half
    as side 2, each out from the stream cell to its first wall, with a DEM row holding the elevations (as the tests
    build it)."""
    from arc.cross_section import CrossSection
    size = elevations.size
    center = size // 2
    left, right = elevations[center::-1].copy(), elevations[center:].copy()
    ends = [int(np.flatnonzero(np.r_[side >= WALL, True])[0]) for side in (left, right)]
    params = {"d_x_section_distance": 4.0 * size * spacing, "b_FindBanksBasedOnLandCover": False,
              "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": trapezoid_height, "b_bathy_use_banks": use_banks,
              "d_degree_manipulation": 0.0, "d_degree_interval": 0.0, "i_boundary_number": 0, "nrows": 3,
              "ncols": size + 2, "s_output_bathymetry_path": None}
    dem = np.zeros((3, size + 2))
    dem[1, 1:size + 1] = np.where(elevations < WALL, elevations, np.nan)
    legacy = CrossSection(1.0, 1.0, dem, np.zeros((3, size + 2), dtype=np.uint8), np.zeros((3, size + 2), np.int64),
                          params)
    legacy.xs1_n, legacy.xs2_n = ends
    legacy.d_ordinate_dist = float(spacing)
    legacy.da_xs_profile1 = np.append(left[:ends[0]], 99999.9)
    legacy.da_xs_profile2 = np.append(right[:ends[1]], 99999.9)
    legacy.ia_xc_row1_index_main = np.ones(ends[0], dtype=np.int64)
    legacy.ia_xc_column1_index_main = center + 1 - np.arange(ends[0])
    legacy.ia_xc_row2_index_main = np.ones(ends[1], dtype=np.int64)
    legacy.ia_xc_column2_index_main = center + 1 + np.arange(ends[1])
    return legacy


def legacy_width_banks(legacy, width: float) -> tuple[int, int, int]:
    """Legacy's bank indices for a channel of a given width: a single cell (1, 1) if it's at most a spacing,
    otherwise its _find_bank_by_target_width."""
    if width <= legacy.d_ordinate_dist:
        return 1, 1, 1
    i1, i2, total = legacy._find_bank_by_target_width(width)
    return (1, 1, 1) if total <= 1 else (i1, i2, total)


def legacy_carve(elevations, spacing, i1, i2, depth, *, bank_elevation=None, trapezoid_height=TRAPEZOID_HEIGHT):
    """The cross section after legacy's carve (left to right), with bank indices i1 (left) and i2 (right): below the
    bank elevation with one (Calculate_Bathymetry_Based_on_RiverBank_Elevations), otherwise below the stream cell
    (Calculate_Bathymetry_Based_on_WSE_or_LC)."""
    use_banks = bank_elevation is not None
    legacy = legacy_section(elevations, spacing, use_banks=use_banks, trapezoid_height=trapezoid_height)
    total = max(i1 + i2 - 1, 1)
    result = {"function_used": "figure", "i_bank_1_index": int(i1), "i_bank_2_index": int(i2),
              "i_total_bank_cells": int(total), "bank_elev_1": float(legacy.da_xs_profile1[i1]),
              "bank_elev_2": float(legacy.da_xs_profile2[i2]), "is_valid": True, "bathymetry_depth": float(depth),
              "bathymetry_should_apply": True,
              "smoothed_bank_elevation": float(bank_elevation) if use_banks else math.nan}
    raster = np.full((3, elevations.size + 2), np.nan)
    if use_banks:
        legacy.Calculate_Bathymetry_Based_on_RiverBank_Elevations(raster, bank_search_result=result)
    else:
        legacy.Calculate_Bathymetry_Based_on_WSE_or_LC(raster, bank_search_result=result)
    out = elevations.copy()
    center = elevations.size // 2
    out[center - legacy.xs1_n + 1:center + 1] = legacy.da_xs_profile1[:legacy.xs1_n][::-1]
    out[center:center + legacy.xs2_n] = legacy.da_xs_profile2[:legacy.xs2_n]
    return out


def new_carve(elevations, spacing, banks, depth, *, bank_elevation=None, trapezoid_height=TRAPEZOID_HEIGHT):
    """A copy of the cross section with the new carve: (the XSection, with its profile, and which ordinates
    changed)."""
    from arc.bathymetry import carve_channel
    from arc.xsection.xsection import XSection
    xs = XSection(elevations.astype(np.float64).copy(), np.full(elevations.size, 0.035), float(spacing))
    changed = carve_channel(xs, banks, float(depth), trapezoid_height=trapezoid_height, bank_elevation=bank_elevation)
    return xs, changed


def polygon_area(stations, elevations, level, lo, hi):
    """The area between a polyline and a level above it, from station lo to hi, finely integrated."""
    x = np.linspace(lo, hi, 20001)
    y = np.interp(x, stations, elevations)
    return float(np.trapezoid(np.clip(level - y, 0.0, None), x))


def _profile_xy(xs):
    p = xs.profile
    return (np.asarray(p[0]), np.asarray(p[1])) if p is not None else (new_stations(xs.elevations.size,
                                                                                    xs.ordinate_distance),
                                                                       xs.elevations)


def _plot_ground(ax, stations, elevations, **kwargs):
    ax.plot(stations, np.where(np.asarray(elevations) < WALL, elevations, np.nan), **kwargs)


def _new_cells(ctx, site, config="as_configured"):
    capture = ctx.capture(config, "new", site)
    return [] if capture is None else [c for c in capture["cells"] if c is not None]


# --- Banks -------------------------------------------------------------------------------------------------------------


@figure("B1", "Where the banks are on a sampled cross section", "Banks")
def bank_positions(ctx):
    from arc.cross_section import _find_bank_using_width_to_depth_ratio
    site = ctx.detail_sites[0]
    cells = [c for c in _new_cells(ctx, site) if c["banks"].method == "width_to_depth_ratio"]
    rng = np.random.default_rng(4)
    picks = [cells[k] for k in rng.choice(len(cells), size=min(6, len(cells)), replace=False)]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    stats = []
    for ax, cell in zip(axes.flat, picks):
        e, s, banks = cell["found"], cell["spacing"], cell["banks"]
        center = e.size // 2
        left, right = e[center::-1], e[center:]
        n1 = int(np.flatnonzero(np.r_[left >= WALL, True])[0])
        n2 = int(np.flatnonzero(np.r_[right >= WALL, True])[0])
        i1, i2 = _find_bank_using_width_to_depth_ratio(e[center], left.copy(), right.copy(), n1, n2, s)
        x = new_stations(e.size, s)
        stage = min(banks.left_elevation, banks.right_elevation)
        lo, hi = between_banks(ax, max(banks.left, i1 * s), max(banks.right, i2 * s), s, curves=[(x, e)])
        water(ax, x, e, stage)
        _plot_ground(ax, x, e, color=GROUND, marker="o", ms=3, lw=1.2)
        mark_banks(ax, banks.left, banks.right, color=NEW, label=f"new: {banks.top_width:.0f} m apart")
        for k, xb in enumerate((-i1 * s, i2 * s)):
            ax.axvline(xb, color=LEGACY, lw=1.0, ls=":", label=f"legacy's bank ordinates: counted "
                       f"{(i1 + i2 - 1) * s:.0f} m" if k == 0 else None)
        ax.set_title(f"row {cell['row']}, column {cell['col']} ({s:.1f} m spacing)", fontsize=8)
        ax.legend(loc="upper center", fontsize=6.5)
        ax.set_xlabel("metres from the stream cell")
        ax.set_ylabel("elevation (m)")
        stats.append(dict(new=rounded(banks.top_width, 1), legacy=rounded((i1 + i2 - 1) * s, 1)))
    return fig, dict(caption=(
        f"Six sampled cross sections of {ctx.site_label(site)} whose banks the width-to-depth ratio found, drawn "
        "between their banks. Both codes pick the same stage, where the ratio of top width to depth stops falling "
        "(shaded water). The new banks are the water's edges at that stage, anywhere between ordinates. Legacy's "
        "were the last ordinates under water, and it counted its channel as bank_1 + bank_2 − 1 spacings wide."),
                 stats=dict(sections=stats))


@figure("B2", "Top widths: legacy's count and the distance between the banks", "Banks")
def top_widths(ctx):
    from arc.cross_section import _find_bank_using_width_to_depth_ratio
    old, new, spacing = [], [], []
    for site in ctx.sites:
        for cell in _new_cells(ctx, site):
            banks = cell["banks"]
            if banks.method != "width_to_depth_ratio":
                continue
            e, s = cell["found"], cell["spacing"]
            center = e.size // 2
            left, right = e[center::-1], e[center:]
            n1 = int(np.flatnonzero(np.r_[left >= WALL, True])[0])
            n2 = int(np.flatnonzero(np.r_[right >= WALL, True])[0])
            i1, i2 = _find_bank_using_width_to_depth_ratio(e[center], left.copy(), right.copy(), n1, n2, s)
            if i1 + i2 == 0:
                continue
            old.append((i1 + i2 - 1) * s)
            new.append(banks.top_width)
            spacing.append(s)
    old, new, spacing = map(np.array, (old, new, spacing))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    ax.loglog(old, new, ".", ms=2, color=NEW, alpha=0.4)
    lims = [max(min(old.min(), new.min()), 5.0), max(old.max(), new.max())]
    ax.plot(lims, lims, color=GROUND, lw=0.8)
    ax.set_xlabel("legacy's top width, (bank₁ + bank₂ − 1) spacings (m)")
    ax.set_ylabel("new top width, left + right (m)")
    ax.set_title(f"{old.size} width-to-depth cross sections, same ground")
    ax = axes[1]
    extra = (new - old) / spacing
    ax.hist(extra, bins=np.arange(0.0, 3.26, 0.05), color=NEW, alpha=0.8)
    ax.set_xlabel("new − legacy top width (spacings)")
    ax.set_ylabel("cross sections")
    q = np.percentile(extra, [10, 50, 90])
    note(ax, f"median {q[1]:.2f} spacings wider\np10 {q[0]:.2f}, p90 {q[2]:.2f}", loc="upper right")
    return fig, dict(caption=(
        "Every cross section of every site whose banks the new code found by the width-to-depth ratio, with legacy's "
        "own width-to-depth search run on the same sampled ground. Legacy's banks were whole ordinates inside the "
        "water's edges, and it counted one spacing fewer than the distance between them, so its channels were one "
        "to three spacings narrower (and for the same baseflow deeper)."),
                 stats=dict(sections=int(old.size), median_extra_spacings=rounded(q[1], 2), p10=rounded(q[0], 2),
                            p90=rounded(q[2], 2)))


LEGACY_FOUND = {"find_single_cell_bathymetry_by_target_width": "single cell (width prior)",
                "find_bank_using_width_to_depth_ratio": "width-to-depth ratio",
                "find_wse_and_banks_by_flat_water": "flat water", "find_bank_using_land_cover": "land cover",
                None: "none"}
NEW_FOUND = {"single_cell": "single cell (width prior)", "width_to_depth_ratio": "width-to-depth ratio",
             "flat_water": "flat water", "land_cover": "land cover", "none": "none", "target_width": "a width"}
WIDTH_CLASSES = ("narrower than a spacing", "1 spacing", "1 to 2 spacings", "2 to 3 spacings", "3 spacings or more",
                 "no valid banks (legacy: one cell)")


def _width_class(width_spacings: float) -> str:
    if not np.isfinite(width_spacings):
        return WIDTH_CLASSES[-1]
    if width_spacings < 1 - 1e-9:
        return WIDTH_CLASSES[0]
    if width_spacings <= 1 + 1e-9:
        return WIDTH_CLASSES[1]
    for k, upper in ((2, 2.0), (3, 3.0)):
        if width_spacings < upper - 1e-9:
            return WIDTH_CLASSES[k]
    return WIDTH_CLASSES[4]


@figure("B3", "How the channels' banks were found, and how wide the channels are", "Banks")
def bank_methods(ctx):
    found = {"legacy": collections.Counter(), "new": collections.Counter()}
    carved = {"legacy": collections.Counter(), "new": collections.Counter()}
    ground = {"legacy": [], "new": []}
    for site in ctx.sites:
        legacy = ctx.capture("as_configured", "legacy", site)
        for cell in [] if legacy is None else legacy["cells"]:
            if cell is None:
                continue
            f = cell["found_banks"]
            found["legacy"][LEGACY_FOUND.get(f.get("function_used") if f.get("is_valid") else None, "other")] += 1
            b = cell["banks"]
            total = b.get("i_total_bank_cells", 0) if b.get("is_valid") else math.nan
            carved["legacy"][_width_class(total)] += 1
            if b.get("is_valid"):
                ground["legacy"].append(b.get("i_bank_1_index", 0) + b.get("i_bank_2_index", 0))
        for cell in _new_cells(ctx, site):
            found["new"][NEW_FOUND[cell["banks"].method] if cell["banks"].valid else "none"] += 1
            h = cell["hydraulic_banks"]
            width = h.top_width / cell["spacing"] if h is not None and h.valid else math.nan
            carved["new"][_width_class(width)] += 1
            if np.isfinite(width):
                ground["new"].append(width)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    ax = axes[0]
    labels = ["single cell (width prior)", "width-to-depth ratio", "flat water", "land cover", "none"]
    y = np.arange(len(labels))
    for k, (code, color) in enumerate((("legacy", LEGACY), ("new", NEW))):
        ax.barh(y + (k - 0.5) * 0.38, [found[code][l] for l in labels], height=0.38, color=color, label=code)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("cross sections, all sites")
    ax.set_title("as each cross section's search found them")
    ax.legend()
    ax = axes[1]
    y = np.arange(len(WIDTH_CLASSES))
    for k, (code, color) in enumerate((("legacy", LEGACY), ("new", NEW))):
        ax.barh(y + (k - 0.5) * 0.38, [carved[code][c] for c in WIDTH_CLASSES], height=0.38, color=color,
                label=code)
    ax.set_yticks(y)
    ax.set_yticklabels(WIDTH_CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel("cross sections, all sites")
    ax.set_title("the channel carved, after the reach's width filter")
    ax.legend()
    return fig, dict(caption=(
        "Left: the method that found each cross section's banks, pooled over the sites. Legacy's single cells, and "
        "the new ones, come from the drainage-area width prior where the DEM can't resolve the channel. The new "
        "validity rule (a channel found from the DEM at least 2 spacings wide) lets more width-to-depth channels "
        "through. Right: the width of the channel the hydraulics and the carve use, after the reach's width filter "
        "and the fallbacks. Legacy's widths are its count, bank₁ + bank₂ − 1 spacings, so its single cell counted "
        "1 spacing though its bank ordinates were 2 spacings apart. The new single cells are the prior's width, "
        "narrower than a cell here."),
                 stats=dict(found={k: dict(v) for k, v in found.items()}, carved={k: dict(v) for k, v in carved.items()}))


def _pick_reach(ctx, site, minimum=40):
    """The reach of a site with the most cross sections that both codes smoothed."""
    new = ctx.capture("as_configured", "new", site)
    best = max((r for r, v in new["reaches"].items() if v["smoothed"] is not None),
               key=lambda r: len(new["reaches"][r]["cells"]))
    return best


def _new_stations(new, reach):
    """Each of a reach's cross sections' station (metres from its upstream end), keyed by stream cell."""
    info = new["reaches"][reach]
    smoothed = info["smoothed"]
    stations = np.empty(len(info["cells"]))
    stations[smoothed["order"]] = smoothed["stations"]
    return {(new["cells"][k]["row"], new["cells"][k]["col"]): stations[j] for j, k in enumerate(info["cells"])}, \
        stations


@figure("B4", "The reach's width filter", "Banks")
def width_filter(ctx):
    site = ctx.detail_sites[0]
    legacy, new = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
    reach = _pick_reach(ctx, site)
    by_cell, stations = _new_stations(new, reach)
    info = new["reaches"][reach]
    smoothed = info["smoothed"]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.0), sharex=True)
    ax = axes[1]
    found = np.array([b.top_width if b.valid else np.nan for b in info["found_banks"]])
    after = np.array([b.top_width if b.valid else np.nan for b in smoothed["banks"]])
    widths = smoothed["widths"]
    ax.axhspan(widths.q25, widths.q75, color=NEW, alpha=0.12, lw=0, label="25th to 75th percentile")
    ax.axhline(widths.median, color=NEW, lw=1.0, ls="--", label=f"median {widths.median:.1f} m")
    ax.plot(stations, found, "o", color=NEW, ms=3.5, mfc="none", label="as found")
    rebuilt = np.abs(after - found) > 1e-9
    ax.plot(stations[rebuilt | np.isnan(found)], after[rebuilt | np.isnan(found)], "o", color=NEW, ms=3.5,
            label="rebuilt at the median")
    spacing = np.median([new["cells"][k]["spacing"] for k in info["cells"]])
    ax.axhline(spacing, color=GROUND, lw=0.7, ls=":")
    ax.text(stations.max(), spacing, " one spacing", va="bottom", ha="right", fontsize=7, color=GROUND)
    ax.set_ylabel("channel top width (m)")
    ax.set_title("new")
    ax.legend(loc="upper left", fontsize=6.8, ncol=2)
    ax = axes[0]
    rows = []
    for cell in legacy["cells"]:
        if cell is None or (cell["row"], cell["col"]) not in by_cell:
            continue
        b, f = cell["banks"], cell["found_banks"]
        station = by_cell[(cell["row"], cell["col"])]
        width_found = f.get("i_total_bank_cells", np.nan) * cell["spacing"] if f.get("is_valid") else np.nan
        width_after = b.get("i_total_bank_cells", np.nan) * cell["spacing"] if b.get("is_valid") else np.nan
        rows.append((station, width_found, width_after, b.get("reach_top_width_filter_q25", np.nan),
                     b.get("reach_top_width_filter_median_top_width", np.nan),
                     b.get("reach_top_width_filter_q75", np.nan)))
    rows = np.array(rows, dtype=float)
    q25, median, q75 = np.nanmedian(rows[:, 3]), np.nanmedian(rows[:, 4]), np.nanmedian(rows[:, 5])
    ax.axhspan(q25, q75, color=LEGACY, alpha=0.12, lw=0, label="25th to 75th percentile")
    ax.axhline(median, color=LEGACY, lw=1.0, ls="--", label=f"median {median:.1f} m")
    ax.plot(rows[:, 0], rows[:, 1], "o", color=LEGACY, ms=3.5, mfc="none", label="as found (counted)")
    changed = np.abs(rows[:, 2] - rows[:, 1]) > 1e-9
    ax.plot(rows[changed | np.isnan(rows[:, 1]), 0], rows[changed | np.isnan(rows[:, 1]), 2], "o", color=LEGACY,
            ms=3.5, label="after the filter")
    ax.axhline(spacing, color=GROUND, lw=0.7, ls=":")
    ax.set_ylabel("channel top width (m)")
    ax.set_title("legacy")
    ax.legend(loc="upper left", fontsize=6.8, ncol=2)
    axes[1].set_xlabel("metres along the reach from its upstream end")
    top = np.nanmax(np.r_[found, after, rows[:, 1], rows[:, 2]]) * 1.1
    for a in axes:
        a.set_ylim(0, top)
    return fig, dict(caption=(
        f"The channel widths along the longest reach of {ctx.site_label(site)}. Each reach's channels narrower "
        "than its 25th percentile width or wider than its 75th, and those without valid banks, are rebuilt at its "
        "median width. Legacy rebuilt at whole ordinates, widening a median it couldn't draw a cell at a time, and "
        "its single cells counted a spacing; the new median can be any width, narrower than a cell included, so a "
        "reach of sub-cell channels keeps a sub-cell median."),
                 stats=dict(site=site, reach=int(reach), new_median=rounded(widths.median, 2),
                            legacy_median=rounded(median, 2), spacing=rounded(spacing, 2)))


def _single_cell_example(ctx, site):
    """A new cross section carved as a single cell narrower than its spacing, with its bank elevation and depth."""
    cells = [c for c in _new_cells(ctx, site) if c["hydraulic_banks"] is not None and c["hydraulic_banks"].single_cell
             and np.isfinite(c["carve_depth"]) and np.isfinite(c["bank_elevation"]) and c["profile"] is not None]
    cells.sort(key=lambda c: abs(c["bank_elevation"] - c["found"][c["found"].size // 2] - 1.0))
    return cells[0]


@figure("B5", "A channel one cell wide, or narrower", "Banks")
def single_cell_channels(ctx):
    from arc.bathymetry import banks_for_width, single_cell_banks
    from arc.xsection.xsection import XSection
    site = ctx.detail_sites[0]
    cell = _single_cell_example(ctx, site)
    e, s = cell["found"], cell["spacing"]
    bank, depth = cell["bank_elevation"], cell["carve_depth"]
    x = new_stations(e.size, s)
    plain = XSection(e.copy(), np.full(e.size, 0.035), s)
    prior = cell["hydraulic_banks"]
    cases = (("legacy: a single cell, banks at the neighbouring ordinates", None),
             ("new, without a width prior: one spacing, ±½ spacing", single_cell_banks(plain)),
             (f"new, the width prior's {prior.top_width:.1f} m", prior))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    stats = {}
    for ax, (title, banks) in zip(axes, cases):
        between_banks(ax, s, s, s, curves=[(x, e), ([0.0], [bank - depth]), ([0.0], [bank])], margin=1.2)
        _plot_ground(ax, x, e, color=GROUND, lw=1.0, ls="--", marker="o", ms=3, label="ground")
        if banks is None:
            carved = legacy_carve(e, s, 1, 1, depth, bank_elevation=bank)
            _plot_ground(ax, x, carved, color=LEGACY, lw=2.0, marker="s", ms=4, label="legacy's carved ordinates")
            water(ax, x, carved, bank)
            area = polygon_area(x, carved, bank, -s, s)
            stats["legacy"] = dict(area=rounded(area, 2), width=rounded(2 * s, 2))
            ax.axvline(-s, color=LEGACY, lw=0.8, ls=":")
            ax.axvline(s, color=LEGACY, lw=0.8, ls=":", label="its bank ordinates")
        else:
            xs, changed = new_carve(e, s, banks, depth, bank_elevation=bank)
            px, py = _profile_xy(xs)
            water(ax, px, py, bank)
            ax.plot(px, py, color=NEW, lw=2.0, label="new profile (what the hydraulics see)")
            _plot_ground(ax, x, xs.elevations, color=NEW, lw=0, marker="s", ms=5, mfc="white",
                         label="its ordinates (raster, XS file)")
            mark_banks(ax, banks.left, banks.right, color=BANK)
            area = polygon_area(px, py, bank, -s, s)
            stats[title] = dict(area=rounded(area, 2), width=rounded(banks.top_width, 2))
        ax.axhline(bank, color=BANK, lw=0.7, ls=":")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("metres from the stream cell")
        note(ax, f"area below the bank elevation: {area:.1f} m²", loc="lower right")
        ax.legend(loc="upper center", fontsize=6.3)
    axes[0].set_ylabel("elevation (m)")
    return fig, dict(caption=(
        f"One of {ctx.site_label(site)}'s cross sections the DEM can't resolve ({s:.1f} m spacing), carved {depth:.2f} m "
        "below its smoothed bank elevation (dotted) with Bathy_Use_Banks. Legacy set the stream cell to the bed and "
        "kept its neighbours, so its channel was the V from the bed to the neighbouring ordinates, two spacings "
        "wide, though it counted it as one. The new single cell is a trapezoid, one spacing wide without a width "
        "prior, or as wide as the prior says, with its exact shape in the cross section's profile. Its bank "
        "elevations for the reach smoothing are still the neighbouring ordinates' ground, and the raster gets just "
        "the stream cell, at the bed."), stats=dict(cell=(cell["row"], cell["col"]), **stats))


# --- The channel's depth -----------------------------------------------------------------------------------------


@figure("DP1", "The channel's depth for its baseflow", "Channel depth")
def channel_depths(ctx):
    from arc.bathymetry import trapezoid_depth
    from arc.cross_section import find_depth_of_bathymetry, find_depth_of_bathymetry_triangle
    n, height = 0.03, TRAPEZOID_HEIGHT  # legacy's fixed bathymetry roughness, and the sites' Bathy_Trap_H
    rng = np.random.default_rng(12)
    shortfall = []
    for _ in range(3000):
        top = float(rng.uniform(5.0, 200.0))
        q = float(10 ** rng.uniform(-1, 2.7))
        slope = float(10 ** rng.uniform(-4, -2))
        exact = trapezoid_depth(q, top * (1 - 2 * height), top, slope, n)
        old = find_depth_of_bathymetry(q, top * (1 - 2 * height), top, slope, n)
        if 0 < exact < 24:
            shortfall.append((exact - old) * 100)
    shortfall = np.array(shortfall)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axes[0]
    ax.hist(shortfall, bins=np.linspace(-0.1, 1.1, 49), color=LEGACY, alpha=0.8)
    ax.set_xlabel("exact depth − legacy's stepped depth (cm)")
    ax.set_ylabel("trapezoids")
    ax.set_title("trapezoids 5 to 200 m wide, random baseflows and slopes", fontsize=8.5)
    note(ax, f"{shortfall.size} trapezoids\nlegacy shallower in {np.mean(shortfall > 1e-9):.0%},\n"
             f"by up to {shortfall.max():.2f} cm", loc="upper center")
    ax = axes[1]
    spacing, slope = 30.0, 0.001
    flows = np.logspace(-1.5, 1.8, 120)
    cases = ((lambda q: find_depth_of_bathymetry_triangle(q, spacing, 0.0, 0.0, 0.0, slope, n), LEGACY, "-",
              "legacy's triangle, neighbours level with the stream cell"),
             (lambda q: find_depth_of_bathymetry_triangle(q, spacing, 0.0, 1.0, 1.0, slope, n), LEGACY, "--",
              "legacy's triangle, neighbours 1 m higher"),
             (lambda q: trapezoid_depth(q, spacing * (1 - 2 * height), spacing, slope, n), NEW, "-",
              "new: a trapezoid one spacing wide"),
             (lambda q: trapezoid_depth(q, 0.5 * spacing * (1 - 2 * height), 0.5 * spacing, slope, n), NEW, "--",
              "new: a width prior of half a spacing"))
    for depth, color, style, label in cases:
        ax.plot(flows, [depth(q) for q in flows], color=color, ls=style, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("baseflow (m³/s)")
    ax.set_ylabel("channel depth (m)")
    ax.set_title(f"a channel one cell wide, {spacing:g} m spacing, slope {slope:g}", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=6.8)
    return fig, dict(caption=(
        "The depth at which the channel carries its baseflow by Manning's equation, both codes at n = 0.03, legacy's "
        "fixed value (since 526553b the new code solves it with the water class's n from the Manning's n table, the n "
        "the channel has between its banks). "
        "Left: legacy stepped the depth by 1 m, 0.5 m, 0.1 m and 1 cm, stopping a step short of the answer, so its "
        "trapezoids came out up to a centimetre shallow; the new depth is solved exactly. Right: a single cell. "
        "Legacy's was a triangle from the stream cell out to the neighbouring ordinates, stepped 10 cm at a time, and "
        "narrowed by how far the neighbours stood above the stream cell; the new one is the trapezoid, one spacing "
        "wide or the width prior's. (The sites take their depths from the drainage-area power law, so this only "
        "matters where a depth comes from the baseflow.)"),
                 stats=dict(trapezoids=int(shortfall.size), legacy_shallower=rounded(np.mean(shortfall > 1e-9), 3),
                            largest_shortfall_cm=rounded(shortfall.max(), 3)))


# --- The carve ---------------------------------------------------------------------------------------------------------


def _flat_valley(spacing, size=41, bank=100.0, side_slope=0.02):
    """Ground at the bank elevation around the channel, rising gently away from it."""
    x = new_stations(size, spacing)
    return bank + side_slope * np.abs(x)


@figure("C1", "Channels from half a cell to four cells wide", "Carving the channel")
def channel_shapes(ctx):
    from arc.bathymetry import banks_for_width
    from arc.xsection.xsection import XSection
    spacing, depth, bank = 30.0, 2.0, 100.0
    e = _flat_valley(spacing)
    x = new_stations(e.size, spacing)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), sharey=True)
    stats = {}
    for ax, multiple in zip(axes.flat, (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
        width = multiple * spacing
        banks = banks_for_width(XSection(e.copy(), np.full(e.size, 0.035), spacing), width)
        xs, _ = new_carve(e, spacing, banks, depth, bank_elevation=bank)
        px, py = _profile_xy(xs)
        legacy = legacy_section(e, spacing, use_banks=True)
        i1, i2, total = legacy_width_banks(legacy, width)
        carved = legacy_carve(e, spacing, i1, i2, depth, bank_elevation=bank)
        design = depth * width * (1 - TRAPEZOID_HEIGHT)
        reach = max(i1, i2) + 1.5
        lo, hi = -reach * spacing, reach * spacing
        new_area = polygon_area(px, py, bank, lo, hi)
        old_area = polygon_area(x, carved, bank, lo, hi)
        ordinates_area = polygon_area(x, xs.elevations, bank, lo, hi)
        water(ax, px, py, bank, color=NEW, alpha=0.15)
        ax.plot(px, py, color=NEW, lw=2.0, label=f"new profile: {new_area:.0f} m²")
        _plot_ground(ax, x, xs.elevations, color=NEW, lw=0.8, ls="--", marker="s", ms=4, mfc="white",
                     label=f"through the new ordinates: {ordinates_area:.0f} m²")
        _plot_ground(ax, x, carved, color=LEGACY, lw=1.3, marker="x", ms=5,
                     label=f"legacy (banks {i1}, {i2}): {old_area:.0f} m²")
        ax.set_xlim(lo, hi)
        ax.set_ylim(bank - depth - 0.4, bank + 1.7)
        ax.set_title(f"{multiple:g} spacing{'s' if multiple != 1 else ''} wide: the trapezoid holds {design:.0f} m²",
                     fontsize=8.5)
        ax.legend(loc="upper center", fontsize=6.3)
        ax.set_xlabel("metres from the stream cell")
        stats[f"{multiple:g}"] = dict(design=rounded(design, 1), new=rounded(new_area, 1),
                                      ordinates=rounded(ordinates_area, 1), legacy=rounded(old_area, 1))
    for a in axes[:, 0]:
        a.set_ylabel("elevation (m)")
    return fig, dict(caption=(
        f"A channel {depth:g} m deep carved below a bank elevation of {bank:g} m (Bathy_Use_Banks, Bathy_Trap_H 0.2) "
        f"into gently rising ground on {spacing:g} m ordinates, at widths from half a spacing to four. The new "
        "profile is the trapezoid itself, whatever its width. Its ordinates, which the raster and the cross-section "
        "file get, are its values there, so a polyline through them loses area. Legacy rounded the width to whole "
        "ordinates, carved one ordinate beyond its banks with bank elevations, and at a spacing or less made a "
        "single cell: the stream cell at the bed."), stats=stats)


@figure("C2", "How much of the channel's area is carved, by width", "Carving the channel")
def channel_areas(ctx):
    from arc.bathymetry import banks_for_width
    from arc.xsection.xsection import XSection
    spacing, depth, bank = 30.0, 2.0, 100.0
    e = np.full(61, bank)
    x = new_stations(e.size, spacing)
    multiples = np.round(np.arange(0.25, 6.01, 0.05), 2)
    ratios = {"new profile": [], "a polyline through the new ordinates": [], "legacy, with bank elevations": [],
              "legacy, without (below the stream cell)": []}
    for multiple in multiples:
        width = multiple * spacing
        design = depth * width * (1 - TRAPEZOID_HEIGHT)
        banks = banks_for_width(XSection(e.copy(), np.full(e.size, 0.035), spacing), width)
        xs, _ = new_carve(e, spacing, banks, depth, bank_elevation=bank)
        px, py = _profile_xy(xs)
        lo, hi = -10 * spacing, 10 * spacing
        ratios["new profile"].append(polygon_area(px, py, bank, lo, hi) / design)
        ratios["a polyline through the new ordinates"].append(polygon_area(x, xs.elevations, bank, lo, hi) / design)
        legacy = legacy_section(e, spacing, use_banks=True)
        i1, i2, _ = legacy_width_banks(legacy, width)
        carved = legacy_carve(e, spacing, i1, i2, depth, bank_elevation=bank)
        ratios["legacy, with bank elevations"].append(polygon_area(x, carved, bank, lo, hi) / design)
        carved = legacy_carve(e, spacing, i1, i2, depth)
        ratios["legacy, without (below the stream cell)"].append(polygon_area(x, carved, bank, lo, hi) / design)
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.axhline(1.0, color=GROUND, lw=0.6)
    styles = ((NEW, "-", 3.0, 5), (NEW, "--", 1.3, 4), (LEGACY, "-", 1.3, 3), (ACCENT, "-.", 1.3, 3))
    for (label, values), (color, style, width, order) in zip(ratios.items(), styles):
        ax.plot(multiples, values, color=color, ls=style, lw=width, label=label, zorder=order)
    ax.set_yscale("log")
    ax.set_yticks([0.4, 0.5, 5 / 8, 1, 2, 5])
    ax.set_yticklabels(["0.4", "0.5", "5/8", "1", "2", "5"])
    ax.axhline(5 / 8, color=GROUND, lw=0.6, ls=":")
    ax.axvline(1.0, color=GROUND, lw=0.6, ls=":")
    ax.axvline(2.0, color=GROUND, lw=0.6, ls=":")
    ax.set_xlabel("channel width (spacings)")
    ax.set_ylabel("area carved / the trapezoid's area")
    ax.legend(loc="upper right", fontsize=7)
    return fig, dict(caption=(
        "The area below the reference level that each carve leaves, over the area of the trapezoid whose depth was "
        f"solved for, on flat ground ({spacing:g} m spacing, {depth:g} m deep, Bathy_Trap_H 0.2). The new profile is "
        "the trapezoid at any width. On the ordinates alone a channel two spacings wide is a triangle with 5/8 of the "
        "area, and a narrower one is a single V. Legacy's rounding to whole ordinates, its single cells two spacings "
        "wide, and with bank elevations its carving one ordinate too far out, gave it too much area or too little."),
                 stats={k: dict(at_1=rounded(v[int(np.argmin(np.abs(multiples - 1.0)))], 3),
                                at_2=rounded(v[int(np.argmin(np.abs(multiples - 2.0)))], 3),
                                at_4=rounded(v[int(np.argmin(np.abs(multiples - 4.0)))], 3)) for k, v in ratios.items()})


@figure("C3", "Two errors in legacy's carve", "Carving the channel")
def legacy_carve_errors(ctx):
    from arc.bathymetry import Banks
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    # (a) with bank elevations legacy carved one ordinate beyond its banks
    e = np.array([12.0, 11.0, 10.4, 10.0, 10.5, 11.0, 12.0])
    x = new_stations(e.size, 1.0)
    banks = Banks("test", 2.0, 2.0, 11.0, 11.0, False, True)
    xs, _ = new_carve(e, 1.0, banks, 1.5, bank_elevation=11.0, trapezoid_height=0.1)
    carved = legacy_carve(e, 1.0, 2, 2, 1.5, bank_elevation=11.0, trapezoid_height=0.1)
    ax = axes[0]
    _plot_ground(ax, x, e, color=GROUND, ls="--", marker="o", ms=4, label="ground")
    _plot_ground(ax, x, carved, color=LEGACY, lw=1.5, marker="x", ms=7, mew=1.5, label="legacy")
    px, py = _profile_xy(xs)
    ax.plot(px, py, color=NEW, lw=2.0, label="new profile")
    mark_banks(ax, 2.0, 2.0)
    ax.axhline(11.0, color=BANK, lw=0.7, ls=":")
    ax.set_title("banks 2 m out at 11 m, 1.5 m deep (legacy's staged test)", fontsize=8.5)
    ax.set_xlabel("metres from the stream cell")
    ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper center", fontsize=7)
    # (b) each half carved from its own bank
    e = np.full(43, 13.0)
    e[1:23] = 10.0
    e = np.roll(e, 0)
    x = new_stations(e.size, 1.0)
    banks = Banks("test", 20.0, 1.0, 10.0, 10.0, False, True)
    xs, _ = new_carve(e, 1.0, banks, 2.0)
    center = e.size // 2
    halves = []
    for i1, i2 in ((20, 1),):
        carved = legacy_carve(e, 1.0, i1, i2, 2.0)
    legacy = legacy_section(e, 1.0, use_banks=False)
    result = {"function_used": "figure", "i_bank_1_index": 20, "i_bank_2_index": 1, "i_total_bank_cells": 20,
              "bank_elev_1": 10.0, "bank_elev_2": 10.0, "is_valid": True, "bathymetry_depth": 2.0,
              "bathymetry_should_apply": True, "smoothed_bank_elevation": math.nan}
    legacy.Calculate_Bathymetry_Based_on_WSE_or_LC(np.full((3, e.size + 2), np.nan), bank_search_result=result)
    left_half = legacy.da_xs_profile1[0]
    right_half = legacy.da_xs_profile2[0]
    ax = axes[1]
    _plot_ground(ax, x, e, color=GROUND, ls="--", label="ground")
    _plot_ground(ax, x, carved, color=LEGACY, lw=1.5, marker="x", ms=4, label="legacy")
    px, py = _profile_xy(xs)
    ax.plot(px, py, color=NEW, lw=2.0, label="new profile")
    ax.plot([0, 0], [left_half, right_half], "o", color=LEGACY, ms=6, mfc="none")
    ax.annotate(f"legacy's halves: {left_half:.2f} m and {right_half:.2f} m;\nthe raster averaged them to "
                f"{0.5 * (left_half + right_half):.2f} m", xy=(0, 0.5 * (left_half + right_half)), xytext=(-18, 12.2),
                fontsize=7, color=LEGACY, arrowprops=dict(arrowstyle="->", color=LEGACY, lw=0.8))
    mark_banks(ax, 20.0, 1.0)
    ax.set_xlim(-23, 6)
    ax.set_ylim(7.6, 13.4)
    ax.set_title("banks 20 m left and 1 m right, 2 m deep, without bank elevations", fontsize=8.5)
    ax.set_xlabel("metres from the stream cell")
    ax.legend(loc="lower left", fontsize=7)
    return fig, dict(caption=(
        "Left: with bank elevations, legacy passed its bank index plus one as well as the distance beyond the bank, "
        "so it carved one ordinate too far: the bank ordinate went down to the bed and the ground beyond it down to "
        "the bank elevation. Right: legacy carved each half of the cross section from its own bank, so where the "
        "stream cell lay on the far bank's slope the halves gave it two elevations, which the raster averaged. The "
        "new carve measures every point from the nearer bank."),
                 stats=dict(lopsided_halves=[rounded(left_half, 3), rounded(right_half, 3)],
                            new_stream_cell=rounded(xs.elevations[center], 3)))


def _made_up_channel(spacing, water, bank_top, floodplain_rise=0.005, half_water=30.0, half_top=40.0, size=41):
    """A made-up cross section: flat water between ±half_water (the DEM's water surface), banks rising to bank_top
    at ±half_top, then gently rising floodplain."""
    x = new_stations(size, spacing)
    a = np.abs(x)
    ground = np.where(a <= half_water, water,
                      np.where(a <= half_top, water + (bank_top - water) * (a - half_water) / (half_top - half_water),
                               bank_top + floodplain_rise * (a - half_top)))
    return x, ground


BANK_MODE_CASES = (
    ("the bank elevation at the bank tops", 100.5, 102.0, 102.0),
    ("the bank elevation 0.75 m below them\n(legacy's smoothing's lower envelope, C4b)", 100.5, 102.0, 101.25),
    ("the bank elevation below the stream cell\n(where legacy's smoothing often put it, BS3)", 100.5, 102.0, 99.8),
    ("an incised channel, the bank elevation\nat its bank tops", 99.0, 104.0, 104.0),
)


@figure("C4", "What Bathy_Use_Banks does, on made-up cross sections", "Carving the channel")
def bank_mode_synthetic(ctx):
    from arc.bathymetry import Banks
    spacing, depth = 10.0, 2.0
    fig, axes = plt.subplots(2, len(BANK_MODE_CASES), figsize=(15, 7.2), sharex=True)
    stats = {}
    for column, (title, water_level, bank_top, bank_elevation) in enumerate(BANK_MODE_CASES):
        x, e = _made_up_channel(spacing, water_level, bank_top)
        banks = Banks("test", 40.0, 40.0, bank_top, bank_top, False, True)
        for row, use_banks in enumerate((True, False)):
            ax = axes[row, column]
            reference = bank_elevation if use_banks else None
            level = bank_elevation if use_banks else water_level
            xs, _ = new_carve(e, spacing, banks, depth, bank_elevation=reference)
            px, py = _profile_xy(xs)
            carved = legacy_carve(e, spacing, 4, 4, depth, bank_elevation=reference)
            ax.fill_between(x, np.minimum(e, level), level, where=np.abs(x) <= 40, color=WATER, alpha=0.18, lw=0)
            _plot_ground(ax, x, e, color=GROUND, lw=1.0, ls="--", marker="o", ms=3, label="ground (the DEM)")
            _plot_ground(ax, x, carved, color=LEGACY, lw=1.2, marker="x", ms=5, label="legacy carved")
            ax.plot(px, py, color=NEW, lw=2.2, label="new profile")
            mark_banks(ax, 40.0, 40.0)
            ax.axhline(level, color=BANK, lw=0.9, ls=":",
                       label="reference: the bank elevation" if use_banks else "reference: the stream cell")
            bed = level - depth
            change = bed - water_level
            ax.annotate(f"bed {bed:.2f} m: {abs(change):.2f} m {'above' if change > 0 else 'below'}\nthe DEM's water "
                        f"at {water_level:g} m", xy=(0, bed), xytext=(0.5, 0.04), textcoords="axes fraction",
                        ha="center", fontsize=7, color=NEW,
                        arrowprops=dict(arrowstyle="->", color=NEW, lw=0.7))
            ax.set_xlim(-75, 75)
            ax.set_ylim(min(water_level, level) - depth - 1.3, max(bank_top, level) + 0.9)
            if row == 0:
                ax.set_title(title, fontsize=8.5)
            if column == 0:
                ax.set_ylabel(("Bathy_Use_Banks true" if use_banks else "Bathy_Use_Banks false") + "\nelevation (m)")
            if row == 1:
                ax.set_xlabel("metres from the stream cell")
            stats[f"{column}_{'banks' if use_banks else 'wse'}"] = dict(
                bed=rounded(bed, 2), stream_cell_change=rounded(xs.elevations[x.size // 2] - water_level, 2),
                bank_top_change=rounded(np.interp(40.0, px, py) - bank_top, 2),
                legacy_stream_cell_change=rounded(carved[x.size // 2] - water_level, 2))
    axes[0, 0].legend(loc="upper center", fontsize=6.3)
    axes[1, 0].legend(loc="upper center", fontsize=6.3)
    return fig, dict(caption=(
        f"A channel carved {depth:g} m deep (a drainage-area depth, as the sites use) between banks 40 m either side "
        f"of the stream cell, on {spacing:g} m ordinates, in four made-up cross sections: the DEM's water is flat "
        "between ±30 m and the banks rise to their tops at ±40 m. With Bathy_Use_Banks (top) the channel's top is "
        "the smoothed bank elevation at the banks and its bed is that less the depth, wherever that leaves it: the "
        "trapezoid starts at the bank tops only when the smoothing puts the bank elevation there (the new smoothing "
        "puts it at the DEM's water plus a low percentile of the nearby banks' heights, never below the stream cell; "
        "legacy's put it along the reach's lowest banks, C4b and BS2). Below them it cuts "
        "the bank tops down to it, below the stream cell it takes the whole channel under the DEM's water, and in an "
        "incised channel it fills the bed (decided on 2025-09-18). Without (bottom) the reference is the stream "
        "cell, so "
        "the bed is always the depth below the DEM's water, and the channel between the banks is lowered to the "
        "trapezoid: the bank slopes above the water down to the water's level at the banks. Legacy (vermillion) "
        "carved one ordinate too far with bank elevations (C3), and with its whole-ordinate banks."), stats=stats)


C4B_CELL = (730, 776)  # the cross section of Cuyahoga that the first C4 showed, which the question was about


def _channels_with(ctx, site, smooth):
    """The new code's site with its banks smoothed by smooth (a smooth_bank_elevations), and the channels the bed
    smoothing then gives, from the capture's depths as the pipeline would: (the capture, the smoothing, the
    channels)."""
    from arc.bathymetry import smooth_channel_depths
    from arc.bathymetry.bed_smoothing import MAX_BED_GRADE
    network, reaches, new = _rebuilt_reaches(ctx, site)
    grid = ctx.grid(site)
    smoothed = smooth(network, reaches, grid.dx, grid.dy)
    use_banks = bool(new["bathy_use_banks"])
    kept = {reach: sections for reach, sections in reaches.items()
            if new["reaches"][reach]["depths"] is not None
            and (not use_banks or np.isfinite(np.asarray(smoothed[reach].bank_elevations)).any())}
    channels = smooth_channel_depths(network, kept, {reach: smoothed[reach] for reach in kept},
                                     {reach: new["reaches"][reach]["depths"] for reach in kept}, grid.dx, grid.dy,
                                     use_banks=use_banks,
                                     max_bed_grade=MAX_BED_GRADE if new["bathy_bed_cap"] else None)
    return new, smoothed, channels


def _legacy_way(ctx, site):
    """_channels_with legacy's bank smoothing."""
    from functools import partial

    from arc.bathymetry import smooth_bank_elevations
    return _channels_with(ctx, site, partial(smooth_bank_elevations, method="legacy"))


@figure("C4b", "The real cross section behind the question: where each smoothing puts the channel's top",
        "Carving the channel")
def carve_modes(ctx):
    site = ctx.detail_sites[0]
    cells = [c for c in _new_cells(ctx, site) if c["hydraulic_banks"] is not None and c["hydraulic_banks"].valid
             and np.isfinite(c["carve_depth"]) and np.isfinite(c["bank_elevation"])]
    cell = next((c for c in cells if (c["row"], c["col"]) == C4B_CELL), None)
    if cell is None:  # another site: its widest channel
        cell = max(cells, key=lambda c: c["hydraulic_banks"].top_width)
    new, legacy_way, channels = _legacy_way(ctx, site)
    reach = cell["reach"]
    ks = new["reaches"][reach]["cells"]
    position = next(i for i, k in enumerate(ks) if new["cells"][k] is cell)
    e, s, banks = cell["found"], cell["spacing"], cell["hydraulic_banks"]
    thalweg = float(e[e.size // 2])
    own = min(banks.left_elevation, banks.right_elevation)
    ways = {"new": (float(cell["bank_elevation"]), float(cell["carve_depth"])),
            "legacy": (float(legacy_way[reach].bank_elevations[position]),
                       float(channels[reach].depths[position]) if reach in channels else np.nan)}
    x = new_stations(e.size, s)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    profiles = {}
    for key, (bank, depth) in ways.items():
        if np.isfinite(bank) and np.isfinite(depth):
            profiles[key] = _profile_xy(new_carve(e, s, banks, depth, bank_elevation=bank)[0])
    between_banks(ax, banks.left, banks.right, s, curves=[(x, e), *profiles.values()], margin=2.0)
    _plot_ground(ax, x, e, color=GROUND, ls="--", marker="o", ms=3, label="ground")
    if "legacy" in profiles:
        ax.plot(*profiles["legacy"], color=ACCENT, lw=1.4, label="carved under legacy's smoothing")
    ax.plot(*profiles["new"], color=NEW, lw=2.0, label="carved under the new smoothing")
    mark_banks(ax, banks.left, banks.right)
    ax.axhline(own, color=GROUND, lw=0.8, ls=":", label=f"this cross section's lower bank: {own:.2f} m")
    ax.axhline(ways["legacy"][0], color=ACCENT, lw=0.9, ls=":",
               label=f"legacy's smoothed bank elevation: {ways['legacy'][0]:.2f} m")
    ax.axhline(ways["new"][0], color=NEW, lw=0.9, ls=":", label=f"the new smoothed bank elevation: {ways['new'][0]:.2f} m")
    ax.set_title(f"row {cell['row']}, column {cell['col']}", fontsize=8.5)
    ax.set_xlabel("metres from the stream cell")
    ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper center", fontsize=6.3)

    # the reach around it: why each smoothed bank elevation is where it is
    p = _reach_profiles(ctx, site, reach)
    order = np.argsort(p["stations"])
    st = p["stations"][order]
    here = p["stations"][position]
    ax = axes[1]
    obs = p["observations"][order]
    old = legacy_way[reach]
    anchors = np.asarray(old.anchors, bool)[order]
    new_channel = new["reaches"][reach]["channel"]
    ax.plot(st, p["thalweg"][order], color=GROUND, lw=0.9, label="the stream cells' ground")
    ax.plot(st, obs, "o", color=NEW, ms=3, mfc="none", label="each cross section's bank observation")
    ax.plot(st[anchors], obs[anchors], "o", color=ACCENT, ms=4.5, label="legacy's anchors (observations below its line)")
    ax.plot(st, np.asarray(old.bank_elevations)[order], color=ACCENT, lw=1.6, label="legacy's smoothed bank elevation")
    if reach in channels:
        ax.plot(st, (np.asarray(old.bank_elevations) - np.asarray(channels[reach].depths))[order], color=ACCENT,
                lw=0.9, ls="--", label="its bed")
    ax.plot(st, p["bank"][order], color=NEW, lw=1.8, label="the new smoothed bank elevation")
    if new_channel is not None:
        ax.plot(st, (p["bank"] - np.asarray(new_channel["depths"]))[order], color=NEW, lw=0.9, ls="--", label="its bed")
    ax.axvline(here, color=GROUND, lw=1.0, ls=":")
    ax.text(here, 0.99, "this cross section ", color=GROUND, fontsize=7, va="top", ha="right",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("metres along the reach from its upstream end")
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"its reach, {reach}", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=6.3, ncol=2)

    # the carved beds over the site, each way
    beds = {"new": [], "legacy": []}
    for r, info in new["reaches"].items():
        for i, k in enumerate(info["cells"]):
            c = new["cells"][k]
            if c is None or not np.isfinite(c["carve_depth"]):
                continue
            stream = c["found"][c["found"].size // 2]
            beds["new"].append(c["bank_elevation"] - c["carve_depth"] - stream)
            if r in channels:
                beds["legacy"].append(legacy_way[r].bank_elevations[i] - channels[r].depths[i] - stream)
    beds = {key: np.array(values, float) for key, values in beds.items()}
    gap = {key: float(np.nanmedian(obs - bank[order])) for key, bank in
           (("legacy", np.asarray(old.bank_elevations)), ("new", p["bank"]))}
    under = {key: own - ways[key][0] for key in ways}
    bed_here = {key: ways[key][0] - ways[key][1] - thalweg for key in ways}
    return fig, dict(caption=(
        f"The cross section of {ctx.site_label(site)} that the first C4 showed, with its reach. Its lower bank is at "
        f"{own:.2f} m. The smoothed bank elevation, the top of the carved channel with Bathy_Use_Banks, is the "
        "reach's, not this cross section's. Legacy's smoothing, the new code's until 2026-09-26, put it "
        f"{under['legacy']:.2f} m lower: a line falling to the reach's outlet and refitted only through observations "
        "below it, it follows the reach's lowest banks, and the median observation here is "
        f"{gap['legacy']:.1f} m above it (most are single cells, whose observations are the neighbouring ordinates' "
        f"ground). That the bed then landed on the stream cell ({bed_here['legacy']:+.2f} m) was a coincidence: over "
        f"the site legacy's smoothing put the carved bed a median {-np.nanmedian(beds['legacy']):.2f} m below the "
        f"stream cell. The new smoothing puts the channel's top at the DEM's water plus the 10th percentile of the "
        f"nearby banks' heights, here {abs(under['new']):.2f} m {'below' if under['new'] > 0 else 'above'} this "
        f"cross section's bank, and its bed {bed_here['new']:+.2f} m from the stream cell (over the site a median "
        f"{-np.nanmedian(beds['new']):.2f} m below it)."),
                 stats=dict(cell=(cell["row"], cell["col"]), reach=int(reach), own_bank=rounded(own, 3),
                            legacy_smoothed=rounded(ways["legacy"][0], 3), new_smoothed=rounded(ways["new"][0], 3),
                            legacy_bed_minus_thalweg=rounded(bed_here["legacy"], 3),
                            new_bed_minus_thalweg=rounded(bed_here["new"], 3),
                            median_observation_above_legacy=rounded(gap["legacy"], 2),
                            median_observation_above_new=rounded(gap["new"], 2),
                            site_median_bed_below_stream_cell_legacy=rounded(-np.nanmedian(beds["legacy"]), 3),
                            site_median_bed_below_stream_cell_new=rounded(-np.nanmedian(beds["new"]), 3)))


@figure("C5", "Carved cross sections on a real site", "Carving the channel")
def real_carves(ctx):
    site = ctx.detail_sites[0]
    legacy = {(c["row"], c["col"]): c for c in ctx.capture("as_configured", "legacy", site)["cells"] if c is not None}
    cells = [c for c in _new_cells(ctx, site) if np.isfinite(c["carve_depth"]) and (c["row"], c["col"]) in legacy]
    groups = {"a single cell, narrower than a spacing": [c for c in cells if c["hydraulic_banks"].single_cell],
              "the reach's median width": [c for c in cells if c["hydraulic_banks"].method == "target_width"],
              "found by the width-to-depth ratio": [c for c in cells
                                                    if c["hydraulic_banks"].method == "width_to_depth_ratio"]}
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.6))
    rng = np.random.default_rng(5)
    for row, (name, group) in enumerate(groups.items()):
        picks = [group[k] for k in rng.choice(len(group), size=min(3, len(group)), replace=False)] if group else []
        for ax, cell in zip(axes[row], picks):
            old = legacy[(cell["row"], cell["col"])]
            s, banks = cell["spacing"], cell["hydraulic_banks"]
            x = new_stations(cell["found"].size, s)
            px, py = np.asarray(cell["profile"][0]), np.asarray(cell["profile"][1])
            lx, lf = legacy_stations(*old["found"], old["spacing"], cell["side_one"])
            _, lc = legacy_stations(*old["final"], old["spacing"], cell["side_one"])
            b = old["banks"]
            reach = max(banks.left, banks.right, (b.get("i_bank_1_index", 1) + 1) * old["spacing"],
                        (b.get("i_bank_2_index", 1) + 1) * old["spacing"])
            between_banks(ax, reach, reach, s, curves=[(x, cell["found"]), (px, py), (lx, lc)], margin=1.0)
            ax.plot(x, np.where(cell["found"] < WALL, cell["found"], np.nan), color=NEW, lw=0.8, ls="--",
                    label="new ground")
            ax.plot(lx, lf, color=LEGACY, lw=0.8, ls="--", label="legacy's ground")
            ax.plot(lx, lc, color=LEGACY, lw=1.4, marker="x", ms=4, label="legacy carved")
            ax.plot(px, py, color=NEW, lw=2.0, label="new profile")
            mark_banks(ax, banks.left, banks.right, color=BANK, label="new banks")
            ax.axhline(cell["bank_elevation"], color=NEW, lw=0.6, ls=":", label="new bank elevation")
            ax.axhline(b.get("smoothed_bank_elevation", np.nan), color=LEGACY, lw=0.6, ls=":",
                       label="legacy's bank elevation")
            ax.set_title(f"{name}\nrow {cell['row']}, column {cell['col']}", fontsize=8)
            ax.set_xlabel("metres from the stream cell")
        axes[row, 0].set_ylabel("elevation (m)")
    axes[0, 0].legend(loc="lower left", fontsize=6.3)
    return fig, dict(caption=(
        f"Nine stream cells of {ctx.site_label(site)} as the two codes carved them, three of each kind of new "
        "channel. Each code samples its own cross section (their directions differ, D3), so their ground differs "
        "too. Dotted lines are each code's smoothed bank elevation and the green dashed verticals the new banks. The new "
        "profile is what the new rating curve sees; legacy's hydraulics saw its carved ordinates."), stats={})


@figure("C6", "Why the ground runs straight from each bank top", "Carving the channel")
def no_moat(ctx):
    site = ctx.detail_sites[0]
    cells = [c for c in _new_cells(ctx, site) if c["hydraulic_banks"] is not None and c["hydraulic_banks"].single_cell
             and np.isfinite(c["carve_depth"]) and c["profile"] is not None]
    # a single cell whose bank elevation is well above the ground beside the stream cell
    def gap(c):
        e, s = c["found"], c["spacing"]
        center = e.size // 2
        half = 0.5 * c["hydraulic_banks"].top_width
        ground = min(np.interp([-half, half], new_stations(e.size, s), e))
        return c["bank_elevation"] - ground
    cell = max(cells, key=gap)
    e, s, banks = cell["found"], cell["spacing"], cell["hydraulic_banks"]
    bank = cell["bank_elevation"]
    x = new_stations(e.size, s)
    px, py = np.asarray(cell["profile"][0]), np.asarray(cell["profile"][1])
    # the vertical faces the design first had: the channel between its banks, then straight down to the ground there
    half = 0.5 * banks.top_width
    inside = (px > -half + 1e-9) & (px < half - 1e-9)
    ground_at = np.interp([-half, half], x, e)
    fx = np.concatenate([x[x < -half], [-half, -half], px[inside], [half, half], x[x > half]])
    fy = np.concatenate([e[x < -half], [ground_at[0], bank], py[inside], [bank, ground_at[1]], e[x > half]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    for ax, (sx, sy, title, color) in zip(axes, ((fx, fy, "vertical faces at the banks (not used)", ACCENT),
                                                 (px, py, "the new profile: straight from the bank tops", NEW))):
        between_banks(ax, s, s, s, curves=[(x, e), (sx, sy)], margin=1.3)
        water(ax, sx, sy, bank + 0.3)
        _plot_ground(ax, x, e, color=GROUND, ls="--", marker="o", ms=4, label="ground at the ordinates")
        ax.plot(sx, sy, color=color, lw=2.0, label="profile")
        mark_banks(ax, banks.left, banks.right)
        ax.axhline(bank, color=BANK, lw=0.7, ls=":")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("metres from the stream cell")
        ax.legend(loc="upper center", fontsize=7)
    axes[0].annotate("a moat: the ground beside the\nbank, inside the stream cell,\nfar below the bank top",
                     xy=(-half - 0.25 * s, ground_at[0] + 0.1), xytext=(0.03, 0.12), textcoords="axes fraction",
                     fontsize=7, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    axes[0].set_ylabel("elevation (m)")
    return fig, dict(caption=(
        f"A single cell of {ctx.site_label(site)} whose smoothed bank elevation is {gap(cell):.2f} m above the ground "
        "beside it. The first design joined each bank top to the ground by a vertical face. With bank elevations "
        "that put the bank at the smoothed level and the ground just beside it, still inside the stream cell, far "
        "below: a moat along every such channel that water fills before it tops the banks. The ground now runs "
        "straight from each bank top to the first ordinate beyond it, as it ran between the ordinates before."),
                 stats=dict(cell=(cell["row"], cell["col"]), gap=rounded(gap(cell), 2)))


# --- Smoothing along the network -------------------------------------------------------------------------------------


def _reach_profiles(ctx, site, reach):
    """The new and legacy values along a reach, at the new stations: stream cell elevations, observations and
    smoothed bank elevations, depths and beds."""
    legacy, new = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
    info = new["reaches"][reach]
    by_cell, stations = _new_stations(new, reach)
    smoothed, channel = info["smoothed"], info["channel"]
    cells = [new["cells"][k] for k in info["cells"]]
    out = dict(stations=stations, thalweg=np.array([c["found"][c["found"].size // 2] for c in cells]),
               observations=smoothed["observations"], anchors=smoothed["anchors"],
               bank=smoothed["bank_elevations"], depths=np.asarray(info["depths"], float),
               beds=None if channel is None else channel["beds"],
               carved_bed=np.array([c["final"][c["final"].size // 2] if np.isfinite(c["carve_depth"]) else np.nan
                                    for c in cells]))
    old = []
    for cell in legacy["cells"]:
        if cell is None or (cell["row"], cell["col"]) not in by_cell:
            continue
        b = cell["banks"]
        old.append((by_cell[(cell["row"], cell["col"])], b.get("raw_bank_elevation", np.nan),
                    b.get("smoothed_bank_elevation", np.nan), b.get("smoothed_bed_elevation", np.nan),
                    cell["final"][0][0], cell["found"][0][0]))
    out["legacy"] = np.array(sorted(old), dtype=float)
    return out


@figure("BS1", "Bank elevations along a reach", "Smoothing along the network")
def bank_elevations_along_reach(ctx):
    from arc.bathymetry import smooth_bank_elevations
    site = ctx.detail_sites[0]
    reach = _pick_reach(ctx, site)
    p = _reach_profiles(ctx, site, reach)
    order = np.argsort(p["stations"])
    network, reaches, _ = _rebuilt_reaches(ctx, site)
    grid = ctx.grid(site)
    legacy_method = smooth_bank_elevations(network, reaches, grid.dx, grid.dy, method="legacy")[reach]
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    st = p["stations"][order]
    ax.plot(st, p["thalweg"][order], color=GROUND, lw=0.9, label="the stream cells' ground (new)")
    ax.plot(st, p["observations"][order], "o", color=NEW, ms=3, mfc="none", label="new observations")
    ax.plot(st, p["bank"][order], color=NEW, lw=1.8, label="new smoothed bank elevation")
    ax.plot(st, np.asarray(legacy_method.bank_elevations)[order], color=NEW, lw=1.0, ls="--",
            label="legacy's smoothing of the new observations")
    old = p["legacy"]
    ax.plot(old[:, 0], old[:, 1], "x", color=LEGACY, ms=3.5, label="legacy observations")
    ax.plot(old[:, 0], old[:, 2], color=LEGACY, lw=1.4, label="legacy smoothed bank elevation")
    ax.set_xlabel("metres along the reach from its upstream end")
    ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    return fig, dict(caption=(
        f"The longest reach of {ctx.site_label(site)}. Each cross section's lower bank above its stream cell is an "
        "observation of the reach's bank elevation (outliers beyond the 2nd and 97th percentiles left out). "
        "Legacy's smoothing, and the new code's until 2026-09-26, made the bank elevation a line falling to the "
        "reach's outlet, refitted through every observation below it, with the network making each outlet lower "
        "than those upstream: a lower envelope. The new one is the DEM's water surface along the reach, fitted to "
        "fall, plus the 10th percentile of the observations' heights above their stream cells within 500 m, fitted "
        "to fall again, and never below the stream cell (BS2 to BS4). The new observations are the ground at the "
        "banks, where legacy took its bank ordinates' elevations, and a single cell's are the neighbouring "
        "ordinates' ground, as legacy's were."),
                 stats=dict(site=site, reach=int(reach),
                            new_above_stream=rounded(np.nanmedian(p["bank"] - p["thalweg"]), 2),
                            legacy_method_above_stream=rounded(
                                np.nanmedian(np.asarray(legacy_method.bank_elevations) - p["thalweg"]), 2)))


def _smooth_made_up(thalweg, observations, cell=30.0, phantom=False, width=60.0, method="water_plus_height"):
    """The bank smoothing on a made-up reach of cross sections one cell apart along a row, flowing into a
    reach with one cross section beyond its end (so its downstream end is known), and with phantom an inflow with no
    cross sections of its own. Every channel is the same width, so the width filter leaves the banks alone. method is
    smooth_bank_elevations' method, or a smoothing of its own, called as smooth_bank_elevations is."""
    import networkx as nx
    from arc.bathymetry import Banks, ReachSections, smooth_bank_elevations
    from arc.xsection.xsection import XSection
    thalweg, observations = np.asarray(thalweg, float), np.asarray(observations, float)
    count = thalweg.size
    network = nx.DiGraph()
    network.add_node(1, length=count * cell)
    network.add_node(2, length=cell)
    network.add_edge(1, 2)
    if phantom:
        network.add_node(0, length=2000.0)
        network.add_edge(0, 1)

    def section(z):
        e = np.full(9, z + 5.0)
        e[4] = z
        return XSection(e, np.full(9, 0.035), cell)

    def banks(z):
        return Banks("test", width / 2, width / 2, z, z, False, True)
    # the reach downstream has no observation of its own (its banks at its stream cell), as below Du Page's
    last = float(thalweg[-1]) - 0.01
    reaches = {1: ReachSections(np.zeros(count, np.int64), np.arange(count), [section(z) for z in thalweg],
                                [banks(z) for z in observations]),
               2: ReachSections(np.zeros(1, np.int64), np.array([count]), [section(last)], [banks(last)])}
    if callable(method):  # another smoothing, such as vc_variants.joseph_smoothing
        return method(network, reaches, cell, cell)[1]
    return smooth_bank_elevations(network, reaches, cell, cell, method=method)[1]


@figure("BS2", "The bank smoothing on made-up reaches", "Smoothing along the network")
def bank_smoothing_synthetic(ctx):
    cell, count = 30.0, 300
    x = np.arange(count) * cell
    rng = np.random.default_rng(21)
    steady = 100.0 - 0.001 * x
    mild_then_steep = np.where(x < 6000, 165.0 - 0.0001 * x, 164.4 - 8.0 * (x - 6000) / 3000)
    cases = (("a steady reach falling 0.1%", steady, False),
             ("mild, then steep (as the Du Page reach)", mild_then_steep, False),
             ("the steady reach, below an inflow with no cross sections", steady, True))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    stats = {}
    for ax, (title, water, phantom) in zip(axes, cases):
        rng = np.random.default_rng(21)
        stream = water + rng.normal(0.0, 0.15, count)  # the DEM's water surface at the stream cells
        tops = water + 1.5  # the true bank tops
        observed = tops + rng.normal(0.0, 0.3, count)
        walls = rng.random(count) < 0.4  # single cells, whose observations are the neighbouring ordinates' ground
        observed[walls] += rng.uniform(1.0, 4.0, walls.sum())
        banks = {}
        for method in ("legacy", "water_plus_height"):
            result = _smooth_made_up(stream, observed, cell, phantom=phantom, method=method)
            banks[method] = np.asarray(result.bank_elevations)[result.order]
        ax.plot(x, stream, color=GROUND, lw=0.8, label="the stream cells (the DEM's water)")
        ax.plot(x, tops, color=GROUND, lw=1.0, ls="--", label="the true bank tops")
        ax.plot(x, observed, "o", color=NEW, ms=2.2, mfc="none", alpha=0.5, label="each cross section's observation")
        ax.plot(x, banks["legacy"], color=LEGACY, lw=1.8, label="legacy's smoothing (a lower envelope)")
        ax.plot(x, banks["water_plus_height"], color=NEW, lw=1.8, label="the new: water + 10th-percentile height")
        lines = []
        for method, label in (("legacy", "legacy's"), ("water_plus_height", "new")):
            under = float(np.median(tops - banks[method]))
            below = float(np.mean(banks[method] < stream))
            lines.append(f"{label}: {under:+.2f} m under the banks, below the water at {below:.0%}")
            stats.setdefault(title, {})[method] = dict(median_under_banks=rounded(under, 2),
                                                       below_stream=rounded(below, 3))
        note(ax, "\n".join(lines), loc="lower left")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("metres along the reach")
    axes[0].set_ylabel("elevation (m)")
    axes[0].legend(loc="upper right", fontsize=6.3)
    return fig, dict(caption=(
        "Both bank smoothings on made-up reaches of 300 cross sections 30 m apart: the true banks 1.5 m above the "
        "DEM's water, observations 0.3 m about them, and 40% single cells whose observations are the neighbouring "
        "ordinates' ground, 1 to 4 m higher. Legacy's (the new code's too, until 2026-09-26) is a line falling to the "
        "reach's lowest observation, refitted through every observation below it, so it runs along the lowest "
        "observations, under the banks. Where a reach runs mild then steep, the line from its top to its outlet "
        "passes under the whole mild stretch, and no observation is below it to refit it. And an inflow with no "
        "cross sections gets an outlet extrapolated from the reach itself, its own lowest observation plus "
        "the minimum grade, which then caps the reach: its bank elevation stays at its outlet's all the way up. The "
        "new smoothing follows the stream cells' water surface, fitted to fall downstream, and adds the 10th "
        "percentile of the observations' heights within 500 m, so it takes the reach's shape from the water and none "
        "from the network; here it sits a little under the true banks, since a low percentile of noisy heights is "
        "below their centre, but it never goes below the water."), stats=stats)


LEGACY_DEM_RAISE = 100.0  # legacy raised a DEM with any elevation below 0 by this much (arc.rating_curve's notes)


def legacy_offset(capture) -> float:
    """How far a legacy-style capture's bank elevations are above its sampled ground's frame: legacy raised a DEM with
    any elevation below 0 (such as a nodata value read as data, S6) by LEGACY_DEM_RAISE, and its bank elevations are
    in that raised frame, where the ground it samples isn't. No bank is tens of metres above its stream cell, so a
    median raw bank that far up is the raise."""
    cells = [c for c in capture["cells"] if c is not None]
    heights = np.array([c["banks"].get("raw_bank_elevation", np.nan) - c["found"][0][0] for c in cells], float)
    heights = heights[np.isfinite(heights)]
    return LEGACY_DEM_RAISE if heights.size and np.median(heights) > LEGACY_DEM_RAISE / 2 else 0.0


def _rebuilt_reaches(ctx, site):
    """The site's ReachSections as the new pipeline smoothed them (from the capture), and its network."""
    from arc import pipeline
    from arc.bathymetry import ReachSections
    from arc.xsection.xsection import XSection
    new = ctx.capture("as_configured", "new", site)
    configs = ctx.configs(site)
    network = pipeline.stream_network(configs, pipeline.read_stream_layer(configs))
    reaches = {}
    for reach, info in new["reaches"].items():
        if info["found_banks"] is None:
            continue
        cells = [new["cells"][k] for k in info["cells"]]
        reaches[reach] = ReachSections(np.array([c["row"] for c in cells]), np.array([c["col"] for c in cells]),
                                       [XSection(c["found"].copy(), np.full(c["found"].size, 0.035), c["spacing"])
                                        for c in cells], list(info["found_banks"]))
    return network, reaches, new


@figure("BS3", "Where the smoothed bank elevation ends up on the real sites", "Smoothing along the network")
def bank_elevation_real(ctx):
    from arc.bathymetry import smooth_bank_elevations
    heights = {"legacy": [], "legacy_method": [], "new": []}
    shares = []
    largest_rebuild, mismatched = 0.0, 0  # the default smoothing rebuilt from the capture, against the capture
    du_page = None
    for site in ctx.sites:
        new = ctx.capture("as_configured", "new", site)
        legacy = ctx.capture("as_configured", "legacy", site)
        if new is None:
            continue
        network, reaches, _ = _rebuilt_reaches(ctx, site)
        grid = ctx.grid(site)
        old_way = smooth_bank_elevations(network, reaches, grid.dx, grid.dy, method="legacy")
        rebuilt = smooth_bank_elevations(network, reaches, grid.dx, grid.dy)  # the pipeline's, from the capture
        with_legacy_method, with_new = [], []
        for reach, info in new["reaches"].items():
            if reach not in old_way:
                continue
            thalweg = np.array([new["cells"][k]["found"][new["cells"][k]["found"].size // 2] for k in info["cells"]])
            captured = np.array([new["cells"][k]["bank_elevation"] for k in info["cells"]], dtype=float)
            with_legacy_method.append(np.asarray(old_way[reach].bank_elevations) - thalweg)
            with_new.append(captured - thalweg)
            again = np.asarray(rebuilt[reach].bank_elevations, dtype=float)
            both = np.isfinite(captured) & np.isfinite(again)
            mismatched += int(np.sum(np.isfinite(captured) != np.isfinite(again)))
            if both.any():
                largest_rebuild = max(largest_rebuild, float(np.max(np.abs(captured[both] - again[both]))))
        values = np.concatenate(with_legacy_method) if with_legacy_method else np.empty(0)
        values = values[np.isfinite(values)]
        heights["legacy_method"].append(values)
        mine = np.concatenate(with_new) if with_new else np.empty(0)
        heights["new"].append(mine[np.isfinite(mine)])
        raise_ = 0.0 if legacy is None else legacy_offset(legacy)
        old = np.array([c["banks"].get("smoothed_bank_elevation", np.nan) - raise_ - c["found"][0][0]
                        for c in ([] if legacy is None else legacy["cells"]) if c is not None])
        heights["legacy"].append(old[np.isfinite(old)])
        shares.append((site, float(np.mean(values < 0)) if values.size else np.nan,
                       float(np.mean(old[np.isfinite(old)] < 0)) if np.isfinite(old).any() else np.nan))
        if site.startswith("Du_Page"):
            du_page = (site, network, reaches, old_way, new)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1, 1.1, 1.5]))
    ax = axes[0]
    bins = np.arange(-8, 8.01, 0.25)
    below = {}
    for key, color, label in (("legacy", LEGACY, "legacy"), ("legacy_method", ACCENT, "new code, legacy's smoothing"),
                              ("new", NEW, "new code, the new smoothing")):
        values = np.concatenate(heights[key])
        below[key] = float(np.mean(values < 0))
        ax.hist(np.clip(values, bins[0], bins[-1]), bins=bins, histtype="step", color=color, lw=1.3,
                label=f"{label}: below the stream cell at {below[key]:.0%}")
    ax.axvline(0, color=GROUND, lw=0.8)
    ax.set_xlabel("smoothed bank elevation − the stream cell (m)")
    ax.set_ylabel("cross sections, all sites")
    ax.legend(loc="upper left", fontsize=6.5)
    ax = axes[1]
    shares.sort(key=lambda s: -s[1])
    y = np.arange(len(shares))
    ax.barh(y, [s[1] for s in shares], color=ACCENT, height=0.8, label="new code, legacy's smoothing")
    ax.plot([s[2] for s in shares], y, "|", color=LEGACY, ms=6, mew=1.5, label="legacy")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0].replace("_", " ")[:28] for s in shares], fontsize=4.6)
    ax.invert_yaxis()
    ax.set_xlabel("share of the site's cross sections whose bank elevation is below the stream cell\n"
                  "(the new smoothing: none, at every site)")
    ax.legend(loc="lower right", fontsize=6.8)
    stats = dict(below_legacy=rounded(below["legacy"], 3), below_legacy_method=rounded(below["legacy_method"], 3),
                 below_new=rounded(below["new"], 3), rebuild_largest_difference_m=largest_rebuild,
                 rebuild_mismatched=mismatched, reproduces_capture=bool(largest_rebuild < 1e-9 and mismatched == 0))
    ax = axes[2]
    if du_page is not None:
        site, network, reaches, old_way, new = du_page
        reach = 760524857 if 760524857 in old_way else max(old_way, key=lambda r: len(reaches[r].sections))
        info = old_way[reach]
        order = np.asarray(info.order)
        st = np.asarray(info.stations)
        thalweg = np.array([xs.elevations[xs.elevations.size // 2] for xs in reaches[reach].sections])[order]
        now = np.array([new["cells"][k]["bank_elevation"] for k in new["reaches"][reach]["cells"]])[order]
        ax.plot(st, thalweg, color=GROUND, lw=0.9, label="the stream cells")
        ax.plot(st, np.asarray(info.observations)[order], "o", color=NEW, ms=2.2, mfc="none", alpha=0.6,
                label="observations")
        ax.plot(st, np.asarray(info.bank_elevations)[order], color=ACCENT, lw=1.8, label="legacy's smoothing")
        ax.plot(st, now, color=NEW, lw=1.8, label="the new smoothing")
        ax.set_xlabel("metres along the reach from its upstream end")
        ax.set_ylabel("elevation (m)")
        ax.set_title(f"{ctx.site_label(site).split(' (')[0]}, reach {reach}", fontsize=8.5)
        ax.legend(loc="upper right", fontsize=6.3)
        stats.update(du_page_reach=int(reach),
                     du_page_under_stream_m=rounded(float(np.median(thalweg - np.asarray(info.bank_elevations)[order])), 2),
                     du_page_new_above_stream_m=rounded(float(np.median(now - thalweg)), 2))
    return fig, dict(caption=(
        "Left: the smoothed bank elevation, the level each channel is carved below with Bathy_Use_Banks, less its "
        "stream cell, over every cross section of the sites. With legacy's smoothing it is below the stream cell, "
        f"under the DEM's water, at {below['legacy']:.0%} of legacy's cross sections and {below['legacy_method']:.0%} "
        "of the new code's; the new smoothing never goes below it. Middle: the share at each site. Right: the Du Page "
        "reach of BS4. One of its two inflows has no cross sections, so legacy's network gives it this reach's own "
        "outlet elevation plus the minimum grade over the reach, and as the lower of the two inflows it caps the "
        f"reach, whose bank elevation then runs a median {stats.get('du_page_under_stream_m', float('nan')):.1f} m "
        "under its stream cells; the new smoothing follows the reach down from its stream cells, "
        f"{stats.get('du_page_new_above_stream_m', float('nan')):.1f} m above them at the median."), stats=stats)


def _reach_with_cell(capture, row, col):
    k = next((i for i, c in enumerate(capture["cells"]) if c is not None and (c["row"], c["col"]) == (row, col)), None)
    return None if k is None else capture["cells"][k]["reach"]


@figure("BS4", "The new bank smoothing on real reaches", "Smoothing along the network")
def new_smoothings(ctx):
    from functools import partial

    import vc_variants
    from arc.bathymetry import smooth_bank_elevations
    cases = (("Du_Page", None, 760524857, "mild, then steep, below a phantom inflow"),
             ("Flint", (641, 669), None, "where the carve under legacy's smoothing dug a 110 m box"),
             ("Cuyahoga", (708, 759), None, "C4b's reach"))
    legacy_way = partial(smooth_bank_elevations, method="legacy")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    stats = {}
    for ax, (prefix, cell, reach, title) in zip(axes, cases):
        site = next((s for s in ctx.sites if s.startswith(prefix)), None)
        if site is None:
            ax.set_visible(False)
            continue
        network, reaches, new = _rebuilt_reaches(ctx, site)
        if reach is None:
            reach = _reach_with_cell(new, *cell)
        grid = ctx.grid(site)
        lines = {"legacy's smoothing": legacy_way(network, reaches, grid.dx, grid.dy)[reach],
                 "the new: water surface + 10th-percentile height": smooth_bank_elevations(network, reaches, grid.dx,
                                                                                          grid.dy)[reach],
                 "the same, free to rise downstream": vc_variants.bank_reference(
                     smooth_bank_elevations, "water_plus_height", 0.1, 500.0, "none")(network, reaches, grid.dx,
                                                                                      grid.dy)[reach],
                 "falling fit of the banks' 25th percentile": vc_variants.bank_reference(
                     legacy_way, "falling_quantile", 0.25)(network, reaches, grid.dx, grid.dy)[reach]}
        info = lines["legacy's smoothing"]
        order, st = np.asarray(info.order), np.asarray(info.stations)
        thalweg = np.array([xs.elevations[xs.elevations.size // 2] for xs in reaches[reach].sections])[order]
        ax.plot(st, thalweg, color=GROUND, lw=0.9, label="the stream cells")
        ax.plot(st, np.asarray(info.observations)[order], "o", color=NEW, ms=2.2, mfc="none", alpha=0.5,
                label="observations")
        styles = {"legacy's smoothing": (ACCENT, "-", 1.8), "the new: water surface + 10th-percentile height": (NEW, "-", 2.0),
                  "the same, free to rise downstream": (GROUND, "--", 1.1),
                  "falling fit of the banks' 25th percentile": ("#CC79A7", ":", 1.6)}
        for label, result in lines.items():
            color, style, width = styles[label]
            bank = np.asarray(result.bank_elevations)[order]
            ax.plot(st, bank, color=color, ls=style, lw=width, label=label)
            stats.setdefault(site, {})[label] = rounded(np.nanmedian(bank - thalweg), 2)
        observed = np.asarray(info.observations)[order]
        low, high = np.nanpercentile(np.r_[thalweg, observed[np.isfinite(observed)]], [0, 99])
        ax.set_ylim(low - 1.5, high + 1.0)
        ax.set_title(f"{ctx.site_label(site).split(' (')[0].split(',')[0]}, reach {reach}\n{title}", fontsize=8.5)
        ax.set_xlabel("metres along the reach from its upstream end")
        ax.legend(loc="upper right", fontsize=6.0)
    axes[0].set_ylabel("elevation (m)")
    return fig, dict(caption=(
        "The new bank smoothing (since 2026-09-26) on three reaches where legacy's goes wrong: the DEM's water "
        "surface along the reach, fitted to fall downstream, plus the 10th percentile of the observations' heights "
        "above their stream cells within 500 m, fitted to fall again, and never below the stream cell. It takes "
        "nothing from the network, so no inflow caps a reach, and it follows a reach that runs mild and then steep. "
        "Free to rise downstream (grey), it follows the stream cells' bumps. The falling fit of the banks' 25th "
        "percentile (dotted) was the other design tried; F2 has what each does to the flood maps. The observations "
        "are mostly single cells' neighbouring ground and width-to-depth shoulders, well above the banks, hence a "
        "low percentile."), stats=stats)


def _carved_beds(new, smoothed, channels):
    """Each carved cross section's bed (bank elevation less its smoothed depth) less its stream cell, by reach."""
    beds = {}
    for reach, channel in channels.items():
        bank = np.asarray(smoothed[reach].bank_elevations, float)
        values = []
        for i, k in enumerate(new["reaches"][reach]["cells"]):
            c = new["cells"][k]
            if c is None or not np.isfinite(c["carve_depth"]):
                values.append(np.nan)
                continue
            values.append(bank[i] - channel.depths[i] - c["found"][c["found"].size // 2])
        beds[reach] = np.array(values)
    return beds


@figure("BS5", "Where the carved bed ends up: the fills the new smoothing makes", "Smoothing along the network")
def carved_beds(ctx):
    from functools import partial

    import vc_variants
    from arc.bathymetry import smooth_bank_elevations
    ways = {"legacy's smoothing": (partial(smooth_bank_elevations, method="legacy"), ACCENT, "-"),
            "the new smoothing": (smooth_bank_elevations, NEW, "-"),
            "the new, free to rise downstream": (vc_variants.bank_reference(
                smooth_bank_elevations, "water_plus_height", 0.1, 500.0, "none"), GROUND, "--")}
    beds = {way: [] for way in ways}
    shares = []
    example = None
    for site in ctx.sites:
        if ctx.capture("as_configured", "new", site) is None:
            continue
        row = [site]
        for way, (smooth, _, _) in ways.items():
            new, smoothed, channels = _channels_with(ctx, site, smooth)
            by_reach = _carved_beds(new, smoothed, channels)
            values = np.concatenate(list(by_reach.values())) if by_reach else np.empty(0)
            values = values[np.isfinite(values)]
            beds[way].append(values)
            row.append(float(np.mean(values > 0.5)) if values.size else np.nan)
            if site.startswith("Salt_Creek"):
                example = example or {}
                example[way] = (new, smoothed, channels)
        shares.append(row)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), gridspec_kw=dict(width_ratios=[1, 1.05, 1.4]))
    ax = axes[0]
    bins = np.arange(-8, 8.01, 0.25)
    stats = {}
    for way, (_, color, style) in ways.items():
        values = np.concatenate(beds[way])
        stats[way] = dict(cross_sections=int(values.size), median=rounded(np.median(values), 2),
                          above=rounded(np.mean(values > 0), 3), above_half_metre=rounded(np.mean(values > 0.5), 3),
                          above_2m=rounded(np.mean(values > 2), 3), below_2m=rounded(np.mean(values < -2), 3))
        ax.hist(np.clip(values, bins[0], bins[-1]), bins=bins, histtype="step", color=color, ls=style, lw=1.3,
                label=f"{way}: above it at {np.mean(values > 0):.0%}")
    ax.axvline(0, color=GROUND, lw=0.8)
    ax.set_xlabel("carved bed − the stream cell (m); positive: the channel filled")
    ax.set_ylabel("carved cross sections, all sites")
    ax.legend(loc="upper left", fontsize=6.5)
    ax = axes[1]
    shares.sort(key=lambda r: -r[2])
    y = np.arange(len(shares))
    ax.barh(y, [r[2] for r in shares], color=NEW, height=0.8, label="the new smoothing")
    ax.plot([r[1] for r in shares], y, "|", color=ACCENT, ms=6, mew=1.5, label="legacy's smoothing")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0].replace("_", " ")[:28] for r in shares], fontsize=4.6)
    ax.invert_yaxis()
    ax.set_xlabel("share of the site's carved cross sections\nwhose bed is more than 0.5 m above the stream cell")
    ax.legend(loc="lower right", fontsize=6.8)
    ax = axes[2]
    if example is not None:
        new = example["the new smoothing"][0]
        reach = max(example["the new smoothing"][2], key=lambda r: len(new["reaches"][r]["cells"]))
        site = next(s for s in ctx.sites if s.startswith("Salt_Creek"))
        p = _reach_profiles(ctx, site, reach)
        order = np.argsort(p["stations"])
        st = p["stations"][order]
        ax.plot(st, p["thalweg"][order], color=GROUND, lw=1.0, label="the stream cells (the DEM's water)")
        for way, (_, color, style) in ways.items():
            _, smoothed, channels = example[way]
            if reach not in channels:
                continue
            bank = np.asarray(smoothed[reach].bank_elevations, float)
            bed = bank - np.asarray(channels[reach].depths, float)
            ax.plot(st, bank[order], color=color, ls=style, lw=1.6, label=f"{way}: bank elevation")
            ax.plot(st, bed[order], color=color, ls=style, lw=0.9, alpha=0.8, label="and its bed")
            stats.setdefault("salt_creek_bed_above_stream_m", {})[way] = rounded(float(np.nanmedian(bed - p["thalweg"])), 2)
        ax.set_xlabel("metres along the reach from its upstream end")
        ax.set_ylabel("elevation (m)")
        ax.set_title(f"{ctx.site_label(site).split(' (')[0]}, reach {reach}", fontsize=8.5)
        ax.legend(loc="upper right", fontsize=6.0)
        stats["salt_creek_reach"] = int(reach)
    legacy, new, free = (stats[w] for w in ways)
    full = sum(r[2] >= 0.999 for r in shares)
    stats["sites_all_filled"] = [r[0] for r in shares if r[2] >= 0.999]
    return fig, dict(caption=(
        "Where each smoothing leaves the carved bed, the smoothed bank elevation less the channel's depth after the "
        f"bed smoothing, against the DEM's water at the stream cell, over the carved cross sections of the {len(shares)} "
        "sites "
        "(all three from the new code's cross sections and depths). Legacy's smoothing ran the bank elevation along "
        f"the reach's lowest banks and often under the water, so the bed was more than 2 m under the water at "
        f"{legacy['below_2m']:.0%} of cross sections. The new one puts it at the water plus the 10th percentile of "
        "the nearby banks' heights, so where the channel's depth is less than that height the carve fills the "
        f"channel: the bed is above the stream cell at {new['above']:.0%} of cross sections (legacy's smoothing "
        f"{legacy['above']:.0%}) and more than 2 m above it at {new['above_2m']:.0%}; at {full} sites, every one is "
        "more than 0.5 m above it. Free to rise downstream, the new smoothing fills about as much "
        f"({free['above']:.0%}). Bank-based bathymetry may raise the DEM (decided on 2025-09-18), and F2 has what the "
        "fills do to the flood maps. Right: the longest reach of Salt Creek, one of the sites the new smoothing "
        "fills throughout."), stats=stats)


def _two_reach_case(banks, depths, cell=10.0, max_bed_grade=0.01):
    """A reach of cross sections one cell apart along a row, flowing into a reach with one cross section beyond its
    end (so both codes know which way it flows), with these smoothed bank elevations and depths (the last of each
    for the reach downstream). The new depths and beds of the first reach, and legacy's."""
    import networkx as nx

    import arc.Automated_Rating_Curve_Generator as legacy
    from arc.bathymetry import Banks, ReachSections, SmoothedReach, smooth_channel_depths
    from arc.bathymetry.smoothing import order_reach
    from arc.xsection.xsection import XSection
    banks, depths = np.asarray(banks, float), np.asarray(depths, float)
    count = banks.size - 1
    cells = [(1, 0, k) for k in range(count)] + [(2, 0, count)]
    network = nx.DiGraph()
    network.add_node(1, length=100.0)
    network.add_node(2, length=100.0)
    network.add_edge(1, 2)
    no_banks = Banks("none", np.nan, np.nan, np.nan, np.nan, False, False)
    indices = {1: list(range(count)), 2: [count]}
    positions = {r: (np.array([cells[k][1] for k in ks]), np.array([cells[k][2] for k in ks])) for r, ks in
                 indices.items()}
    reaches, smoothed = {}, {}
    for reach, ks in indices.items():
        rows, cols = positions[reach]
        sections = [XSection(np.array([105.0, 90.0, 105.0]), np.full(3, 0.035), cell) for _ in ks]
        reaches[reach] = ReachSections(rows, cols, sections, [no_banks] * len(ks))
        order, stations = order_reach(network, reach, rows, cols, cell, cell, positions, banks[ks])
        smoothed[reach] = SmoothedReach([no_banks] * len(ks), banks[ks], banks[ks], np.zeros(len(ks), bool), order,
                                        stations, None)
    new = smooth_channel_depths(network, reaches, smoothed, {r: list(depths[ks]) for r, ks in indices.items()},
                                cell, cell, use_banks=True, max_bed_grade=max_bed_grade)[1]
    saved = {name: getattr(legacy, name) for name in ("_CELL_COMIDS", "_CELL_SOURCE_STREAM_IDS", "_CELL_ROWS",
                                                      "_CELL_COLS", "_build_reach_network_graph")}
    try:  # legacy's module state, set as its own run sets it, then put back
        legacy._CELL_COMIDS = np.array([c[0] for c in cells], dtype=np.int64)
        legacy._CELL_SOURCE_STREAM_IDS = None
        legacy._CELL_ROWS = np.array([c[1] for c in cells], dtype=np.int64)
        legacy._CELL_COLS = np.array([c[2] for c in cells], dtype=np.int64)
        legacy._build_reach_network_graph = lambda *args: (network, {})
        records = [{"bank_search_result": {"smoothed_bank_elevation": float(b), "bathymetry_depth": float(d)}}
                   for b, d in zip(banks, depths)]
        legacy._smooth_reach_bathymetry_depths(records, {"dx": cell, "dy": cell})
        legacy._smooth_reach_excavated_bed_elevations(records, {"dx": cell, "dy": cell})
    finally:
        for name, value in saved.items():
            setattr(legacy, name, value)
    old_depths = np.array([r["bank_search_result"]["bathymetry_depth"] for r in records])[:count]
    old_beds = np.array([r["bank_search_result"].get("smoothed_bed_elevation", np.nan) for r in records])[:count]
    return new.depths, new.beds, old_depths, old_beds


CAPS = (("legacy's cap, 1 cm per metre", 0.01, LEGACY, "-", 1.6),
        ("MAX_SLOPE, 50 cm per metre (the new default)", 0.5, NEW, "-", 1.8),
        ("no cap (Bathy_Bed_Cap false)", None, ACCENT, ":", 1.6))


def _site_channels(ctx, site, grades):
    """The site's bed smoothing redone from its captured bank smoothing and depths at each bed grade: {grade:
    {reach: ChannelDepths}}."""
    from arc.bathymetry import SmoothedReach, smooth_channel_depths
    network, reaches, new = _rebuilt_reaches(ctx, site)
    smoothed, depths = {}, {}
    for reach, info in new["reaches"].items():
        if reach in reaches and info["smoothed"] is not None and info["depths"] is not None \
                and np.isfinite(info["smoothed"]["bank_elevations"]).any():
            smoothed[reach] = SmoothedReach(**info["smoothed"])
            depths[reach] = list(info["depths"])
    reaches = {r: reaches[r] for r in smoothed}
    return {grade: smooth_channel_depths(network, reaches, smoothed, depths, new["dx"], new["dy"], use_banks=True,
                                         max_bed_grade=grade) for grade in grades}, new


@figure("BD1", "The bed smoothing's cap: legacy's 1%, MAX_SLOPE (the new default), or none",
        "Smoothing along the network")
def bed_cap(ctx):
    grades = [c[1] for c in CAPS]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    stats = {}
    # across the sites: how many channels each cap changes, and the reach it changes most
    changed = {g: [0, 0, 0.0] for g in grades[1:]}
    most = (0.0, None, None)
    for site in ctx.sites:
        if ctx.capture("as_configured", "new", site) is None:
            continue
        channels, _ = _site_channels(ctx, site, grades)
        for grade in grades[1:]:
            for reach, base in channels[0.01].items():
                other = channels[grade][reach]
                d = np.abs(np.asarray(other.depths) - np.asarray(base.depths))
                changed[grade][0] += int(np.sum(d > 0.01))
                changed[grade][1] += d.size
                changed[grade][2] = max(changed[grade][2], float(np.nanmax(d)) if d.size else 0.0)
                if grade == grades[1] and d.size and np.nanmax(d) > most[0]:
                    most = (float(np.nanmax(d)), site, reach)
    for grade, (count, total, largest) in changed.items():
        stats[f"vs_{grade}"] = dict(changed_over_1cm=count, sections=total, largest_m=rounded(largest, 2))

    # Du Page, where the reach began metres below the bed flowing in with legacy's bank smoothing
    du_page = next((s for s in ctx.sites if s.startswith("Du_Page")), None)
    if du_page is not None:
        channels, _ = _site_channels(ctx, du_page, grades)
        if 760524857 in channels[0.01]:
            p = _reach_profiles(ctx, du_page, 760524857)
            order = np.argsort(p["stations"])
            stats["du_page_inflow_step_m"] = rounded(float(np.asarray(channels[0.01][760524857].beds)[order][0])
                                                     - float(np.asarray(channels[None][760524857].beds)[order][0]), 1)

    # left: the reach the cap changes most
    site, reach = most[1], most[2]
    if site is None:
        site = ctx.detail_sites[0]
    channels, new = _site_channels(ctx, site, grades)
    if reach is None:
        reach = max(channels[0.01], key=lambda r: len(channels[0.01][r].depths))
    p = _reach_profiles(ctx, site, reach)
    order = np.argsort(p["stations"])
    st = p["stations"][order]
    ax = axes[0]
    ax.plot(st, p["thalweg"][order], color=GROUND, lw=0.9, label="the stream cells' ground")
    ax.plot(st, p["bank"][order], color=BANK, lw=1.2, label="smoothed bank elevation")
    ax.plot(st, (p["bank"] - p["depths"])[order], ".", color=GROUND, ms=2.5, label="bank elevation − each depth")
    for label, grade, color, style, width in CAPS:
        ax.plot(st, np.asarray(channels[grade][reach].beds)[order], color=color, ls=style, lw=width, label=label)
    old = p["legacy"]
    if old.size:
        ax.plot(old[:, 0], old[:, 3], color=GROUND, lw=1.0, ls="-.", label="legacy's own smoothed bed")
    ax.set_title(f"the reach the cap changes most\n{ctx.site_label(site).split(' (')[0][:40]}, reach {reach}",
                 fontsize=8.5)
    ax.set_xlabel("metres along the reach from its upstream end")
    ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper right", fontsize=6.2)
    deeper = (np.asarray(channels[0.01][reach].depths) - np.asarray(channels[0.5][reach].depths))[order]
    held_down = int(np.sum(deeper > 0.01)) >= int(np.sum(deeper < -0.01))  # legacy's cap carves it deeper
    affected = deeper > 0.01 if held_down else deeper < -0.01
    own_first = float(np.asarray(channels[None][reach].beds)[order][0])
    capped_first = float(np.asarray(channels[0.01][reach].beds)[order][0])
    stats["most_changed"] = dict(site=site, reach=int(reach), largest_m=rounded(most[0], 2), held_down=held_down,
                                 sections=int(affected.sum()), of=int(affected.size),
                                 length_m=rounded(float(np.ptp(st[affected])) if affected.any() else 0.0, 0),
                                 first_bed_capped_minus_own_m=rounded(capped_first - own_first, 2),
                                 own_bed_above_stream_m=rounded(float(np.nanmedian(
                                     np.asarray(channels[None][reach].beds)[order] - p["thalweg"][order])), 2))

    # middle and right: made-up reaches, one steeper than 1% and one mild, then steep
    cell = 10.0
    distance = cell * np.arange(81)
    knick = np.where(distance < 400, 100.0 - 0.0005 * distance,
                     np.maximum(99.8 - 0.05 * (distance - 400), 92.3 - 0.0005 * (distance - 550)))
    cases = ((axes[1], "banks falling 2% (none of the sites has such a reach)", 100.0 - 0.02 * cell * np.arange(41)),
             (axes[2], "mild, then steep: banks falling 0.05%, then 5% over 150 m", knick))
    for ax, title, banks in cases:
        count = banks.size - 1
        x = np.arange(count) * cell
        depths = np.ones(count + 1)
        ax.plot(x, banks[:count], color=BANK, lw=1.2, label="smoothed bank elevation")
        ax.plot(x, banks[:count] - 1.0, ".", color=GROUND, ms=3.5, label="bank elevation − the 1 m depth")
        old_beds = None
        for label, grade, color, style, width in CAPS:
            depth, bed, _, old_beds_here = _two_reach_case(banks, depths, cell, max_bed_grade=grade)
            old_beds = old_beds_here if old_beds is None else old_beds
            ax.plot(x, bed, color=color, ls=style, lw=width, label=label)
            gone = np.flatnonzero(depth <= 1e-9)
            stats.setdefault(title, {})[str(grade)] = dict(
                shallowest=rounded(float(np.min(depth)), 2),
                first_without_channel_m=rounded(x[gone[0]], 1) if gone.size else None)
        ax.plot(x, old_beds, color=GROUND, lw=1.0, ls="-.", label="legacy's own")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("metres along the reach from its upstream end")
        ax.legend(loc="upper right", fontsize=6.2)
    counts = stats["vs_0.5"], stats["vs_None"]
    most_changed = stats["most_changed"]
    fill_note = " (one of the new smoothing's fills, BS5)" if most_changed["own_bed_above_stream_m"] > 0 else ""
    return fig, dict(caption=(
        "Each bed (the bank elevation less the channel's depth) is smoothed with a running median of five cross "
        "sections, and the cap then holds it to so many centimetres per metre along the stream, starting within that "
        "of the lowest bed flowing in. Left: the reach where the cap matters most across the sites. Its own first "
        f"bed is {abs(most_changed['first_bed_capped_minus_own_m']):.1f} m "
        f"{'above' if most_changed['first_bed_capped_minus_own_m'] < 0 else 'below'} the lowest bed flowing in (the "
        "new smoothing takes nothing from the network, so a reach's banks can sit above those of the reach flowing "
        "into it), and at 1 cm a metre the bed is held near the inflow's, so the channel is carved "
        f"{'deeper' if most_changed['held_down'] else 'shallower'} by up to {most_changed['largest_m']:.1f} m over "
        f"{most_changed['sections']} of its {most_changed['of']} cross sections; MAX_SLOPE lets the bed reach its "
        f"own level within a cross section or two, here {abs(most_changed['own_bed_above_stream_m']):.1f} m "
        f"{'above' if most_changed['own_bed_above_stream_m'] > 0 else 'below'} the stream cells at the median"
        f"{fill_note}. "
        "Middle and right: made-up reaches steeper than "
        "1%. At 1% the bed can't follow the banks down, so the channel shallows and then fills in; at MAX_SLOPE "
        "(0.5, the steepest stream slope ARC allows, and the new default since 2026-09-26) it follows them, as with "
        "no cap. Across the sites, MAX_SLOPE's depths differ from legacy's cap's by more than 1 cm at "
        f"{counts[0]['changed_over_1cm']:,} of {counts[0]['sections']:,} cross sections, and no cap's at "
        f"{counts[1]['changed_over_1cm']:,}: at MAX_SLOPE the cap is effectively off. The Du Page reach, which "
        "began metres below the bed flowing in with legacy's bank smoothing (BS3), now begins within "
        f"{stats.get('du_page_inflow_step_m', float('nan')):.1f} m of it."), stats=stats)


@figure("BD2", "A missing bank elevation", "Smoothing along the network")
def missing_bed(ctx):
    cell, count, missing = 10.0, 50, 20
    banks = 100.0 - 0.005 * cell * np.arange(count + 1)
    banks[missing] = np.nan
    depths = np.ones(count + 1)
    new_depths, new_beds, old_depths, old_beds = _two_reach_case(banks, depths, cell)
    x = np.arange(count) * cell
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.plot(x, banks[:count], color=BANK, lw=1.2, label="smoothed bank elevation (one missing)")
    ax.plot(x, banks[:count] - depths[:count], ".", color=GROUND, ms=4, label="bank elevation − depth")
    ax.plot(x, old_beds, color=LEGACY, lw=2.0, label=f"legacy's bed: {int(np.isfinite(old_beds).sum())} of {count}")
    ax.plot(x, new_beds, color=NEW, lw=1.2, label=f"new bed: {int(np.isfinite(new_beds).sum())} of {count}")
    ax.axvline(x[missing], color=GROUND, lw=0.6, ls=":")
    ax.set_xlabel("metres along the reach")
    ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper right", fontsize=7)
    ax = axes[1]
    ax.plot(x, np.nan_to_num(old_depths, nan=-0.05), "x", color=LEGACY, ms=5,
            label=f"legacy: {int(np.isnan(old_depths).sum())} of {count} NaN (drawn at −0.05)")
    ax.plot(x, new_depths, "o", color=NEW, ms=3, label="new")
    ax.axvline(x[missing], color=GROUND, lw=0.6, ls=":")
    ax.set_xlabel("metres along the reach")
    ax.set_ylabel("depth to carve (m)")
    ax.set_ylim(-0.15, 1.2)
    ax.legend(loc="lower right", fontsize=7)
    return fig, dict(caption=(
        f"A reach of {count} cross sections whose {missing + 1}st has no smoothed bank elevation (legacy's own test "
        "case: the reach flows into a reach with one cross section, so both codes know its downstream end). Legacy's "
        "running median took the NaN into the beds within two cross sections of it, and its cap then carried it to "
        "every bed downstream, and back up; a NaN depth carves nothing. Here a missing bed is left out of the median "
        "and the cap, and that cross section keeps its depth."),
                 stats=dict(legacy_nan_depths=int(np.isnan(old_depths).sum()),
                            new_nan_depths=int(np.isnan(new_depths).sum()),
                            new_beds=int(np.isfinite(new_beds).sum())))


@figure("BD3", "A reach on its own: legacy smoothed its bed in raster order", "Smoothing along the network")
def raster_order_bed(ctx):
    site = next((s for s in ctx.detail_sites if s.startswith("South_Fork_Peachtree")), ctx.detail_sites[0])
    legacy, new = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
    # the reach where legacy carved deepest below the DEM at its stream cells
    deepest = {}
    for cell in legacy["cells"]:
        if cell is not None:
            deficit = cell["found"][0][0] - cell["final"][0][0]
            deepest[cell["comid"]] = max(deepest.get(cell["comid"], -np.inf), deficit)
    reach = max((r for r in deepest if r in new["reaches"] and new["reaches"][r]["smoothed"] is not None),
                key=lambda r: deepest[r])
    p = _reach_profiles(ctx, site, reach)
    order = np.argsort(p["stations"])
    st = p["stations"][order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    ax = axes[0]
    ax.plot(st, p["thalweg"][order], color=GROUND, lw=1.0, label="the stream cells' ground")
    ax.plot(st, p["bank"][order], color=NEW, lw=1.6, label="new smoothed bank elevation")
    ax.plot(st, p["carved_bed"][order], color=NEW, lw=1.0, ls="--", label="new carved bed")
    old = p["legacy"]
    ax.plot(old[:, 0], old[:, 2], color=LEGACY, lw=1.4, label="legacy smoothed bank elevation")
    ax.plot(old[:, 0], old[:, 3], color=LEGACY, lw=1.0, ls=":", label="legacy smoothed bed")
    ax.plot(old[:, 0], old[:, 4], color=LEGACY, lw=1.0, ls="--", label="legacy carved bed")
    ax.set_xlabel("metres along the reach from its upstream end")
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"reach {reach}", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7)
    grid = ctx.grid(site)
    dem = np.where(grid.dem >= WALL, np.nan, grid.dem)
    old_bathy = ctx.bathymetry("as_configured", "legacy", site)
    change = old_bathy - dem
    r, c = np.unravel_index(np.nanargmin(change), change.shape)
    half_r, half_c = int(700 / grid.dy), int(900 / grid.dx)
    window = Window(max(r - half_r, 0), min(r + half_r, grid.dem.shape[0] - 1), max(c - half_c, 0),
                    min(c + half_c, grid.dem.shape[1] - 1), grid.dx, grid.dy)
    stats = dict(site=site, reach=int(reach), legacy_deepest=rounded(deepest[reach], 2))
    for ax, code in zip(axes[1:], ("legacy", "new")):
        bathy = ctx.bathymetry("as_configured", code, site)
        hillshade(ax, window, grid.dem, alpha=0.5)
        image = show_raster(ax, window, bathy - dem, cmap="RdBu", vmin=-12, vmax=12)
        ax.set_title(f"{code}: bathymetry − DEM", fontsize=8.5)
        stats[f"{code}_lowest"] = rounded(np.nanmin(bathy - dem), 2)
        stats[f"{code}_below_5m"] = int(np.nansum(bathy - dem < -5))
    fig.colorbar(image, ax=axes[1:], shrink=0.85, label="metres (blue raised, red lowered)")
    return fig, dict(caption=(
        f"The one reach of {ctx.site_label(site)}, which has no neighbouring reach with cross sections. Legacy put "
        "such a reach's cross sections in the order its cells came in the raster, row by row, with stations 0, 1, 2 "
        "..., so its running median mixed cross sections from across the reach and its 1% cap let the bed fall only "
        "1 cm per cross section. The banks fall 31 m along the reach; legacy's bed fell about 2.5 m. So near the top "
        "legacy carved about 11 m below the DEM, and downstream its bed was above the banks and it carved nothing. "
        "The new code smooths the bed along the stream, in metres. Right: bathymetry minus the DEM where legacy "
        "carved deepest, to scale."), stats=stats)


# --- The raster ---------------------------------------------------------------------------------------------------------


@figure("G1", "The bathymetry raster", "The bathymetry raster")
def bathymetry_map(ctx):
    site = ctx.detail_sites[0]
    grid = ctx.grid(site)
    dem = np.where(grid.dem >= WALL, np.nan, grid.dem)
    old, new = ctx.bathymetry("as_configured", "legacy", site), ctx.bathymetry("as_configured", "new", site)
    either = np.isfinite(old) | np.isfinite(new)
    rows, cols = np.nonzero(either)
    # a window around the most bathymetry
    size_r, size_c = int(900 / grid.dy), int(900 / grid.dx)
    best, score = (rows[0], cols[0]), -1
    for k in range(0, rows.size, 25):
        inside = (np.abs(rows - rows[k]) <= size_r) & (np.abs(cols - cols[k]) <= size_c)
        if inside.sum() > score:
            best, score = (rows[k], cols[k]), inside.sum()
    window = Window(best[0] - size_r, best[0] + size_r, best[1] - size_c, best[1] + size_c, grid.dx, grid.dy)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    limit = 4.0
    for ax, values, title in ((axes[0], old - dem, "legacy: bathymetry − DEM"), (axes[1], new - dem,
                                                                                "new: bathymetry − DEM")):
        hillshade(ax, window, grid.dem, alpha=0.5)
        image = show_raster(ax, window, values, cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_title(title, fontsize=9)
    fig.colorbar(image, ax=axes[:2], shrink=0.8, label="metres (blue raised, red lowered)")
    ax = axes[2]
    hillshade(ax, window, grid.dem, alpha=0.5)
    only_old = np.where(np.isfinite(old) & ~np.isfinite(new), 1.0, np.nan)
    only_new = np.where(np.isfinite(new) & ~np.isfinite(old), 1.0, np.nan)
    both = np.where(np.isfinite(new) & np.isfinite(old), new - old, np.nan)
    image = show_raster(ax, window, both, cmap="PuOr", vmin=-limit, vmax=limit)
    from matplotlib.colors import ListedColormap
    show_raster(ax, window, only_old, cmap=ListedColormap([LEGACY]), alpha=0.9)
    show_raster(ax, window, only_new, cmap=ListedColormap([NEW]), alpha=0.9)
    ax.set_title("new − legacy where both carve; cells only legacy carves\n(vermillion) or only the new code (blue)",
                 fontsize=8.5)
    fig.colorbar(image, ax=axes[2], shrink=0.8, label="metres")
    for a in axes:
        a.set_xlim(window.extent[0], window.extent[1])
        a.set_ylim(window.extent[2], window.extent[3])
    return fig, dict(caption=(
        f"The bathymetry rasters of part of {ctx.site_label(site)}, to scale, as the bed elevation minus the DEM. "
        "Both carve with bank elevations, so both raise ground as well as lower it. The new raster carries a "
        "sub-cell channel as just its stream cell (at the bed), where legacy's single cell carved only its stream "
        "cell too; wider new channels are carved over the ordinates between their banks, and the gaps between cross "
        "sections are filled as legacy filled them."),
                 stats=dict(site=site, legacy_cells=int(np.isfinite(old).sum()), new_cells=int(np.isfinite(new).sum()),
                            only_legacy=int(np.isfinite(only_old).sum()), only_new=int(np.isfinite(only_new).sum())))


@figure("G2", "How many cells get bathymetry", "The bathymetry raster")
def bathymetry_counts(ctx):
    old, new, names = [], [], []
    for site in ctx.sites:
        a, b = ctx.bathymetry("as_configured", "legacy", site), ctx.bathymetry("as_configured", "new", site)
        if a is None or b is None:
            continue
        old.append(int(np.isfinite(a).sum()))
        new.append(int(np.isfinite(b).sum()))
        names.append(site)
        ctx._cache.pop(("bathy", "as_configured", "legacy", site), None)
        ctx._cache.pop(("bathy", "as_configured", "new", site), None)
    old, new = np.array(old), np.array(new)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.loglog(old, new, "o", color=NEW, ms=4)
    lims = [min(old.min(), new.min()) * 0.8, max(old.max(), new.max()) * 1.2]
    ax.plot(lims, lims, color=GROUND, lw=0.8)
    ax.set_xlabel("legacy's bathymetry cells")
    ax.set_ylabel("new bathymetry cells")
    ax.set_title(f"each of the {len(names)} sites")
    note(ax, f"totals: legacy {old.sum():,}, new {new.sum():,}", loc="upper left")
    ax = axes[1]
    ratio = new / np.maximum(old, 1)
    ax.hist(ratio, bins=np.linspace(0.5, 2.0, 31), color=NEW, alpha=0.8)
    ax.axvline(1.0, color=GROUND, lw=0.8)
    ax.set_xlabel("new cells / legacy's")
    ax.set_ylabel("sites")
    return fig, dict(caption=(
        "The number of cells with bathymetry in each site's raster, after the gap fill. With single cells one "
        "spacing wide or the prior's width, and reach medians rebuilt as they are, the new code carves about as many "
        f"cells as legacy at most sites (median {np.median(ratio):.2f} times), from {ratio.min():.2f} to "
        f"{ratio.max():.1f} times as many, and {new.sum() / old.sum() - 1:+.0%} in all (the earlier two-spacing single "
        "cells carved 80,508 cells)."),
                 stats=dict(sites=len(names), legacy_total=int(old.sum()), new_total=int(new.sum()),
                            median_ratio=rounded(np.median(ratio), 3)))


@figure("G3", "Sanity check: filling the gaps, as legacy did", "The bathymetry raster")
def gap_fill(ctx):
    from arc.Automated_Rating_Curve_Generator import _fill_bathymetry_nan_cells
    from arc.bathymetry import fill_bathymetry_gaps
    rng = np.random.default_rng(8)
    raster = (100.0 + rng.normal(0, 1, (40, 60))).astype(np.float32)
    raster[rng.random(raster.shape) < 0.45] = np.nan
    old = _fill_bathymetry_nan_cells(raster.copy())
    new = fill_bathymetry_gaps(raster.copy())
    identical = bool(np.array_equal(old, new, equal_nan=True))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    for ax, values, title in ((axes[0, 0], raster, "a random raster, 45% gaps"),
                              (axes[0, 1], old, "legacy's fill"), (axes[0, 2], new, "the new fill")):
        ax.imshow(values, cmap="viridis", vmin=97, vmax=103, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    note(axes[0, 2], "identical, bit for bit" if identical else "DIFFERENT", loc="lower right")
    # the pool: a cell whose ground is below the channel bed, so the carve skipped it
    ground = np.full((9, 9), 100.0)
    ground[4, 4] = 94.0  # a pool
    bathymetry = np.full((9, 9), np.nan, dtype=np.float32)
    bathymetry[3:6, 2:7] = 95.0
    bathymetry[4, 4] = np.nan  # the carve left the pool alone, as its ground is below the bed
    legacy_dropped = np.where(bathymetry > ground, np.nan, bathymetry).astype(np.float32)
    old = _fill_bathymetry_nan_cells(legacy_dropped.copy())
    new = fill_bathymetry_gaps(bathymetry.copy(), ground.astype(np.float32))
    for ax, values, title in ((axes[1, 0], bathymetry, "carved to 95 m around a pool at 94 m"),
                              (axes[1, 1], old, f"legacy fills the pool to {old[4, 4]:.0f} m"),
                              (axes[1, 2], new, f"the new fill leaves it ({'gap' if np.isnan(new[4, 4]) else f'{new[4, 4]:.0f} m'})")):
        ax.imshow(values, cmap="viridis", vmin=93, vmax=100, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, dict(caption=(
        "Top: legacy's gap fill and the new one on a random raster: a cell without bathymetry and at least four of "
        "its eight neighbours with it takes their mean, in one synchronous pass. They agree bit for bit (also on "
        "4000 random rasters in the tests). Bottom: without bank elevations, a pool below the channel's bed isn't "
        "carved; legacy's fill then put the bed back over it, above the ground. The new fill doesn't fill a gap "
        "above the ground."), stats=dict(identical=identical, legacy_pool=rounded(old[4, 4], 2),
                                         new_pool=None if np.isnan(new[4, 4]) else rounded(new[4, 4], 2)))
