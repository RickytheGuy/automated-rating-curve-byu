"""The cross section pivoting as the rating curve rises (vc_pivot): where and how far it turns, what that does to the
rating curves, and a real cell."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from vc_plot import ACCENT, GROUND, NEW, WATER, Window, figure, hillshade, note, rounded

SECTION = "Pivoting the cross section as the water rises"
Q, V, T, WSE = 0, 1, 2, 3


def _cells(ctx, site):
    capture = ctx.capture("pivot", "new", site)
    return [] if capture is None else capture["cells"]


def _rows(curve):
    """A curve's increments with their own values: (discharge, top width, water surface) arrays."""
    if curve is None or not curve["last"] > max(curve["start"], 0):
        return None
    inc = np.asarray(curve["increments"])[:curve["last"]]
    keep = inc[:, Q] > 0.0
    return inc[keep, Q], inc[keep, T], inc[keep, WSE]


def _wse_at(curve, q):
    rows = _rows(curve)
    if rows is None or not rows[0][0] <= q <= rows[0][-1]:
        return math.nan
    return float(np.interp(q, rows[0], rows[2]))


def _rise(curve, wse, height):
    """How much a curve's discharge rises from a water surface to height above it, as a share."""
    rows = _rows(curve)
    if rows is None:
        return math.nan
    q, _, w = rows
    low, high = np.interp(wse, w, q), np.interp(wse + height, w, q)
    return float(high / low - 1.0) if low > 0 else math.nan


def _held(curve):
    """How many increments repeat the one before (the rating curve's rule where discharge falls), of how many."""
    if curve is None or not curve["last"] > max(curve["start"], 0):
        return 0, 0
    inc = np.asarray(curve["increments"])
    same = (inc[1:, Q] == inc[:-1, Q]) & (inc[1:, WSE] == inc[:-1, WSE]) & (inc[1:, Q] > 0)
    return int(same.sum()), int(inc.shape[0] - 1)


def _angle(cell, j):
    return math.degrees(float(cell["offsets"][j]))


def _all(ctx):
    for site in ctx.sites:
        for cell in _cells(ctx, site):
            yield site, cell


@figure("PV1", "Where the cross sections pivot, and how far, on the 51 sites", SECTION)
def pivot_sites(ctx):
    count = None
    pivoted, angles, bank_below = [], [], []
    top_angles, d_top, d_half, d_top_once, d_half_once = [], [], [], [], []
    reproduced = total = only_fixed = only_pivot = both = 0
    held = np.zeros((2, 2), dtype=int)
    seconds = np.zeros(2)
    for site in ctx.sites:
        capture = ctx.capture("pivot", "new", site)
        if capture is None:
            continue
        seconds += [capture["seconds"]["fixed"], capture["seconds"]["pivot"]]
        for cell in capture["cells"]:
            total += 1
            reproduced += cell["reproduces_fixed"]
            fixed, pivot = cell["fixed"], cell["pivot"]
            has_fixed = fixed is not None and fixed["last"] > max(fixed["start"], 0)
            has_pivot = pivot is not None and pivot["last"] > max(pivot["start"], 0)
            only_fixed += has_fixed and not has_pivot
            only_pivot += has_pivot and not has_fixed
            if not (has_fixed and has_pivot):
                continue
            both += 1
            chosen = np.asarray(cell["chosen"])
            count = chosen.size
            offsets = np.degrees(np.asarray(cell["offsets"]))
            pivoted.append(chosen != 0)
            angles.append(np.abs(offsets[chosen]))
            wse = np.asarray(pivot["increments"])[:, WSE]
            bank_below.append(np.asarray(fixed["increments"])[:, WSE] <= cell["bank_elevation"])
            top_angles.append(abs(offsets[cell["top"]]))
            d_top.append(pivot["max_wse"] - fixed["max_wse"])
            half = 0.5 * cell["q_max"]
            d_half.append(_wse_at(pivot, half) - _wse_at(fixed, half))
            once = cell["top_curve"]
            d_top_once.append(once["max_wse"] - fixed["max_wse"])
            d_half_once.append(_wse_at(once, half) - _wse_at(fixed, half))
            for row, curve in enumerate((fixed, pivot)):
                h, n = _held(curve)
                held[row] += (h, n)
    candidates = max(len(cell["offsets"]) for _, cell in _all(ctx))
    pivoted, angles, bank_below = np.array(pivoted), np.array(angles), np.array(bank_below)
    top_angles, d_top, d_half = np.array(top_angles), np.array(d_top), np.array(d_half)
    d_top_once, d_half_once = np.array(d_top_once), np.array(d_half_once)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    steps = np.arange(1, count + 1)
    ax.plot(steps, pivoted.mean(0), color=NEW, lw=1.8, label="cross sections pivoted from the fixed direction")
    ax.plot(steps, bank_below.mean(0), color=GROUND, lw=1.0, ls="--",
            label="increments at or below the bank elevation (no pivot there)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("increment of the rating curve (1 to the maximum flow's water surface)")
    ax.set_ylabel("share of the cells with both curves")
    ax2 = ax.twinx()
    median_angle = np.array([np.median(angles[pivoted[:, i], i]) if pivoted[:, i].any() else np.nan
                             for i in range(count)])
    ax2.plot(steps, median_angle, color=ACCENT, lw=1.2, label="median turn of those pivoted (right)")
    ax2.set_ylabel("degrees from the fixed direction")
    ax2.set_ylim(0, max(10.0, np.nanmax(median_angle) * 1.2))
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left", fontsize=6.5)
    ax = axes[1]
    bins = np.arange(-1.25, 91.25, 2.5)
    ax.hist(top_angles, bins=bins, color=NEW, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("the top's turn from the fixed direction (degrees)")
    ax.set_ylabel("cells")
    note(ax, f"pivoted at the top: {np.mean(top_angles > 0):.0%}\nmedian turn of those: "
             f"{np.median(top_angles[top_angles > 0]):.1f}°", loc="upper right")
    ax = axes[2]
    for values, color, style, label in ((d_top, NEW, "-", "the top (the same either way)"),
                                        (d_half, ACCENT, "-", "at half the maximum flow, pivoting at each increment"),
                                        (d_half_once, ACCENT, ":", "at half the maximum flow, the top's direction")):
        v = np.sort(values[np.isfinite(values)])
        ax.plot(v, np.linspace(0, 1, v.size), color=color, ls=style, lw=1.5, label=label)
    ax.axvline(0, color=GROUND, lw=0.8)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("water surface, pivoted − fixed (m)")
    ax.set_ylabel("share of cells")
    ax.legend(loc="upper left", fontsize=6.5)
    finite = np.isfinite(d_half)
    stats = dict(cells=total, reproduces_fixed=reproduced, both=both, only_fixed=only_fixed, only_pivot=only_pivot,
                 any_pivot=rounded(np.mean(pivoted.any(1)), 3), increments_pivoted=rounded(pivoted.mean(), 3),
                 increments_below_bank=rounded(bank_below.mean(), 3),
                 top_pivoted=rounded(np.mean(top_angles > 0), 3),
                 top_turn_median=rounded(float(np.median(top_angles[top_angles > 0])), 1),
                 top_turn_p90=rounded(float(np.percentile(top_angles[top_angles > 0], 90)), 1),
                 d_top=dict(median=rounded(np.median(d_top), 3), p10=rounded(np.percentile(d_top, 10), 3),
                            p90=rounded(np.percentile(d_top, 90), 3), abs_median=rounded(np.median(np.abs(d_top)), 3),
                            higher=rounded(np.mean(d_top > 0.005), 3), lower=rounded(np.mean(d_top < -0.005), 3)),
                 d_half=dict(median=rounded(np.median(d_half[finite]), 3),
                             p10=rounded(np.percentile(d_half[finite], 10), 3),
                             p90=rounded(np.percentile(d_half[finite], 90), 3),
                             abs_median=rounded(np.median(np.abs(d_half[finite])), 3)),
                 d_top_once=dict(median=rounded(np.median(d_top_once), 3),
                                 abs_median=rounded(np.median(np.abs(d_top_once)), 3)),
                 held=dict(fixed=rounded(held[0, 0] / max(held[0, 1], 1), 4),
                           pivot=rounded(held[1, 0] / max(held[1, 1], 1), 4)),
                 seconds=dict(fixed=rounded(seconds[0], 1), pivot=rounded(seconds[1], 1)))
    return fig, dict(caption=(
        "The cross section pivoting as its rating curve rises (testing/visual_comparison/vc_pivot.py, an experiment): "
        f"at every increment's water surface it takes the direction, of the angle search's {candidates}, whose water is "
        "narrowest there, and the increment's discharge, velocity and top width are that cross section's. Each "
        "candidate is sampled as the pipeline samples its own and carved with the same channel, so up to the bank "
        "elevation they all hold the same channel and the fixed direction is kept. Left: as the water rises, the "
        f"share of the {both:,} cells with both curves whose cross section has pivoted, and how far. Middle: the "
        "direction at the top, the maximum flow's water surface. Right: how far the water surface moves, at the top "
        "and at half the maximum flow, pivoting at every increment and with the direction narrowest at the top taken "
        f"for the whole curve (the top is the same either way). Limited to the fixed direction, the pivoting curve is "
        f"the pipeline's at "
        f"{reproduced:,} of {total:,} cells, bit for bit."), stats=stats)


def _example(ctx):
    """The cell whose top water surface pivoting moves most, of those at the detail and flood-map sites."""
    best, score = None, -1.0
    for site, cell in _all(ctx):
        if not site.startswith(("Flint", "Cuyahoga", "East_Fork", "South_Platte")):
            continue
        fixed, pivot = cell["fixed"], cell["pivot"]
        if _rows(fixed) is None or _rows(pivot) is None:
            continue
        change = abs(pivot["max_wse"] - fixed["max_wse"])
        turn = abs(_angle(cell, cell["top"]))
        if turn >= 10.0 and change > score and len(set(np.asarray(cell["chosen"]).tolist())) >= 3:
            best, score = (site, cell), change
    return best


def _extent(grid, row, col, direction, wse, length=5000.0):
    """How far the water reaches along a cross section, each side, on the DEM."""
    from arc.hydraulics import top_widths
    from arc.xsection.sampling import sample_elevations
    values, spacing = sample_elevations(grid.dem, row, col, direction, length, grid.dx, grid.dy)
    return top_widths(values, spacing, wse)


@figure("PV2", "A cross section pivoting on a real cell", SECTION)
def pivot_cell(ctx):
    found = _example(ctx)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(width_ratios=[1.25, 0.9, 1.0]))
    if found is None:
        return fig, dict(caption="No pivoting runs.", stats={})
    site, cell = found
    grid = ctx.grid(site)
    fixed, pivot, once = cell["fixed"], cell["pivot"], cell["top_curve"]
    chosen = np.asarray(cell["chosen"])
    wse = np.asarray(pivot["increments"])[:, WSE]
    thalweg = cell["thalweg"]
    # the stages where the choice changes, and the top
    changes = [0] + [i for i in range(1, chosen.size) if chosen[i] != chosen[i - 1]] + [chosen.size - 1]
    stages = sorted(set(changes))[-4:]
    ax = axes[0]
    reach = 2500.0
    window = Window.around([cell["row"]], [cell["col"]], grid.dx, grid.dy, pad_metres=reach * 0.62, shape=grid.dem.shape)
    hillshade(ax, window, grid.dem)
    x0, y0 = window.xy(cell["row"], cell["col"])
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(stages)))
    drawn = []
    for color, i in zip(colors, stages):
        direction = cell["direction"] + cell["offsets"][chosen[i]]
        xs_dir = direction - math.pi / 2
        left, right = _extent(grid, cell["row"], cell["col"], direction, wse[i])
        drawn += [left, right]
        ax.plot([x0 - math.cos(xs_dir) * left, x0 + math.cos(xs_dir) * right],
                [y0 + math.sin(xs_dir) * left, y0 - math.sin(xs_dir) * right], color=color, lw=2.2,
                label=f"increment {i + 1}, {wse[i] - thalweg:.1f} m up: {_angle(cell, chosen[i]):+.1f}°")
    xs_dir = cell["direction"] - math.pi / 2
    left, right = _extent(grid, cell["row"], cell["col"], cell["direction"], wse[-1])
    fixed_reach = left + right
    drawn += [left, right]
    view = max(1.4 * max(drawn), 300.0)
    ax.plot([x0 - math.cos(xs_dir) * left, x0 + math.cos(xs_dir) * right],
            [y0 + math.sin(xs_dir) * left, y0 - math.sin(xs_dir) * right], color="white", lw=1.2, ls="--",
            label=f"the fixed direction at the same water surface as increment {chosen.size}")
    ax.plot([x0], [y0], "o", color=ACCENT, ms=5)
    ax.set_xlim(x0 - view, x0 + view)
    ax.set_ylim(y0 - view, y0 + view)
    ax.set_title(f"{ctx.site_label(site).split(' (')[0]}, row {cell['row']}, column {cell['col']}\n"
                 "the water's reach on the DEM at the pivoted directions", fontsize=8.5)
    ax.legend(loc="lower left", fontsize=6.2)
    ax.set_xlabel("metres east")
    ax.set_ylabel("metres north")
    ax = axes[1]
    turn = np.degrees(np.asarray(cell["offsets"])[chosen])
    ax.plot(turn, wse - thalweg, "o-", color=NEW, ms=3.5, lw=0.8)
    ax.axhline(cell["bank_elevation"] - thalweg, color=GROUND, lw=0.9, ls="--", label="the bank elevation")
    ax.axvline(0, color=GROUND, lw=0.6)
    ax.set_xlabel("the cross section's turn from the fixed direction (degrees)")
    ax.set_ylabel("water surface above the stream cell (m)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax = axes[2]
    for curve, color, style, label in ((fixed, NEW, "-", "fixed (the pipeline's)"),
                                       (pivot, ACCENT, "-", "pivoting at every increment"),
                                       (once, GROUND, ":", "chosen once, at the top")):
        rows = _rows(curve)
        if rows is not None:
            ax.plot(rows[0], rows[2] - thalweg, color=color, ls=style, lw=1.6, marker="." if style == "-" else None,
                    ms=3, label=label)
    ax.axvline(cell["q_max"], color=GROUND, lw=0.8, ls=":")
    ax.text(cell["q_max"], 0.02, " maximum flow", transform=ax.get_xaxis_transform(), fontsize=7, color=GROUND)
    ax.set_xlabel("discharge (m³/s)")
    ax.set_ylabel("water surface above the stream cell (m)")
    ax.legend(loc="lower right", fontsize=6.5)
    first = int(np.argmax(chosen != 0)) if (chosen != 0).any() else chosen.size - 1
    turns = np.abs(np.degrees(np.asarray(cell["offsets"])[chosen]))
    stats = dict(site=site, cell=(cell["row"], cell["col"]), top_turn=rounded(_angle(cell, cell["top"]), 1),
                 first_pivot_m=rounded(wse[first] - thalweg, 2), bank_m=rounded(cell["bank_elevation"] - thalweg, 2),
                 largest_turn=rounded(float(turns.max()), 1), fixed_reach_m=rounded(fixed_reach, 0),
                 pivot_reach_m=rounded(sum(_extent(grid, cell["row"], cell["col"],
                                                   cell["direction"] + cell["offsets"][chosen[-1]], wse[-1])), 0), fixed_rise=rounded(_rise(fixed, wse[first], 1.0), 3),
                 top_fixed_m=rounded(fixed["max_wse"] - thalweg, 2), top_pivot_m=rounded(pivot["max_wse"] - thalweg, 2),
                 top_once_m=rounded(once["max_wse"] - thalweg, 2), directions=len(set(chosen.tolist())))
    return fig, dict(caption=(
        f"One cell of {ctx.site_label(site)}, the one of the flood-map sites whose top pivoting moves most. Left: the "
        "cross section at the increments where its direction changes, and at the top, each drawn as far as its water "
        "reaches on the DEM, over the fixed direction at the pivoted top's water surface (dashed). Middle: the "
        "direction at each "
        "increment; up to the bank elevation every candidate holds the same channel and the fixed direction is "
        f"kept. Right: the rating curves. Here the cross section keeps the fixed direction up to "
        f"{stats['first_pivot_m']:.1f} m above the stream cell (the bank elevation is {stats['bank_m']:.1f} m up), then "
        f"turns by up to {stats['largest_turn']:.1f}°, to where the water is still narrow (at the top, "
        f"{stats['pivot_reach_m']:,.0f} m across on the DEM, where the fixed direction's is "
        f"{stats['fixed_reach_m']:,.0f} m)"
        + (f", while the fixed cross section's discharge rises only {stats['fixed_rise']:.0%} over the next metre"
           if stats["fixed_rise"] < 0.1 else "")
        + ". The pivoted cross section carries the maximum flow at "
        f"{stats['top_pivot_m']:.2f} m, the fixed one at {stats['top_fixed_m']:.2f} m. With each turn the discharge at "
        "a level jumps to the new cross section's."), stats=stats)


@figure("PV3", "Rating curves, fixed and pivoting, at six cells", SECTION)
def pivot_curves(ctx):
    pairs = []
    for site, cell in _all(ctx):
        if _rows(cell["fixed"]) is not None and _rows(cell["pivot"]) is not None:
            pairs.append((cell["pivot"]["max_wse"] - cell["fixed"]["max_wse"], site, cell))
    pairs.sort(key=lambda p: p[0])
    picks = [pairs[int(q * (len(pairs) - 1))] for q in (0.02, 0.1, 0.3, 0.7, 0.9, 0.98)] if pairs else []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    stats = {}
    for ax, (change, site, cell) in zip(axes.ravel(), picks):
        thalweg = cell["thalweg"]
        for curve, color, label in ((cell["fixed"], NEW, "fixed"), (cell["pivot"], ACCENT, "pivoting")):
            q, _, w = _rows(curve)
            ax.plot(q, w - thalweg, color=color, lw=1.5, marker=".", ms=3, label=label)
        chosen = np.asarray(cell["chosen"])
        inc = np.asarray(cell["pivot"]["increments"])
        turned = (chosen != 0) & (inc[:, Q] > 0)
        ax.plot(inc[turned, Q], inc[turned, WSE] - thalweg, "o", ms=4, mfc="none", color=ACCENT, alpha=0.6,
                label="pivoted increments")
        ax.axhline(cell["bank_elevation"] - thalweg, color=GROUND, lw=0.8, ls="--", label="bank elevation")
        ax.set_title(f"{site.split('_(')[0].replace('_', ' ')[:30]}, row {cell['row']}, column {cell['col']}\n"
                     f"top {change:+.2f} m, turned {_angle(cell, cell['top']):+.1f}° there", fontsize=8)
        ax.set_xlabel("discharge (m³/s)")
        ax.set_ylabel("water surface above the stream cell (m)")
        stats[f"{site}:{cell['row']},{cell['col']}"] = rounded(change, 3)
    if picks:
        axes[0, 0].legend(loc="lower right", fontsize=6.5)
    return fig, dict(caption=(
        "Rating curves at six cells of the 51 sites, fixed and pivoting, chosen at the 2nd, 10th, 30th, 70th, 90th "
        "and 98th percentiles of how far pivoting moves the top. Circles are increments whose cross section has "
        "pivoted. Up to the bank elevation the two are the same; above it, a cross section turned to where the water is "
        "narrowest carries less at a level where it is narrower but no deeper, and more where it is narrower and "
        "deeper or smoother, or where the fixed one's water spreads onto flat ground, so the top moves both ways, and "
        "where the direction switches the discharge jumps."), stats=stats)
