"""Hydraulics (arc.hydraulics) and rating curves (arc.rating_curve)."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from vc_plot import (ACCENT, BANK, GROUND, LEGACY, NEW, PINK, WATER, between_banks, figure, increments, mark_banks,
                     new_stations, note, rounded, water)

WALL = 9999.0
SENTINEL = 99999.9  # legacy's sampler put this after each side's last ordinate


def _sides(elevations: np.ndarray, n: np.ndarray):
    """A new cross section (left to right) as legacy's two sides, each out from the stream cell and ending at its
    first wall, with legacy's sentinel after it: (side1, count1, n1, side2, count2, n2), side 1 on the right."""
    center = elevations.size // 2
    out = []
    for step in (1, -1):
        side = elevations[center::step]
        ns = n[center::step]
        walls = np.flatnonzero(side >= WALL)
        count = int(walls[0]) if walls.size else side.size
        out += [np.append(side[:count], SENTINEL).astype(np.float64), count,
                np.append(ns[:count], ns[count - 1]).astype(np.float64)]
    return tuple(out)


def _legacy_q(sides, wse, slope, spacing, roughness=(6.0, 1.0, 1.0), banks=(-1, -1)):
    from arc.cross_section import calculate_discharge_from_wse
    s1, c1, n1, s2, c2, n2 = sides
    return calculate_discharge_from_wse(wse, math.sqrt(slope), s1, c1, n1, s2, c2, n2, spacing, *roughness, 1.0,
                                        *banks)


def _ground_top(elevations: np.ndarray) -> float:
    """The lower of the highest ground either side before a wall: water below it stays inside the section."""
    center = elevations.size // 2
    tops = []
    for side in (elevations[center:], elevations[center::-1]):
        walls = np.flatnonzero(side >= WALL)
        tops.append(side[:walls[0] if walls.size else side.size].max())
    return min(tops)


def _sample_sections(ctx, site, count=150, seed=0):
    """Sampled (uncarved) cross sections of a site from the new run: elevations, n, spacing, slope, the cell."""
    capture = ctx.capture("as_configured", "new", site)
    cells = [c for c in capture["cells"] if c is not None]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cells), size=min(count, len(cells)), replace=False)
    return [cells[k] for k in pick]


@figure("H1", "Sanity check: with constant n, the discharge is legacy's", "Hydraulics")
def discharge_sanity(ctx):
    from arc.hydraulics import discharge
    from arc.xsection.xsection import XSection
    site = ctx.detail_sites[0]
    ratios, examples = [], []
    for cell in _sample_sections(ctx, site):
        elevations, spacing = cell["found"], cell["spacing"]
        n = np.where(elevations < WALL, 0.035, WALL)  # uniform n: the codes take different ends' n (H2)
        xs = XSection(elevations.copy(), n.copy(), spacing)
        sides = _sides(elevations, n)
        thalweg = elevations[elevations.size // 2]
        top = min(_ground_top(elevations) - 0.01, thalweg + 10.0)
        stages = np.round(np.linspace(thalweg + 0.05, top, 25), 3)
        new = np.array([discharge(xs, cell["slope"], wse=w) for w in stages])
        old = np.array([_legacy_q(sides, w, cell["slope"], spacing) for w in stages])
        ok = old > 0
        ratios.append(np.abs(new[ok] / old[ok] - 1.0))
        if len(examples) < 3:
            examples.append((stages - thalweg, old, new))
    ratios = np.concatenate(ratios)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for k, (depth, old, new) in enumerate(examples):
        ax.plot(old, depth, color=LEGACY, lw=3, alpha=0.4, label="legacy" if k == 0 else None)
        ax.plot(new, depth, color=NEW, lw=1.0, label="new" if k == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("discharge (m³/s)")
    ax.set_ylabel("depth above the stream cell (m)")
    ax.set_title("three of the cross sections")
    ax.legend()
    ax = axes[1]
    positive = ratios[ratios > 0]
    bins = np.logspace(-17, -11, 61)
    ax.hist(np.clip(positive, bins[0], bins[-1]), bins=bins, color=NEW, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("|new / legacy − 1|")
    ax.set_ylabel("water levels")
    note(ax, f"{ratios.size} water levels on 150 cross sections\nidentical: {np.mean(ratios == 0):.0%}\n"
             f"largest: {ratios.max():.1e}", loc="upper right")
    ax.set_title("relative difference")
    return fig, dict(caption=(
        f"The discharge at 25 water levels on each of 150 sampled cross sections of {ctx.site_label(site)}, with "
        "a uniform Manning's n of 0.035, not varying with depth, and no banks, from legacy's "
        "calculate_discharge_from_wse and from arc.hydraulics, which works out the same geometry directly. They "
        "agree to rounding. (With n varying across the section the codes differ by design: see H2.)"),
                 stats=dict(levels=int(ratios.size), largest=float(ratios.max()),
                            identical=rounded(np.mean(ratios == 0), 3)))


@figure("H2", "Which end of a segment its Manning's n comes from", "Hydraulics")
def depth_varying_n(ctx):
    from arc.hydraulics import DepthRoughness, discharge
    from arc.xsection.xsection import XSection
    roughness = DepthRoughness()
    site = ctx.detail_sites[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    x = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    z = np.array([3.0, 2.2, 1.0, 0.0, 1.2, 2.0, 3.0])
    wse = 2.5
    water(ax, x, z, wse)
    ax.plot(x, z, "o-", color=GROUND, ms=4)
    ax.axhline(wse, color=WATER, lw=1.0)
    for x0, z0, x1, z1 in ((10.0, 1.2, 20.0, 2.0),):
        ax.annotate("", xy=(x0, wse), xytext=(x0, z0), arrowprops=dict(arrowstyle="<->", color=NEW, lw=1.3))
        ax.annotate("", xy=(x1, wse), xytext=(x1, z1), arrowprops=dict(arrowstyle="<->", color=LEGACY, lw=1.3))
        ax.text(x0 - 1, (wse + z0) / 2, f"new: the inner\nend's depth, {wse - z0:.1f} m", ha="right", fontsize=7,
                color=NEW)
        ax.text(x1 + 1, (wse + z1) / 2 - 0.35, f"legacy: the outer\nend's, {wse - z1:.1f} m", ha="left",
                fontsize=7, color=LEGACY)
    ax.set_xlabel("metres from the stream cell")
    ax.set_ylabel("elevation (m)")
    ax.set_title("one segment's depth")
    ax.set_xlim(-32, 36)
    ax = axes[1]
    depth = np.linspace(0, 3, 301)
    ax.plot(depth, roughness.scale(1.0, depth), color=GROUND, lw=1.5)
    for h, color in ((1.3, NEW), (0.5, LEGACY)):
        ax.plot([h], [roughness.scale(1.0, h)], "o", color=color, ms=6)
    ax.set_xlabel("water depth over the segment (m)")
    ax.set_ylabel("n / n₀")
    ax.set_title("n(h) = n₀ (1 + 1 / (1 + 6h))  (legacy's defaults, both codes)")
    ax = axes[2]
    cases = (("inner end's n₀ (the sites' n, no depth scaling)", False, (6.0, 1.0, 1.0), GROUND),
             ("inner end's depth (uniform n₀)", True, tuple(roughness), ACCENT),
             ("both, as the sites run", False, tuple(roughness), NEW))
    stats = {}
    bins = np.linspace(0.5, 2.0, 121)
    for label, uniform, parameters, color in cases:
        ratios = []
        for cell in _sample_sections(ctx, site):
            elevations, n, spacing = cell["found"], cell["n"], cell["spacing"]
            if uniform:
                n = np.where(elevations < WALL, 0.035, WALL)
            xs = XSection(elevations.copy(), n.copy(), spacing)
            sides = _sides(elevations, n)
            thalweg = elevations[elevations.size // 2]
            top = min(_ground_top(elevations) - 0.01, thalweg + 10.0)
            varying = DepthRoughness(*parameters) if parameters[1] != 1.0 else None
            for w in np.round(np.linspace(thalweg + 0.05, top, 25), 3):
                old = _legacy_q(sides, w, cell["slope"], spacing, roughness=parameters)
                if old > 0:
                    ratios.append(discharge(xs, cell["slope"], wse=w, roughness=varying) / old)
        ratios = np.array(ratios)
        q = np.percentile(ratios, [10, 50, 90])
        ax.hist(np.clip(ratios, bins[0], bins[-1]), bins=bins, histtype="step", color=color, lw=1.3,
                label=f"{label}: median {q[1]:.3f}")
        stats[label] = dict(levels=int(ratios.size), p10=rounded(q[0], 4), median=rounded(q[1], 4),
                            p90=rounded(q[2], 4))
    ax.axvline(1.0, color=GROUND, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("new discharge / legacy's")
    ax.set_ylabel("water levels")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title(f"150 cross sections of {ctx.site_label(site)[:24]}")
    return fig, dict(caption=(
        "Which end of a wetted segment of ground its roughness comes from. Legacy took a wholly wet segment's n₀, "
        "and with depth-varying n its depth, at the segment's outer end (and at the inner end of the segment where "
        "the water meets the ground). The new code takes both at the inner end of every segment, so an off-raster "
        "wall's n never counts. Both use legacy's formula, which makes n up to twice n₀ in shallow water. Right: the "
        "new discharge over legacy's on the same sections for each convention on its own, and both together. The "
        "stream cells are water, the smoothest class, and the ground mostly gets rougher away from the channel, so "
        "the inner end's n is mostly the lower: the new discharge is almost never lower, and for one water level in "
        "ten it is 29% or more higher. The depth convention alone adds a few per cent. On flat, uniform ground they "
        "agree to 1e-15 (a test)."), stats=stats)


@figure("H3", "Dividing a cross section at its banks", "Hydraulics")
def compound_conveyance(ctx):
    from arc.cross_section import _compound_section_conveyance
    from arc.hydraulics import compound_geometry, discharge
    from arc.xsection.xsection import XSection
    site = ctx.detail_sites[0]
    capture = ctx.capture("as_configured", "new", site)
    candidates = [c for c in capture["cells"] if c is not None and c["banks"].valid
                  and c["banks"].method == "width_to_depth_ratio" and c["banks"].top_width > 3 * c["spacing"]]
    cell = candidates[len(candidates) // 2] if candidates else next(c for c in capture["cells"] if c is not None)
    elevations, n, spacing = cell["found"], cell["n"], cell["spacing"]
    banks = cell["banks"]
    stations = new_stations(elevations.size, spacing)
    xs = XSection(elevations.copy(), n.copy(), spacing)
    divided = XSection(elevations.copy(), n.copy(), spacing)
    divided.left_bank_distance, divided.right_bank_distance = banks.left, banks.right
    on_ordinates = XSection(elevations.copy(), n.copy(), spacing)
    on_ordinates.left_bank_distance = math.floor(banks.left / spacing) * spacing
    on_ordinates.right_bank_distance = math.floor(banks.right / spacing) * spacing
    thalweg = elevations[elevations.size // 2]
    bank_level = min(banks.left_elevation, banks.right_elevation)
    top = min(_ground_top(elevations) - 0.01, bank_level + 4.0)
    stages = np.linspace(thalweg + 0.02, top, 300)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    wse = bank_level + 1.0
    lo, hi = between_banks(ax, banks.left, banks.right, spacing, curves=[(stations, elevations)], margin=4.0)
    water(ax, stations, elevations, wse)
    ax.plot(stations, np.where(elevations < WALL, elevations, np.nan), "o-", color=GROUND, ms=3)
    mark_banks(ax, banks.left, banks.right)
    parts = compound_geometry(divided, wse=wse)
    for x0, text in ((0.5 * (lo - banks.left), "left overbank"), (0.0, "channel"),
                     (0.5 * (hi + banks.right), "right overbank")):
        ax.text(x0, wse + 0.15, text, ha="center", va="bottom", fontsize=7)
    ax.axhline(wse, color=WATER, lw=1.0)
    ax.set_ylim(thalweg - 0.5, wse + 1.2)
    ax.set_xlabel("metres from the stream cell")
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"banks {banks.left:.1f} m left and {banks.right:.1f} m right")
    ax = axes[1]
    for section, color, label, style in ((xs, GROUND, "undivided", "-"), (divided, NEW, "divided at the banks", "-"),
                                         (on_ordinates, LEGACY, "divided at the ordinates inside them", "--")):
        q = [discharge(section, cell["slope"], wse=w) for w in stages]
        ax.plot(q, stages - thalweg, color=color, lw=1.3, ls=style, label=label)
    ax.axhline(bank_level - thalweg, color=BANK, lw=0.8, ls=":")
    ax.set_xlabel("discharge (m³/s)")
    ax.set_ylabel("depth above the stream cell (m)")
    ax.set_title("the rating curve, constant n")
    ax.legend(loc="lower right")
    ax = axes[2]
    ratios = []
    for other in _sample_sections(ctx, site, count=100, seed=1):
        e, s = other["found"], other["spacing"]
        nn = np.where(e < WALL, 0.035, WALL)  # uniform n, as in H1
        center = e.size // 2
        right, left = e[center:].copy(), e[center::-1].copy()
        n_right, n_left = nn[center:].copy(), nn[center::-1].copy()
        limit = min(np.flatnonzero(np.r_[right >= WALL, True])[0], np.flatnonzero(np.r_[left >= WALL, True])[0])
        right, left, n_right, n_left = right[:limit], left[:limit], n_right[:limit], n_left[:limit]
        t = e[center]
        top_other = min(right.max(), left.max()) - 0.01
        for bl, br in ((1, 1), (2, 3), (4, 2)):
            if max(bl, br) >= limit - 1:
                continue
            section = XSection(np.concatenate([left[:0:-1], right]), np.concatenate([n_left[:0:-1], n_right]), s)
            section.left_bank_distance, section.right_bank_distance = bl * s, br * s
            for w in np.linspace(t + 0.05, min(top_other, t + 8.0), 12):
                _, _, _, k = _compound_section_conveyance(left, n_left, bl, right, n_right, br, w, s, 6.0, 1.0, 1.0)
                if k > 0:
                    ratios.append(abs(discharge(section, 1.0, wse=w) / k - 1.0))
    ratios = np.array(ratios)
    bins = np.logspace(-17, -11, 61)
    ax.hist(np.clip(ratios[ratios > 0], bins[0], bins[-1]), bins=bins, color=NEW, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("|new / legacy − 1|")
    ax.set_ylabel("water levels")
    ax.set_title("sanity: banks on ordinates, vs legacy")
    note(ax, f"{ratios.size} levels\nidentical: {np.mean(ratios == 0):.0%}\nlargest {ratios.max():.1e}",
         loc="upper right")
    return fig, dict(caption=(
        f"A sampled cross section of {ctx.site_label(site)} with width-to-depth banks. Divided at its banks, its "
        "conveyance is the sum of the left overbank's, the channel's and the right overbank's, which jumps less "
        "when the water spills over the banks than the undivided section's. The new banks can fall between "
        "ordinates, where legacy's were ordinates; dividing at the ordinates inside the banks gives a different "
        "curve above the banks. Right: with banks on ordinates the new division is legacy's, to rounding."),
                 stats=dict(cell=(cell["row"], cell["col"]), left=rounded(banks.left, 2), right=rounded(banks.right, 2),
                            sanity_levels=int(ratios.size), sanity_largest=float(ratios.max())))


def _legacy_max_flow_wse(sides, spacing, slope, q_max, thalweg, roughness=(6.0, 1.0, 1.0), banks=(-1, -1)):
    """Legacy's search for the maximum flow's water surface (calculate_hydraulic_data_for_cell): Brent's method to
    a millimetre, then steps of 0.5 m, 0.05 m and 0.01 m, keeping whichever came closer. Its water surface and the
    discharge there."""
    from scipy.optimize import brentq

    from arc.Automated_Rating_Curve_Generator import (calculate_discharge_from_wse, find_wse, objective_with_wse,
                                                      safe_signs_differ)
    s1, c1, n1, s2, c2, n2 = sides
    args = (s1, c1, n1, s2, c2, n2, spacing, *roughness, 1.0, *banks)
    root = slope ** 0.5
    lower, upper = thalweg + 0.01, thalweg + 24.99
    f_lower, f_upper = objective_with_wse(lower, root, q_max, args), objective_with_wse(upper, root, q_max, args)
    wse, q = -999.0, 0.0
    if safe_signs_differ(f_lower, f_upper):
        try:
            wse = np.round(brentq(objective_with_wse, lower, upper, xtol=0.001, args=(root, q_max, args)), 3)
            q = calculate_discharge_from_wse(wse, root, *args)
        except Exception:
            pass
    elif np.round(f_lower, 5) == 0 or np.round(f_upper, 5) == 0:
        wse = np.round(lower, 3) if np.round(f_lower, 5) == 0 else np.round(upper, 3)
        q = calculate_discharge_from_wse(wse, root, *args)
    first, _, _ = find_wse(101, thalweg, 0.5, q_max, args, slope)
    first = max(first - 0.5, thalweg)
    medium, _, _ = find_wse(101, first, 0.05, q_max, args, slope)
    medium = max(medium - 0.05, thalweg)
    fine, q_fine, _ = find_wse(2501, medium, 0.01, q_max, args, slope)
    if abs(q_fine - q_max) < abs(q - q_max):
        wse, q = fine, q_fine
    return wse, q


@figure("H4", "The water surface that carries the maximum flow", "Hydraulics")
def max_flow_wse(ctx):
    from arc.hydraulics import discharge, wse_for_discharge
    from arc.xsection.xsection import XSection
    rows = []
    for site in ctx.detail_sites:
        capture = ctx.capture("as_configured", "new", site)
        for cell in capture["cells"]:
            if cell is None or not cell["qmax"] > 0:
                continue
            elevations, spacing = cell["found"], cell["spacing"]
            n = np.where(elevations < WALL, 0.035, WALL)  # uniform n, so both use the same discharge (H1)
            xs = XSection(elevations.copy(), n.copy(), spacing)
            thalweg = elevations[elevations.size // 2]
            exact = wse_for_discharge(xs, cell["qmax"], cell["slope"])
            old, q_old = _legacy_max_flow_wse(_sides(elevations, n), spacing, cell["slope"], cell["qmax"], thalweg)
            rows.append((site, cell, exact, old, q_old))
    difference = np.array([r[3] - r[2] for r in rows if np.isfinite(r[2]) and r[3] > -999])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    # the cell where legacy's answer is furthest above the exact one, among those it could draw
    finite = [r for r in rows if np.isfinite(r[2]) and r[3] > -999]
    site, cell, exact, old, q_old = max(finite, key=lambda r: r[3] - r[2])
    elevations, spacing = cell["found"], cell["spacing"]
    n = np.where(elevations < WALL, 0.035, WALL)
    xs = XSection(elevations.copy(), n.copy(), spacing)
    thalweg = elevations[elevations.size // 2]
    stages = np.linspace(thalweg + 0.01, max(old, exact) + 1.0, 1500)
    q = np.array([discharge(xs, cell["slope"], wse=w) for w in stages])
    ax = axes[0]
    ax.plot(q, stages - thalweg, color=GROUND, lw=1.2, label="discharge at each water level")
    ax.axvline(cell["qmax"], color=ACCENT, lw=1.0, ls="--", label="the maximum flow")
    ax.plot([discharge(xs, cell["slope"], wse=exact)], [exact - thalweg], "o", color=NEW, ms=7,
            label=f"new: the lowest level carrying it ({exact - thalweg:.2f} m)")
    ax.plot([q_old], [old - thalweg], "x", color=LEGACY, ms=9, mew=2, label=f"legacy's answer ({old - thalweg:.2f} m)")
    ax.set_xlabel("discharge (m³/s)")
    ax.set_ylabel("depth above the stream cell (m)")
    ax.set_title(f"{ctx.site_label(site)[:28]}: row {cell['row']}, column {cell['col']}")
    ax.legend(loc="lower right", fontsize=6.8)
    ax = axes[1]
    edges = np.array([-np.inf, -0.5, -0.05, -0.01, -0.002, 0.002, 0.01, 0.05, 0.5, np.inf])
    counts, _ = np.histogram(difference, bins=edges)
    labels = ["< −0.5", "−0.5…−0.05", "−0.05…−0.01", "−0.01…−0.002", "±0.002", "0.002…0.01", "0.01…0.05",
              "0.05…0.5", "> 0.5"]
    ax.bar(range(len(counts)), counts, color=NEW, alpha=0.85)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("legacy's water surface − the exact lowest (m)")
    ax.set_ylabel("cross sections")
    ax.set_title(f"{difference.size} cross sections of {len(ctx.detail_sites)} sites")
    return fig, dict(caption=(
        "The water surface each code finds for the maximum flow, on the same sampled cross sections with constant n "
        "and no banks. The new code solves exactly for the lowest level whose discharge reaches it. Legacy ran "
        "Brent's method to a millimetre and three rounds of steps, keeping whichever came closer, so it was usually "
        "within a centimetre; where the discharge doesn't rise steadily with the water (as where water spreads onto "
        "flat ground) it could settle on a higher crossing."),
                 stats=dict(sections=int(difference.size),
                            within_2mm=rounded(np.mean(np.abs(difference) <= 0.002), 3),
                            within_1cm=rounded(np.mean(np.abs(difference) <= 0.01), 3),
                            over_5cm=int(np.sum(np.abs(difference) > 0.05)),
                            largest=rounded(difference.max(), 3)))


def _legacy_section(cell: dict):
    """A legacy cross section from its capture, as an XSection: side 2 on the left, side 1 on the right, the shorter
    side padded with walls so the stream cell is in the middle, and legacy's bank indices as bank distances."""
    from arc.xsection.xsection import XSection
    side1, side2 = cell["final"]
    n1, n2 = cell["n"]
    half = max(side1.size, side2.size) - 1
    def padded(values, fill):
        out = np.full(half + 1, fill)
        out[:values.size] = values
        return out
    elevations = np.concatenate([padded(side2, WALL)[:0:-1], padded(side1, WALL)])
    n = np.concatenate([padded(n2, WALL)[:0:-1], padded(n1, WALL)])
    xs = XSection(elevations, n, cell["spacing"])
    b1, b2 = cell["hydraulic_banks"]
    if b1 > 0 and b2 > 0:
        xs.left_bank_distance, xs.right_bank_distance = b2 * cell["spacing"], b1 * cell["spacing"]
    return xs


def _capture_sides(cell: dict):
    """A legacy capture's cross section as legacy's discharge function takes it."""
    (side1, side2), (n1, n2) = cell["final"], cell["n"]
    return (np.append(side1, SENTINEL), side1.size, np.append(n1, n1[-1]), np.append(side2, SENTINEL), side2.size,
            np.append(n2, n2[-1]))


def _legacy_top(cell: dict, count: int, roughness) -> tuple[float, float] | None:
    """Legacy's depth step between increments (its maximum flow's water surface, over the count, rounded to the
    millimetre) and its discharge at that water surface, which capped the increments."""
    parameters = (6.0, 1.0, 1.0) if roughness is None else tuple(roughness)
    thalweg = float(cell["final"][0][0])
    wse, q = _legacy_max_flow_wse(_capture_sides(cell), cell["spacing"], cell["slope"], cell["qmax"], thalweg,
                                  parameters, tuple(cell["hydraulic_banks"]))
    if not wse > thalweg:
        return None
    return round((wse - thalweg) / count, 3), q


def _increments_at(xs, step: float, q_top: float, count: int, slope: float, baseflow: float, roughness):
    """The new code's increments on a cross section at legacy's own depth step and cap (arc.rating_curve's
    _increments, with rating_curve's baseflow rule)."""
    from arc.hydraulics import _parameters, hydraulic_profile
    from arc.rating_curve import BASEFLOW_MARGIN, Q, _increments
    profile = hydraulic_profile(xs)
    parameters = _parameters(roughness)
    left, right = float(xs.left_bank_distance), float(xs.right_bank_distance)
    thalweg = float(profile.elevations[profile.center])
    top = thalweg + step * count
    scale = math.sqrt(slope)
    out = np.full((count, 5), np.nan)
    start, last = _increments(*profile, left, right, *parameters, thalweg, top, count, scale, float(q_top), out)
    if last > start and baseflow > BASEFLOW_MARGIN and out[start, Q] >= baseflow:
        out[start, Q] = baseflow - BASEFLOW_MARGIN
    return out


def _rating_curves_on_legacy_geometry(ctx, config, roughness, sites, at_legacy_step=False):
    """New rating curves on legacy's own cross sections (its capture), next to legacy's VDT rows for them: the new
    code's whole rating curve, or with at_legacy_step, its increments at legacy's own depth step."""
    from arc.rating_curve import rating_curve
    pairs = []
    for site in sites:
        capture = ctx.capture(config, "legacy", site)
        vdt = ctx.vdt(config, "legacy", site)
        if capture is None or vdt is None:
            continue
        count = increments(vdt)
        rows = {(int(r.COMID), int(r.Row), int(r.Col)): r for r in vdt.itertuples(index=False)}
        for cell in capture["cells"]:
            if cell is None:
                continue
            row = rows.get((cell["comid"], *cell["center"]))
            if row is None:
                continue
            xs = _legacy_section(cell)
            if at_legacy_step:
                top = _legacy_top(cell, count, roughness)
                if top is None or not top[0] > 0:
                    continue
                values = _increments_at(xs, top[0], top[1], count, cell["slope"], cell["baseflow"], roughness)
            else:
                curve = rating_curve(xs, cell["qmax"], cell["slope"], count, baseflow=cell["baseflow"],
                                     roughness=roughness)
                if curve is None or not curve.valid:
                    continue
                values = curve.increments
            old = np.array([[getattr(row, f"{p}_{i}") for p in ("q", "v", "t", "wse")] for i in range(1, count + 1)])
            pairs.append((old, values[:, :4]))
    return pairs


@figure("R1", "Sanity check: the new rating curve on legacy's own cross sections", "Rating curves")
def rating_curve_sanity(ctx):
    from arc.hydraulics import DepthRoughness
    cases = (("uniform_n", None, False, True, "same n, legacy's step"),
             ("uniform_n", None, False, False, "same n, the new top"),
             ("as_configured", DepthRoughness(), False, False, "as configured"))
    fig, axes = plt.subplots(len(cases), 4, figsize=(13.5, 3.2 * len(cases)))
    stats = {}
    for row, (config, roughness, _, at_step, label) in enumerate(cases):
        pairs = _rating_curves_on_legacy_geometry(ctx, config, roughness, ctx.sites, at_legacy_step=at_step)
        old = np.concatenate([p[0] for p in pairs])
        new = np.concatenate([p[1] for p in pairs])
        keep = np.all(np.isfinite(old), axis=1) & np.all(np.isfinite(new), axis=1) & (old[:, 0] > 0)
        old, new = old[keep], new[keep]
        rng = np.random.default_rng(0)
        shown = rng.choice(old.shape[0], size=min(40000, old.shape[0]), replace=False)
        stats[label] = dict(cells=len(pairs), increments=int(old.shape[0]))
        for j, (name, unit) in enumerate((("discharge", "m³/s"), ("velocity", "m/s"), ("top width", "m"),
                                          ("water surface", "m"))):
            ax = axes[row, j]
            x, y = old[shown, j], new[shown, j]
            if j == 3:
                signed = new[:, j] - old[:, j]
                difference = np.abs(signed)
                text = f"|Δ| median {np.median(difference) * 1000:.1f} mm\np99 {np.percentile(difference, 99):.3f} m"
                stats[label][name] = dict(median_m=rounded(np.median(difference), 4),
                                          p99_m=rounded(np.percentile(difference, 99), 4),
                                          within_1mm=rounded(np.mean(difference <= 0.0015), 4))
                text += f"\nwithin 1 mm: {np.mean(difference <= 0.0015):.1%}"
                ax.hist(np.clip(signed, -0.1, 0.1), bins=np.linspace(-0.1, 0.1, 81), color=NEW, alpha=0.8)
                ax.set_yscale("log")
                ax.set_xlabel("new − legacy water surface (m)")
                ax.set_ylabel("increments")
            else:
                relative = np.abs(new[:, j] / np.maximum(old[:, j], 1e-6) - 1)
                text = f"|Δ|/legacy median {np.median(relative):.1e}\np99 {np.percentile(relative, 99):.1e}"
                stats[label][name] = dict(median=float(np.median(relative)), p99=float(np.percentile(relative, 99)))
                ax.loglog(x, y, ".", ms=1, color=NEW, alpha=0.3)
                lims = [max(min(x.min(), y.min()), 1e-3), max(x.max(), y.max())]
                ax.plot(lims, lims, color=GROUND, lw=0.8)
                ax.set_xlabel(f"legacy {name} ({unit})")
                ax.set_ylabel(f"new {name} ({unit})")
            note(ax, text)
            ax.set_title(f"{label}: {name}", fontsize=8.5)
    return fig, dict(caption=(
        "New rating curves (arc.rating_curve) worked out on legacy's own cross sections, with its roughness, banks, "
        "slope and maximum flow, from its runs on every site, against legacy's VDT rows for the same cells (every "
        "increment). Top: every land cover class given n = 0.035, no bathymetry, no angle search and constant n, "
        "with the new increments taken at legacy's own depth step and discharge cap: they agree to legacy's rounding "
        "(3 decimals, and the millimetre it rounded each water surface to). The few that don't are legacy's all-zero "
        "first increments (R4), and last increments where legacy's millimetre rounding tipped the 1% discharge cap "
        "(1.6% of cells). Middle: the same, with the new code's own top, the exact "
        "lowest water surface carrying the maximum flow: most increments move by a few millimetres, and where "
        "legacy's search settled higher (H4) every increment of that cell moves. Bottom: as the sites run, legacy's "
        "carved cross sections with the sites' n, depth-varying: the segment conventions of H2 add their spread. "
        "Cells whose rating curve legacy's slope search changed are left out (legacy wrote none for them)."),
                 stats=stats)


@figure("R2", "The discharge at the top of the rating curve", "Rating curves")
def rating_curve_top(ctx):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    stats = {}
    for code, color in (("legacy", LEGACY), ("new", NEW)):
        ratios = []
        for site in ctx.sites:
            capture, vdt = ctx.capture("as_configured", code, site), ctx.vdt("as_configured", code, site)
            if capture is None or vdt is None:
                continue
            qmax = {(c["comid"], *c["center"]): c["qmax"] for c in capture["cells"] if c is not None}
            count = increments(vdt)
            for row in vdt.itertuples(index=False):
                q = qmax.get((int(row.COMID), int(row.Row), int(row.Col)))
                if q:
                    ratios.append(getattr(row, f"q_{count}") / q)
        ratios = np.array(ratios)
        bins = np.linspace(0.45, 1.55, 111)
        ax.hist(np.clip(ratios, bins[0], bins[-1]), bins=bins, histtype="step", color=color, lw=1.3,
                label=f"{code}: {np.mean(np.abs(ratios - 1) <= 0.01):.0%} within 1%", density=True)
        stats[code] = dict(rows=int(ratios.size), within_1pct=rounded(np.mean(np.abs(ratios - 1) <= 0.01), 3),
                           median=rounded(np.median(ratios), 4))
    ax.set_yscale("log")
    ax.set_xlabel("discharge at the last increment / the maximum flow")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    return fig, dict(caption=(
        "Each VDT row's discharge at its last increment over the cell's maximum flow, on every site as configured. "
        "The new top is the exact lowest water surface carrying the maximum flow, so it's the maximum flow unless the "
        "water spills over a bank there (the discharge jumps past it) or the last increments are held (legacy's "
        "rules for falling discharge, kept). Both accept a top within half of the maximum flow either way."),
                 stats=stats)


def _depth_of(row, count):
    return np.array([getattr(row, f"wse_{i}") for i in range(1, count + 1)]), \
        np.array([getattr(row, f"q_{i}") for i in range(1, count + 1)])


@figure("R3", "Rating curves at six real cells", "Rating curves")
def example_rating_curves(ctx):
    site = ctx.detail_sites[0]
    merged = ctx.merged_vdt("as_configured", site)
    count = increments(merged, "_l")
    top = np.abs(merged[f"wse_{count}_n"] - merged[f"wse_{count}_l"]).to_numpy()
    order = np.argsort(top)
    picks = [order[int(q * (order.size - 1))] for q in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
    plain = {config: {(int(r.COMID), int(r.Row), int(r.Col)): r for r in ctx.vdt("no_bathymetry", config,
                                                                                    site).itertuples(index=False)}
             for config in ("legacy", "new")}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, k, share in zip(axes.flat, picks, (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)):
        row = merged.iloc[k]
        key = (int(row.COMID), int(row.Row), int(row.Col))
        for code, color in (("l", LEGACY), ("n", NEW)):
            wse = np.array([row[f"wse_{i}_{code}"] for i in range(1, count + 1)])
            q = np.array([row[f"q_{i}_{code}"] for i in range(1, count + 1)])
            ax.plot(q, wse, "o-", color=color, ms=2.5, lw=1.2,
                    label=("legacy" if code == "l" else "new") + ", as configured")
            other = plain["legacy" if code == "l" else "new"].get(key)
            if other is not None:
                w2, q2 = _depth_of(other, count)
                ax.plot(q2, w2, "--", color=color, lw=1.0, alpha=0.8,
                        label=("legacy" if code == "l" else "new") + ", no bathymetry")
        ax.axhline(row.Elev_l, color=GROUND, lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"row {key[1]}, column {key[2]} ({share:.0%} of cells differ less at the top)", fontsize=8)
        ax.set_xlabel("discharge (m³/s)")
        ax.set_ylabel("water surface elevation (m)")
    axes.flat[0].legend(loc="upper left", fontsize=6.5)
    return fig, dict(caption=(
        f"Rating curves from the VDT databases at six cells of {ctx.site_label(site)}, chosen by how much the top "
        "water surfaces differ (10th to 99th percentile). Solid: as the site is configured, with bathymetry. Dashed: "
        "without bathymetry. The dotted line is the DEM at the cell. The new channels are carved as their own "
        "trapezoids (sub-cell where narrower than a cell), which mostly changes the lowest increments."),
                 stats=dict(site=site))


@figure("R4", "Legacy's rating curves that started at a water surface of 0 m", "Rating curves")
def zero_first_increments(ctx):
    names, legacy_counts, new_counts = [], [], []
    for site in ctx.sites:
        old, new = ctx.vdt("as_configured", "legacy", site), ctx.vdt("as_configured", "new", site)
        if old is None or new is None:
            continue
        a, b = int((old["wse_1"] == 0).sum()), int((new["wse_1"] == 0).sum())
        if a or b:
            names.append(ctx.site_label(site)[:34])
            legacy_counts.append(a)
            new_counts.append(b)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.28 * len(names) + 1)))
    y = np.arange(len(names))
    ax.barh(y - 0.2, legacy_counts, height=0.4, color=LEGACY, label=f"legacy ({sum(legacy_counts)} rows)")
    ax.barh(y + 0.2, new_counts, height=0.4, color=NEW, label=f"new ({sum(new_counts)} rows)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("VDT rows whose first increment is all zeros, its water surface 0 m")
    ax.legend(loc="lower right")
    return fig, dict(caption=(
        "Legacy rounded each increment's area to 3 decimals and wrote an increment whose area rounded to 0 as all "
        "zeros, its water surface at 0 m, keeping the rating curve. That happens where the first increment is a "
        "couple of centimetres deep in a narrow channel. Nothing is rounded before writing now."),
                 stats=dict(legacy_rows=sum(legacy_counts), new_rows=sum(new_counts), sites=len(names)))
