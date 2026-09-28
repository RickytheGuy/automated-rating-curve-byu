"""Cross-section sampling (arc.xsection.sampling): where the ordinates are, how far apart, and what's beyond the
raster."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from vc_plot import (GROUND, LEGACY, NEW, Window, figure, hillshade, legacy_stations, new_stations, note, rounded,
                     show_raster)

SITE_CELL = (23.22, 30.85)  # Cuyahoga's cells, in metres: dx, dy
SQUARE_CELL = (30.0, 30.0)
LENGTH = 5000.0  # the default X_Section_Dist
LEGACY_STEP = math.pi / 30  # legacy's precomputed directions were 6 degrees apart


def legacy_tables(dx: float, dy: float, length: float = LENGTH):
    """Legacy's precomputed ordinate offsets and interpolation weights for its 31 directions (0, 6, ... 180
    degrees), and its ordinate spacing for each."""
    from arc.cross_section import CrossSection
    return CrossSection.create_cross_section_ordinates({"d_x_section_distance": length, "dx": dx, "dy": dy})


def legacy_index(xs_direction: float) -> int:
    """The precomputed direction legacy sampled a cross section in (_sample_cross_section_for_cell)."""
    angle = xs_direction - math.pi if xs_direction > math.pi else xs_direction
    return int(round(angle / LEGACY_STEP))


def legacy_offsets(tables, j: int, count: int | None = None):
    """Where legacy's side-1 ordinates really are, as row and column offsets from the stream cell: its two cells
    for each ordinate, weighted as it weighted them (its weights are the wrong way round, so this is the point whose
    interpolated value it read, not the one it meant). Side 2 is the same offsets negated."""
    index, _, fractions = tables
    count = index.shape[1] - 1 if count is None else count
    main_r, main_c, second_r, second_c = (index[j, :count, i].astype(float) for i in range(4))
    wm, ws = fractions[j, :count, 0], fractions[j, :count, 1]
    return main_r * wm + second_r * ws, main_c * wm + second_c * ws


def new_offsets(xs_direction: float, dx: float, dy: float, count: int):
    """The new code's side-1 (increasing distance) ordinates as row and column offsets, and their spacing."""
    spacing = 1.0 / max(abs(math.cos(xs_direction)) / dx, abs(math.sin(xs_direction)) / dy)
    k = np.arange(count)
    return math.sin(xs_direction) * spacing * k / dy, math.cos(xs_direction) * spacing * k / dx, spacing


def _fold(degrees):
    """An angle difference in degrees, folded into (-90, 90]."""
    return (np.asarray(degrees) + 90.0) % 180.0 - 90.0


@figure("S1", "Where the ordinates of an oblique cross section are", "Cross-section sampling")
def ordinates_at_oblique_angles(ctx):
    dx, dy = SITE_CELL
    tables = legacy_tables(dx, dy)
    requested = (25.0, 45.0, 65.0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    stats = {}
    count = 7
    for ax, degrees in zip(axes, requested):
        angle = math.radians(degrees)
        j = legacy_index(angle)
        lr, lc = legacy_offsets(tables, j, count)
        nr, nc, spacing = new_offsets(angle, dx, dy, count)
        # the cells, drawn their true shape
        rows = np.arange(-2, 9)
        cols = np.arange(-2, 10)
        for c in cols:
            ax.axvline((c - 0.5) * dx, color="#dddddd", lw=0.6, zorder=0)
        for r in rows:
            ax.axhline(-(r - 0.5) * dy, color="#dddddd", lw=0.6, zorder=0)
        cc, rr = np.meshgrid(cols, rows)
        ax.plot(cc * dx, -rr * dy, ".", color="#bbbbbb", ms=3, zorder=1)
        reach = (count - 1) * max(spacing, 40)
        ax.plot([0, reach * math.cos(angle)], [0, -reach * math.sin(angle)], color=GROUND, lw=0.8, ls=":",
                label=f"the line at {degrees:g} degrees")
        ax.plot(nc * dx, -nr * dy, "o-", color=NEW, ms=5, lw=1.2, label="new ordinates")
        actual = np.hypot(np.diff(lc * dx), np.diff(lr * dy))
        ax.plot(lc * dx, -lr * dy, "x-", color=LEGACY, ms=6, mew=1.5, lw=1.0,
                label=f"legacy's (snapped to {math.degrees(j * LEGACY_STEP):g}°)")
        achieved = math.degrees(math.atan2(lr[-1] * dy, lc[-1] * dx))
        ax.set_title(f"asked for {degrees:g}°: legacy's ordinates run at {achieved:.1f}°")
        ax.set_aspect("equal")
        ax.set_xlim(-1.2 * dx, 7.2 * dx)
        ax.set_ylim(-6.6 * dy, 1.0 * dy)
        ax.set_xlabel("metres east of the stream cell")
        ax.grid(False)
        ax.legend(loc="lower left", fontsize=6.8, framealpha=0.9, frameon=True)
        stats[f"{degrees:g}"] = dict(legacy_sampled_at=round(math.degrees(j * LEGACY_STEP), 1),
                                     legacy_runs_at=round(achieved, 2), legacy_spacing=round(float(actual.mean()), 2),
                                     legacy_reported_spacing=round(float(tables[1][j]), 2), new_spacing=round(spacing, 2))
    axes[0].set_ylabel("metres north of the stream cell")
    fig.suptitle(f"One side of a cross section on {dx:g} m × {dy:g} m cells (drawn to scale; grey dots are cell centres)",
                 fontsize=9.5)
    return fig, dict(caption=(
        "The first ordinates out from the stream cell of cross sections asked for at 25°, 45° and 65° (from east "
        "towards south, on Cuyahoga's cell size). The new ordinates lie on the line, one per row or column of cells "
        "crossed. Legacy snapped the direction to the nearest 6°, took its column offsets with cos where it needed "
        "cot, and swapped its two interpolation weights, so the points whose values it read zigzag off the line and "
        "run in another direction."), stats=stats)


@figure("S2", "The direction a cross section really runs in", "Cross-section sampling")
def achieved_directions(ctx):
    requested = np.arange(0.0, 180.01, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    stats = {}
    for ax, (dx, dy), name in zip(axes, (SQUARE_CELL, SITE_CELL), ("square 30 m cells", "Cuyahoga's 23.2 × 30.9 m cells")):
        tables = legacy_tables(dx, dy)
        errors, snapped = [], []
        for degrees in requested:
            angle = math.radians(degrees)
            j = legacy_index(angle % math.pi)
            lr, lc = legacy_offsets(tables, j, 41)
            achieved = math.degrees(math.atan2(lr[-1] * dy, lc[-1] * dx)) % 180.0
            errors.append(_fold(achieved - degrees))
            snapped.append(_fold(math.degrees(j * LEGACY_STEP) - degrees))
        errors, snapped = np.array(errors), np.array(snapped)
        ax.plot(requested, snapped, color=GROUND, lw=0.8, ls=":", label="legacy's snapping to 6° alone")
        ax.plot(requested, errors, color=LEGACY, lw=1.2, label="legacy, as sampled")
        ax.axhline(0.0, color=NEW, lw=1.5, label="new (exactly as asked)")
        ax.set_xticks(range(0, 181, 30))
        ax.set_xlabel("direction asked for (degrees from east towards south)")
        ax.set_title(name)
        ax.legend(loc="lower left")
        stats[name] = dict(mean_abs_error=round(float(np.mean(np.abs(errors))), 2),
                           max_abs_error=round(float(np.max(np.abs(errors))), 2),
                           at_45=round(float(errors[np.searchsorted(requested, 45.0)]), 2),
                           at_30=round(float(errors[np.searchsorted(requested, 30.0)]), 2))
    axes[0].set_ylabel("direction sampled − direction asked (°)")
    return fig, dict(caption=(
        "How far the direction of legacy's ordinates (the line from the stream cell to its 40th ordinate) is from "
        "the direction it was asked for. Only along rows and columns (0°, 90°, 180°) is it right. The new sampler "
        "takes every direction exactly."), stats=stats)


def _grid_lines(ax, dx, dy, rows, cols, color="#e2e2e2"):
    """Cell edges around the stream cell (row and column offsets), to scale: x east, y north, in metres."""
    for c in range(cols[0], cols[1] + 2):
        ax.plot([(c - 0.5) * dx] * 2, [-(rows[0] - 0.5) * dy, -(rows[1] + 0.5) * dy], color=color, lw=0.6, zorder=0)
    for r in range(rows[0], rows[1] + 2):
        ax.plot([(cols[0] - 0.5) * dx, (cols[1] + 0.5) * dx], [-(r - 0.5) * dy] * 2, color=color, lw=0.6, zorder=0)


def _arrow(ax, angle_xy, length, color, lw=1.8, both=True, label=None):
    """An arrow from the stream cell along a ground direction (radians, x east, y south positive rows)."""
    ux, uy = math.cos(angle_xy), -math.sin(angle_xy)
    start = (-length * ux, -length * uy) if both else (0.0, 0.0)
    ax.annotate("", xy=(length * ux, length * uy), xytext=start, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9))
    if label:
        ax.plot([], [], color=color, lw=lw, label=label)


def _ground_angle(cell_angle: float, dx: float, dy: float) -> float:
    """The direction on the ground of an angle measured in cells (as legacy's stream direction was)."""
    return math.atan2(math.sin(cell_angle) * dy, math.cos(cell_angle) * dx)


@figure("S2b", "The directions on the grid: legacy's and the new vectors from the same cells", "Cross-section sampling")
def direction_vectors(ctx):
    from arc.Automated_Rating_Curve_Generator import get_stream_direction_information
    site = ctx.detail_sites[0]
    grid = ctx.grid(site)
    dx, dy = grid.dx, grid.dy
    tables = legacy_tables(dx, dy)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    stats = {}

    # (a) the fan: cross sections asked for every 15 degrees, as each code samples them
    ax = axes[0, 0]
    count = 9
    _grid_lines(ax, dx, dy, (-count, count), (-count, count))
    worst = 0.0
    for degrees in np.arange(0.0, 180.0, 15.0):
        angle = math.radians(degrees)
        j = legacy_index(angle % math.pi)
        lr, lc = legacy_offsets(tables, j, count + 1)
        nr, nc, _ = new_offsets(angle, dx, dy, count + 1)
        ax.plot(np.r_[-nc[::-1], nc[1:]] * dx, -np.r_[-nr[::-1], nr[1:]] * dy, "-", color=NEW, lw=1.6, alpha=0.9)
        ax.plot(np.r_[-lc[::-1], lc[1:]] * dx, -np.r_[-lr[::-1], lr[1:]] * dy, "--", color=LEGACY, lw=1.1)
        ax.plot([lc[-1] * dx, -lc[-1] * dx], [-lr[-1] * dy, lr[-1] * dy], "x", color=LEGACY, ms=5)
        achieved = math.degrees(math.atan2(lr[-1] * dy, lc[-1] * dx)) % 180.0
        worst = max(worst, abs(_fold(achieved - degrees)))
    ax.plot([], [], color=NEW, lw=1.6, label="new: exactly as asked")
    ax.plot([], [], "--x", color=LEGACY, lw=1.3, label="legacy, as sampled")
    ax.set_aspect("equal")
    ax.set_title(f"cross sections asked for every 15° ({count} ordinates each side)", fontsize=8.5)
    ax.set_xlabel("metres east of the stream cell")
    ax.set_ylabel("metres north")
    ax.legend(loc="lower left", fontsize=6.8, framealpha=0.9, frameon=True)
    ax.grid(False)
    stats["fan_worst_deg"] = rounded(worst, 2)

    # (b)-(f) real stream cells: each code's stream vector from the same cells, and the cross section it sampled
    legacy, new = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
    by_cell = {(c["row"], c["col"]): c for c in new["cells"] if c is not None}
    candidates = []
    for c in legacy["cells"]:
        if c is None or (c["row"], c["col"]) not in by_cell:
            continue
        n = by_cell[(c["row"], c["col"])]
        old_stream, _ = get_stream_direction_information(c["row"] + PAD_CELLS, c["col"] + PAD_CELLS,
                                                         _padded_streams(ctx, site), 5)
        ground = _ground_angle(old_stream, dx, dy)
        difference = abs(_fold(math.degrees(ground - n["initial_direction"])))
        candidates.append((difference, c, n, old_stream))
    candidates.sort(key=lambda t: -t[0])
    picks = []
    for difference, c, n, old_stream in candidates:
        if all(max(abs(c["row"] - p[1]["row"]), abs(c["col"] - p[1]["col"])) > 12 for p in picks):
            picks.append((difference, c, n, old_stream))
        if len(picks) == 3:
            break
    typical = candidates[len(candidates) // 2:]
    for item in typical:
        if all(max(abs(item[1]["row"] - p[1]["row"]), abs(item[1]["col"] - p[1]["col"])) > 12 for p in picks):
            picks.append(item)
        if len(picks) == 5:
            break
    streams = grid.streams
    box = 5
    for ax, (difference, c, n, old_stream) in zip(axes.flat[1:], picks):
        r0, c0 = c["row"], c["col"]
        span = box + 2
        _grid_lines(ax, dx, dy, (-span, span), (-span, span))
        reach = streams[r0, c0]
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                r, cc = r0 + dr, c0 + dc
                if 0 <= r < streams.shape[0] and 0 <= cc < streams.shape[1] and streams[r, cc] > 0:
                    inside = abs(dr) <= box and abs(dc) <= box and streams[r, cc] == reach
                    ax.add_patch(plt.Rectangle(((dc - 0.5) * dx, -(dr + 0.5) * dy), dx, dy, lw=0,
                                               color="#8fc3e6" if inside else "#d7e9f5", zorder=0.5))
        ax.add_patch(plt.Rectangle(((-box - 0.5) * dx, -(box + 0.5) * dy), (2 * box + 1) * dx, (2 * box + 1) * dy,
                                   fill=False, ls=":", lw=0.8, color=GROUND))
        length = (box + 0.5) * min(dx, dy)
        old_ground = _ground_angle(old_stream, dx, dy)
        _arrow(ax, old_ground, length, LEGACY, label=f"legacy's stream direction: fitted in cells, "
                                                    f"{math.degrees(old_ground) % 180:.1f}° on the ground")
        _arrow(ax, n["initial_direction"], length, NEW, label=f"new: principal axis in metres, "
                                                            f"{math.degrees(n['initial_direction']) % 180:.1f}°")
        # the cross sections each then sampled, square to its stream direction (before the angle search)
        j = legacy_index(c["initial_xs_direction"] % math.pi)
        lr, lc = legacy_offsets(tables, j, box + 1)
        ax.plot(np.r_[-lc[::-1], lc[1:]] * dx, -np.r_[-lr[::-1], lr[1:]] * dy, "x", color=LEGACY, ms=5, mew=1.2,
                label="legacy's cross section, as sampled")
        nr, nc, _ = new_offsets(n["initial_direction"] - math.pi / 2, dx, dy, box + 1)
        ax.plot(np.r_[-nc[::-1], nc[1:]] * dx, -np.r_[-nr[::-1], nr[1:]] * dy, "o", color=NEW, ms=3.5, mfc="none",
                label="new cross section's ordinates")
        ax.set_aspect("equal")
        ax.set_xlim(-(span + 0.5) * dx, (span + 0.5) * dx)
        ax.set_ylim(-(span + 0.5) * dy, (span + 0.5) * dy)
        ax.grid(False)
        ax.set_title(f"row {r0}, column {c0}: {difference:.1f}° apart", fontsize=8.5)
        ax.set_xlabel("metres east of the stream cell")
        ax.legend(loc="lower left", fontsize=5.8, framealpha=0.92, frameon=True)
        stats[f"{r0},{c0}"] = rounded(difference, 2)
    return fig, dict(caption=(
        f"Both codes' vectors drawn on the same grid, {dx:.1f} × {dy:.1f} m cells to scale ({ctx.site_label(site)}). "
        "Top left: cross sections asked for every 15°; the new ordinates run exactly as asked, "
        "legacy's (its 6° snapping and oblique geometry, S1) run off in other directions. The rest: real stream "
        "cells, three where the two stream directions differ most and two typical ones. The shaded cells are the "
        "reach's within Gen_Dir_Dist (the dotted box) that both fits use. Legacy fitted the cells' rows against their "
        "columns in cells, so on cells that aren't square its line isn't the stream's on the ground; the new "
        "direction is their principal axis in metres. The markers are the ordinates each code then sampled square "
        "to its stream direction, before the angle search."), stats=stats)


PAD_CELLS = 19


def _padded_streams(ctx, site):
    """The stream raster padded as legacy pads it, for its direction function."""
    key = ("padded_streams", site)
    if key not in ctx._cache:
        streams = ctx.grid(site).streams
        padded = np.zeros((streams.shape[0] + 2 * PAD_CELLS, streams.shape[1] + 2 * PAD_CELLS), dtype=streams.dtype)
        padded[PAD_CELLS:-PAD_CELLS, PAD_CELLS:-PAD_CELLS] = streams
        ctx._cache[key] = padded
    return ctx._cache[key]


@figure("S3", "How far apart the ordinates are, and how far the cross section reaches", "Cross-section sampling")
def ordinate_spacing(ctx):
    dx, dy = SITE_CELL
    tables = legacy_tables(dx, dy)
    count = tables[0].shape[1] - 1  # legacy's ordinates per side, the stream cell included
    requested = np.arange(0.0, 180.01, 0.5)
    new, reported, actual, legacy_reach, new_reach = [], [], [], [], []
    for degrees in requested:
        angle = math.radians(degrees)
        j = legacy_index(angle % math.pi)
        lr, lc = legacy_offsets(tables, j, count)
        spacing = 1.0 / max(abs(math.cos(angle)) / dx, abs(math.sin(angle)) / dy)
        new.append(spacing)
        reported.append(tables[1][j])
        actual.append(math.hypot(lr[-1] * dy, lc[-1] * dx) / (count - 1))
        legacy_reach.append(math.hypot(lr[-1] * dy, lc[-1] * dx))
        new_reach.append(int(LENGTH / 2 / spacing) * spacing)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(requested, new, color=NEW, lw=1.5, label="new: one per row or column crossed")
    ax.plot(requested, actual, color=LEGACY, lw=1.2, label="legacy, as its ordinates really are")
    ax.plot(requested, reported, color=LEGACY, lw=1.0, ls="--", label="legacy's Ordinate_Dist")
    ax.axhline(dx, color=GROUND, lw=0.6, ls=":")
    ax.axhline(dy, color=GROUND, lw=0.6, ls=":")
    ax.text(178, dx, "dx", ha="right", va="bottom", fontsize=7, color=GROUND)
    ax.text(178, dy, "dy", ha="right", va="bottom", fontsize=7, color=GROUND)
    ax.set_ylabel("metres between ordinates")
    ax.set_title("ordinate spacing")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(requested, new_reach, color=NEW, lw=1.5, label="new: half of X_Section_Dist")
    ax.plot(requested, legacy_reach, color=LEGACY, lw=1.2,
            label=f"legacy: {count - 1} ordinates, however far apart")
    ax.set_ylabel("metres from the stream cell to the end")
    ax.set_title(f"how far each side reaches (X_Section_Dist {LENGTH:g} m)")
    ax.legend(loc="lower left")
    for ax in axes:
        ax.set_xticks(range(0, 181, 30))
        ax.set_xlabel("direction (degrees from east towards south)")
    return fig, dict(caption=(
        f"On {dx:g} × {dy:g} m cells. The new spacing is one ordinate per row or column of cells crossed, so every "
        "ordinate lands on a row or column of cell centres, and each side reaches half of X_Section_Dist. Legacy "
        "always took the same number of ordinates, so its cross sections reached further along the columns than "
        "along the rows. Its Ordinate_Dist is the average spacing along its own line, which (S1, S2) wasn't in the "
        "direction asked for, and it changes in 6° steps."), stats=dict(new_spacing_range=[round(min(new), 2), round(max(new), 2)],
                              legacy_actual_spacing_range=[round(min(actual), 2), round(max(actual), 2)],
                              legacy_reach_range=[round(min(legacy_reach)), round(max(legacy_reach))]))


def matched_cells(legacy: dict, new: dict):
    """Pairs of (legacy cell, new cell) captured for the same stream cell."""
    by_cell = {(c["row"], c["col"]): c for c in new["cells"] if c is not None}
    return [(c, by_cell[(c["row"], c["col"])]) for c in legacy["cells"]
            if c is not None and (c["row"], c["col"]) in by_cell]


def legacy_sampler(dem32: np.ndarray, dx: float, dy: float, pad: int = 19, length: float = LENGTH):
    """Legacy's CrossSection on a padded copy of a float32 DEM, as legacy read it (without its +100 m)."""
    from arc.cross_section import CrossSection
    padded = np.zeros((dem32.shape[0] + 2 * pad, dem32.shape[1] + 2 * pad), dtype=np.float32)
    padded[pad:-pad, pad:-pad] = dem32
    params = {"d_x_section_distance": length, "dx": dx, "dy": dy, "d_degree_manipulation": 0.0,
              "d_degree_interval": 0.0, "i_boundary_number": pad, "nrows": dem32.shape[0], "ncols": dem32.shape[1],
              "b_FindBanksBasedOnLandCover": False, "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": 0.2,
              "b_bathy_use_banks": False, "s_output_bathymetry_path": ""}
    section = CrossSection(dx, dy, padded, np.zeros(padded.shape, dtype=np.uint8), None, params)
    section.associate_with_precomputed_index_arrays(*CrossSection.create_cross_section_ordinates(params))
    return section


@figure("S5", "Sanity check: along rows and columns both samplers read the same values", "Cross-section sampling")
def axis_aligned_sanity(ctx):
    from arc.xsection.sampling import sample_elevations
    site = ctx.detail_sites[0]
    grid = ctx.grid(site)
    dem32 = np.where(grid.dem >= 9999.0, -9999.0, grid.dem).astype(np.float32)
    dem64 = dem32.astype(np.float64)
    legacy = legacy_sampler(dem32, grid.dx, grid.dy)
    rows, cols = np.nonzero(grid.streams > 0)
    rng = np.random.default_rng(0)
    pick = rng.choice(rows.size, size=min(300, rows.size), replace=False)
    pairs = {"along a row (legacy's 0°)": [], "along a column (legacy's 90°)": []}
    example = None
    for k in pick:
        r, c = int(rows[k]), int(cols[k])
        for name, j, xs_direction in (("along a row (legacy's 0°)", 0, 0.0),
                                      ("along a column (legacy's 90°)", 15, math.pi / 2)):
            legacy.set_cross_section(r + 19, c + 19, j, j * LEGACY_STEP)
            if legacy.xs1_n < 5 or legacy.xs2_n < 5:
                continue
            values, spacing = sample_elevations(dem64, r, c, xs_direction + math.pi / 2, LENGTH, grid.dx, grid.dy)
            center = values.size // 2
            n1 = min(legacy.xs1_n, values.size - center)
            n2 = min(legacy.xs2_n, center + 1)
            side1 = legacy.da_xs_profile1[:n1]
            side2 = legacy.da_xs_profile2[:n2]
            ok = (values[center:center + n1] < 9999) & (side1 > -9000)
            pairs[name].append((side1[ok], values[center:center + n1][ok]))
            ok2 = (values[center - n2 + 1:center + 1][::-1] < 9999) & (side2 > -9000)
            pairs[name].append((side2[ok2], values[center - n2 + 1:center + 1][::-1][ok2]))
            if example is None and name.startswith("along a column"):
                example = (r, c, legacy.da_xs_profile1[:legacy.xs1_n].copy(), legacy.da_xs_profile2[:legacy.xs2_n].copy(),
                           legacy.d_ordinate_dist, values, spacing)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    stats = {}
    for ax, (name, values) in zip(axes[:2], pairs.items()):
        old = np.concatenate([a for a, _ in values])
        new = np.concatenate([b for _, b in values])
        difference = np.abs(old.astype(np.float64) - new)
        ax.plot(old, new, ".", ms=1.5, color=NEW, alpha=0.4)
        lims = [min(old.min(), new.min()), max(old.max(), new.max())]
        ax.plot(lims, lims, color=GROUND, lw=0.8)
        ax.set_xlabel("legacy's sampled elevation (m)")
        ax.set_ylabel("new sampled elevation (m)")
        ax.set_title(name)
        note(ax, f"{old.size} ordinates of {len(values) // 2} cross sections\nlargest difference {difference.max():.1e} m")
        stats[name] = dict(ordinates=int(old.size), max_abs_difference=float(difference.max()))
    ax = axes[2]
    if example is not None:
        r, c, side1, side2, legacy_spacing, values, spacing = example
        x, y = legacy_stations(side1, side2, legacy_spacing, side_one=1)
        ax.plot(new_stations(values.size, spacing), np.where(values < 9999, values, np.nan), color=NEW, lw=2.5,
                label="new", alpha=0.6)
        ax.plot(x, y, color=LEGACY, lw=0.9, label="legacy")
        ax.set_xlim(-800, 800)
        inside = np.abs(x) <= 800
        ax.set_ylim(np.nanmin(y[inside]) - 2, np.nanmax(y[inside]) + 2)
        ax.set_title(f"one of them, at row {r}, column {c}")
        ax.set_xlabel("metres from the stream cell")
        ax.set_ylabel("elevation (m)")
        ax.legend()
    return fig, dict(caption=(
        f"Cross sections along rows and columns through 300 stream cells of {ctx.site_label(site)}, sampled by "
        "legacy's CrossSection and by the new sampler from the same float32 DEM. Every ordinate agrees, to "
        "floating-point rounding (the new code's cosine of 90° is 6e-17, not 0), so "
        "any difference in the real runs comes from the direction, the ordinate positions at other angles, or how "
        "far the cross section reaches."), stats=stats)


@figure("S6", "Beyond the raster's edge, and cells without data", "Cross-section sampling")
def edges_and_nodata(ctx):
    """A cross section running off the raster, and one crossing nodata, as each code samples them."""
    from arc.xsection.sampling import sample_elevations
    site = next((s for s in ctx.detail_sites if s.startswith("Hohokus")), ctx.detail_sites[0])
    grid = ctx.grid(site)
    raw = grid.reference.read_array(np.float64)
    nodata = grid.reference.nodata_value
    missing = (raw == nodata) if nodata is not None else ~np.isfinite(raw)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    stats = dict(site=site, nodata_cells=int(missing.sum()))
    # the map: where the nodata is
    window = Window(0, grid.dem.shape[0] - 1, 0, grid.dem.shape[1] - 1, grid.dx, grid.dy)
    ax = axes[0]
    shown = np.where(missing, np.nan, raw)
    show_raster(ax, window, shown, cmap="terrain")
    show_raster(ax, window, np.where(missing, 1.0, np.nan), cmap="Reds", vmin=0, vmax=1.2)
    sr, sc = np.nonzero(grid.streams > 0)
    x, y = window.xy(sr, sc)
    ax.plot(x, y, ",", color="navy")
    ax.set_title(f"{ctx.site_label(site)[:40]}\nno-data cells in red ({int(missing.sum())})")
    # a stream cell whose cross section, along a column, crosses nodata or the edge
    # legacy read the DEM as float32, nodata included, and raised it all by 100 m if anything was below 0 m
    offset = 100.0 if (raw < 0).any() else 0.0
    legacy32 = raw.astype(np.float32) + np.float32(offset)
    legacy = legacy_sampler(legacy32, grid.dx, grid.dy)

    def first_case(test):
        for k in np.random.default_rng(1).permutation(sr.size):
            r, c = int(sr[k]), int(sc[k])
            for j, xs_direction in ((0, 0.0), (15, math.pi / 2)):
                values, spacing = sample_elevations(grid.dem, r, c, xs_direction + math.pi / 2, LENGTH, grid.dx,
                                                    grid.dy)
                if test(values, r, c, xs_direction):
                    return r, c, j, xs_direction, values, spacing
        return None

    def crosses_nodata(values, r, c, xs_direction):
        center = values.size // 2
        near = values[max(center - 25, 0):center + 26]
        inside = 0 <= r - 30 and r + 30 < grid.dem.shape[0] and 0 <= c - 30 and c + 30 < grid.dem.shape[1]
        return inside and (near >= 9999).any()

    def reaches_edge(values, r, c, xs_direction):
        center = values.size // 2
        return (values[center:] >= 9999).any() and not (values[center:center + 40] >= 9999).any() and \
            not (values[:center] >= 9999).any()

    for ax, (test, name) in zip(axes[1:], ((crosses_nodata, "crossing cells without data"),
                                           (reaches_edge, "running off the raster"))):
        case = first_case(test)
        if case is None:
            ax.set_visible(False)
            continue
        r, c, j, xs_direction, values, spacing = case
        legacy.set_cross_section(r + 19, c + 19, j, j * LEGACY_STEP)
        lx, ly = legacy_stations(legacy.da_xs_profile1[:legacy.xs1_n] - offset,
                                 legacy.da_xs_profile2[:legacy.xs2_n] - offset, legacy.d_ordinate_dist)
        stations = new_stations(values.size, spacing)
        walls = values >= 9999
        ground = np.where(walls, np.nan, values)
        ax.plot(stations, ground, color=NEW, lw=2.2, alpha=0.6, label="new ground")
        top = np.nanmax(ground) + 5
        for s in stations[walls]:
            ax.axvspan(s - spacing / 2, s + spacing / 2, color=NEW, alpha=0.15, lw=0)
        ax.plot([], [], color=NEW, alpha=0.15, lw=6, label="new: walls (9999)")
        ax.plot(lx, ly, color=LEGACY, lw=0.9, label="legacy")
        low = np.nanmin(ground)
        if (ly < low - 50).any():
            deepest = ly.min()
            ax.annotate(f"legacy reads the no-data\ncells as {deepest:.0f} m pits", xy=(lx[np.argmin(ly)], low - 5),
                        xytext=(0.35, 0.25), textcoords="axes fraction", fontsize=7, color=LEGACY,
                        arrowprops=dict(arrowstyle="->", color=LEGACY, lw=0.8))
            stats["legacy_pit_depth"] = rounded(deepest, 1)
        ax.set_ylim(low - 10, top)
        if legacy.xs1_n and name.startswith("running"):
            end = lx.max() if lx.size else 0
            ax.axvline(end, color=LEGACY, lw=0.8, ls=":")
            ax.text(end, top, " legacy's last ordinate", fontsize=7, color=LEGACY, va="top")
            stats["legacy_stops_before_new_walls_m"] = rounded(stations[walls].min() - end, 1) if walls.any() else None
        ax.set_xlim(-1300, 1300)
        ax.set_title(f"a cross section {name}\n(row {r}, column {c})")
        ax.set_xlabel("metres from the stream cell")
        ax.set_ylabel("elevation (m)")
        ax.legend(loc="upper left", fontsize=6.8)
    return fig, dict(caption=(
        "Ordinates beyond the raster's edge, and cells without data, are walls in the new code (9999), which the "
        "water can't spread past. Legacy read cells without data as their no-data value, -9999 m (raised by the "
        "100 m it added to any DEM with elevations below 0 m), deep pits the water fell into, and ended each side "
        "one cell before the raster's edge."), stats=stats)
