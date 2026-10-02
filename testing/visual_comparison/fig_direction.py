"""The stream's direction, the angle search, the low spot, and the stream slopes (arc.xsection.orientation,
low_spot, stream_path and slope)."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from fig_sampling import LEGACY_STEP, LENGTH, SITE_CELL, SQUARE_CELL, legacy_index, legacy_offsets, legacy_sampler, \
    legacy_tables, matched_cells
from vc_plot import (ACCENT, GROUND, LEGACY, NEW, Window, figure, hillshade, note, rounded, show_raster)

PAD = 19


def _fold(degrees):
    return (np.asarray(degrees) + 90.0) % 180.0 - 90.0


def rasterise_line(angle: float, dx: float, dy: float, size: int = 61, offset: float = 0.0):
    """Stream cells along a straight line through the centre cell at angle (metres, from east towards south),
    shifted sideways by offset metres."""
    streams = np.zeros((size, size), dtype=np.int64)
    center = size // 2
    step = 0.1 * min(dx, dy)
    t = np.arange(-center * max(dx, dy) * 1.5, center * max(dx, dy) * 1.5, step)
    x = t * math.cos(angle) - offset * math.sin(angle)
    y = t * math.sin(angle) + offset * math.cos(angle)
    cols = np.rint(center + x / dx).astype(int)
    rows = np.rint(center + y / dy).astype(int)
    keep = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    streams[rows[keep], cols[keep]] = 1
    streams[center, center] = 1
    return streams, center


@figure("D1", "The stream's direction on straight streams", "Stream direction and the angle search")
def direction_on_straight_streams(ctx):
    from arc.Automated_Rating_Curve_Generator import get_stream_direction_information
    from arc.xsection.orientation import stream_direction
    distance = 5  # the sites' Gen_Dir_Dist
    angles = np.arange(0.0, 180.0, 1.0)
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    stats = {}
    for ax, (dx, dy), name in zip(axes, (SQUARE_CELL, SITE_CELL),
                                  ("square 30 m cells", "Cuyahoga's 23.2 × 30.9 m cells")):
        legacy_error, new_error = [], []
        for degrees in angles:
            le, ne = [], []
            for _ in range(12):
                streams, c = rasterise_line(math.radians(degrees), dx, dy, offset=rng.uniform(-0.3, 0.3) * min(dx, dy))
                old, _ = get_stream_direction_information(c, c, streams, distance)
                new = stream_direction(streams, c, c, distance, dx, dy)
                le.append(abs(_fold(math.degrees(old % math.pi) - degrees)))
                ne.append(abs(_fold(math.degrees(new) - degrees)))
            legacy_error.append((np.mean(le), np.max(le)))
            new_error.append((np.mean(ne), np.max(ne)))
        legacy_error, new_error = np.array(legacy_error), np.array(new_error)
        ax.fill_between(angles, 0, legacy_error[:, 1], color=LEGACY, alpha=0.15, lw=0)
        ax.plot(angles, legacy_error[:, 0], color=LEGACY, lw=1.2, label="legacy: mean (shaded: worst)")
        ax.fill_between(angles, 0, new_error[:, 1], color=NEW, alpha=0.2, lw=0)
        ax.plot(angles, new_error[:, 0], color=NEW, lw=1.2, label="new: mean (shaded: worst)")
        ax.set_xticks(range(0, 181, 30))
        ax.set_xlabel("the stream's true direction (degrees from east towards south)")
        ax.set_title(name)
        ax.legend(loc="upper left")
        stats[name] = dict(legacy_mean=round(float(legacy_error[:, 0].mean()), 2),
                           legacy_worst=round(float(legacy_error[:, 1].max()), 2),
                           new_mean=round(float(new_error[:, 0].mean()), 2),
                           new_worst=round(float(new_error[:, 1].max()), 2))
    axes[0].set_ylabel("error in the stream's direction (°)")
    return fig, dict(caption=(
        "Straight streams rasterised at every whole degree, each 12 times shifted sideways by up to 0.3 of a cell, "
        "with the sites' Gen_Dir_Dist of 5 cells. Legacy fitted the cells' rows against their columns, in cells, "
        "over a box missing its last row and column. The new direction is the principal axis of the cells in "
        "metres over the full box. Legacy's worst errors are near the columns (90°) on square cells, and everywhere "
        "off the axes on cells that aren't square, where it measured the angle in cells."), stats=stats)


def _segments(ax, xs, ys, angles, half, color, lw=1.4, label=None):
    """Short lines through map points in directions (radians from east towards south)."""
    dxs, dys = half * np.cos(angles), -half * np.sin(angles)
    lines = np.stack([np.stack([xs - dxs, ys - dys], -1), np.stack([xs + dxs, ys + dys], -1)], 1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(lines, colors=color, linewidths=lw, label=label))


def _detail_window(ctx, site, cells_needed=60, size_metres=1300.0):
    """A window around the densest stretch of stream cells shared by both codes."""
    legacy, new = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
    pairs = matched_cells(legacy, new)
    grid = ctx.grid(site)
    rows = np.array([n["row"] for _, n in pairs])
    cols = np.array([n["col"] for _, n in pairs])
    half_r, half_c = int(size_metres / 2 / grid.dy), int(size_metres / 2 / grid.dx)
    difference = np.array([abs(_fold(math.degrees(n["xs_angle"] - l["xs_angle"]))) for l, n in pairs])
    best, score = 0, -1
    for k in range(0, rows.size, 3):
        inside = (np.abs(rows - rows[k]) <= half_r) & (np.abs(cols - cols[k]) <= half_c)
        if inside.sum() >= cells_needed and difference[inside].mean() > score:
            best, score = k, difference[inside].mean()
    window = Window(rows[best] - half_r, rows[best] + half_r, cols[best] - half_c, cols[best] + half_c, grid.dx,
                    grid.dy)
    return window, pairs, grid


@figure("D2", "Cross-section directions on a real reach", "Stream direction and the angle search")
def direction_map(ctx):
    site = ctx.detail_sites[0]
    window, pairs, grid = _detail_window(ctx, site)
    legacy = ctx.capture("as_configured", "legacy", site)
    tables = (legacy["index_arrays"], legacy["distances"], legacy["fractions"])
    inside = [(l, n) for l, n in pairs if window.contains(n["row"], n["col"])]
    rows = np.array([n["row"] for _, n in inside])
    cols = np.array([n["col"] for _, n in inside])
    window = Window(rows.min() - 3, rows.max() + 3, cols.min() - 3, cols.max() + 3, grid.dx, grid.dy)
    xs, ys = window.xy(rows, cols)
    half = 1.4 * max(grid.dx, grid.dy)
    width = (window.c1 - window.c0 + 1) * grid.dx
    height = (window.r1 - window.r0 + 1) * grid.dy
    fig, axes = plt.subplots(1, 3, figsize=(13.5, min(13.5 / 3 * height / width + 0.9, 7.0)))
    from matplotlib.colors import ListedColormap
    streams = np.where(grid.streams > 0, 1.0, np.nan)
    for ax in axes:
        hillshade(ax, window, grid.dem, alpha=0.5)
        show_raster(ax, window, streams, cmap=ListedColormap(["#8fc3e6"]), alpha=0.75)
    # (a) the stream direction's cross sections, before the angle search
    legacy_initial = np.array([l["initial_xs_direction"] for l, _ in inside])
    new_initial = np.array([n["initial_direction"] - math.pi / 2 for _, n in inside])
    _segments(axes[0], xs, ys, legacy_initial, half, LEGACY, label="legacy")
    _segments(axes[0], xs, ys, new_initial, half, NEW, label="new")
    axes[0].set_title("(a) square to the stream's direction")
    # (b) after the angle search
    legacy_final = np.array([l["xs_angle"] for l, _ in inside])
    new_final = np.array([n["direction"] - math.pi / 2 for _, n in inside])
    _segments(axes[1], xs, ys, legacy_final, half, LEGACY, label="legacy")
    _segments(axes[1], xs, ys, new_final, half, NEW, label="new")
    axes[1].set_title("(b) after the angle search (XS_Angle)")
    # (c) where the first ordinates either side of the final cross sections really are
    from matplotlib.collections import LineCollection
    old_lines, new_lines = [], []
    for (l, n), x0, y0 in zip(inside, xs, ys):
        lr, lc = legacy_offsets(tables, legacy_index(l["xs_angle"]), 2)
        old_lines.append([(x0 - lc[1] * grid.dx, y0 + lr[1] * grid.dy), (x0 + lc[1] * grid.dx, y0 - lr[1] * grid.dy)])
        angle = n["direction"] - math.pi / 2
        s = n["spacing"]
        new_lines.append([(x0 - s * math.cos(angle), y0 + s * math.sin(angle)),
                          (x0 + s * math.cos(angle), y0 - s * math.sin(angle))])
    axes[2].add_collection(LineCollection(old_lines, colors=LEGACY, linewidths=1.4, label="legacy"))
    axes[2].add_collection(LineCollection(new_lines, colors=NEW, linewidths=1.4, label="new"))
    axes[2].set_title("(c) the ordinates either side as sampled")
    for ax in axes:
        ax.set_xlim(window.extent[0], window.extent[1])
        ax.set_ylim(window.extent[2], window.extent[3])
        ax.legend(loc="upper right", fontsize=7)
    d_initial = np.abs(_fold(np.degrees(new_initial - legacy_initial)))
    d_final = np.abs(_fold(np.degrees(new_final - legacy_final)))
    return fig, dict(caption=(
        f"Part of {ctx.site_label(site)} drawn to scale over a hillshade, with its stream cells. (a) Each cell's "
        "cross section square to the stream's direction as each code works it out: legacy fitted rows against "
        "columns in cells, the new code the principal axis in metres. (b) The cross sections after the angle search, "
        "which turns them up to 45° either way to where the water is narrowest: 0.5 m above the stream cell for "
        "legacy, 5 m for the new code (D6 and D7 show why). (c) The line through the "
        "ordinates either side of the stream cell as each code really sampled them: legacy's snapping to 6° and its "
        "oblique geometry turn its lines away from (b)."),
                 stats=dict(site=site, cells=len(inside), median_initial_difference_deg=rounded(np.median(d_initial), 1),
                            median_final_difference_deg=rounded(np.median(d_final), 1)))


@figure("D3", "How much the directions differ on the real sites", "Stream direction and the angle search")
def direction_differences(ctx):
    initial, final = [], []
    for site in ctx.sites:
        legacy, new = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
        if legacy is None or new is None:
            continue
        for l, n in matched_cells(legacy, new):
            initial.append(_fold(math.degrees(n["initial_direction"] - math.pi / 2 - l["initial_xs_direction"])))
            final.append(_fold(math.degrees(n["xs_angle"] - l["xs_angle"])))
    no_search = []
    for site in ctx.sites:
        m = ctx.merged_vdt("no_angle_search", site)
        if m is not None:
            no_search.append(_fold(np.degrees(m["XS_Angle_n"] - m["XS_Angle_l"])))
    no_search = np.concatenate(no_search) if no_search else np.array([])
    initial, final = np.abs(initial), np.abs(final)
    no_search = np.abs(no_search)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.arange(0, 90.5, 1.0)
    for values, color, label, style in ((initial, NEW, "square to the stream's direction", "-"),
                                        (no_search, ACCENT, "final, without the angle search (Degree_Manip 0)", "--"),
                                        (final, LEGACY, "final, with the sites' angle search (±45°)", "-")):
        if values.size:
            counts, _ = np.histogram(values, bins=bins)
            ax.step(bins[:-1], np.cumsum(counts) / values.size, where="post", color=color, lw=1.4, ls=style,
                    label=f"{label}: median {np.median(values):.1f}°")
    ax.set_xlabel("|new direction − legacy direction| (°)")
    ax.set_ylabel("share of stream cells")
    ax.set_xlim(0, 90)
    ax.legend(loc="lower right")
    return fig, dict(caption=(
        f"The difference between the two codes' cross-section directions at each stream cell of the "
        f"{len(ctx.sites)} sites, as cumulative shares. Square to the stream, they differ by {np.median(initial):.0f}° "
        "at the median (the metres-versus-cells fit, and the box). The searches then differ too: legacy's is 0.5 m "
        "deep and the new one 5 m, and each chooses among directions 2.5° apart, so a few degrees at the start can "
        f"become a quite different choice: {np.mean(final >= 45):.0%} of the cells end up 45° or more apart. Without "
        "the search the differences stay as they started."),
                 stats=dict(cells=int(initial.size), initial_median=rounded(np.median(initial), 2),
                            initial_p90=rounded(np.percentile(initial, 90), 2),
                            final_median=rounded(np.median(final), 2), final_p90=rounded(np.percentile(final, 90), 2),
                            no_search_median=rounded(np.median(no_search), 2) if no_search.size else None,
                            no_search_p90=rounded(np.percentile(no_search, 90), 2) if no_search.size else None))


@figure("D4", "The angle search at one stream cell", "Stream direction and the angle search")
def angle_search_example(ctx):
    from arc.hydraulics import top_widths
    from arc.xsection.orientation import LEGACY_TEST_DEPTH, TEST_DEPTH, _reach, angle_offsets
    from arc.xsection.sampling import sample_elevations
    site = ctx.detail_sites[0]
    legacy_capture, new_capture = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new",
                                                                                           site)
    pairs = matched_cells(legacy_capture, new_capture)
    grid = ctx.grid(site)
    # cells whose starting directions agree but whose searches chose differently
    scored = sorted(pairs, key=lambda p: -abs(_fold(math.degrees(p[1]["xs_angle"] - p[0]["xs_angle"])))
                    + 5 * abs(_fold(math.degrees(p[1]["initial_direction"] - math.pi / 2 - p[0]["initial_xs_direction"]))))
    configs = ctx.configs(site)
    dem32 = grid.dem.astype(np.float32)
    legacy = legacy_sampler(dem32, grid.dx, grid.dy)
    legacy.l_angles_to_test = angle_offsets(configs.degree_manip, configs.degree_interval)
    offsets = angle_offsets(configs.degree_manip, configs.degree_interval)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    stats = {}
    for ax, (l, n) in zip(axes, scored[:2]):
        r, c = n["row"], n["col"]
        # legacy: its candidates, each sampled in the nearest of its 6-degree directions, width rounded to the mm
        start = l["initial_xs_direction"]
        legacy.set_cross_section(r + PAD, c + PAD, legacy_index(start), start)
        old = []
        for adjustment in legacy.l_angles_to_test:
            angle = (start + adjustment) % math.pi
            legacy.set_cross_section(r + PAD, c + PAD, int(round(angle / LEGACY_STEP)), start)
            old.append((math.degrees(angle),
                        legacy.calculate_top_width_of_wse(legacy.get_thalweg() + LEGACY_TEST_DEPTH)))
        old = np.array(sorted(old))
        # new: each candidate sampled as it is, each side capped at the shortest candidate's reach
        direction = n["initial_direction"]
        new = []
        for offset in offsets:
            values, spacing = sample_elevations(grid.dem, r, c, direction + offset, LENGTH, grid.dx, grid.dy)
            center = values.size // 2
            left, right = top_widths(values, spacing, values[center] + TEST_DEPTH)
            new.append((math.degrees((direction + offset - math.pi / 2) % math.pi), left, right,
                        _reach(values, center, -1, spacing), _reach(values, center, 1, spacing)))
        new = np.array(new)
        left_limit, right_limit = new[:, 3].min(), new[:, 4].min()
        width = np.minimum(np.minimum(new[:, 1], new[:, 3]), left_limit) + np.minimum(np.minimum(new[:, 2], new[:, 4]),
                                                                                       right_limit)
        order = np.argsort(new[:, 0])
        ax.plot(old[:, 0], old[:, 1] / old[:, 1].min(), "x-", color=LEGACY, ms=4, lw=0.8,
                label=f"legacy's candidates, {LEGACY_TEST_DEPTH:g} m up")
        ax.plot(new[order, 0], width[order] / width.min(), "o-", color=NEW, ms=3, lw=0.8,
                label=f"new candidates, {TEST_DEPTH:g} m up")
        ax.axvline(math.degrees(l["xs_angle"]), color=LEGACY, lw=1.2, ls="--", label="legacy's choice")
        ax.axvline(math.degrees(n["xs_angle"]), color=NEW, lw=1.2, ls="--", label="new choice")
        ax.set_xlabel("cross-section direction (degrees from east towards south)")
        ax.set_ylabel("water's top width / its narrowest candidate's")
        ax.set_title(f"row {r}, column {c}")
        ax.legend(loc="best", fontsize=6.8)
        stats[f"{r},{c}"] = dict(legacy_choice=rounded(math.degrees(l["xs_angle"]), 1),
                                 new_choice=rounded(math.degrees(n["xs_angle"]), 1))
    return fig, dict(caption=(
        f"Two stream cells of {ctx.site_label(site)} where the searches started in nearly the same direction but "
        "chose differently. Each point is a candidate direction's water width, as a share of the narrowest "
        "candidate's: legacy's 0.5 m above the stream cell, the new code's 5 m. Legacy "
        "sampled each candidate in the nearest of its 6°-apart directions (so neighbouring candidates can share a "
        "width) with its oblique geometry; the new code samples each as it is. Widths of neighbouring directions "
        "differ by a cell or so of sampling noise, which decides the choice."), stats=stats)


def _exact_directions(dx: float, dy: float) -> np.ndarray:
    """The cross-section directions (degrees) whose ordinates all land on cell centres: along the rows and columns,
    and along the cells' diagonals."""
    diagonal = math.degrees(math.atan2(dy, dx))
    return np.array([0.0, diagonal, 90.0, 180.0 - diagonal, 180.0])


def _to_nearest(angles_deg: np.ndarray, exact: np.ndarray) -> np.ndarray:
    return np.min(np.abs(np.asarray(angles_deg)[:, None] % 180.0 - exact[None, :]), axis=1)


@figure("D5", "Where the angle search's directions end up", "Stream direction and the angle search")
def direction_clustering(ctx):
    site = ctx.detail_sites[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    bins = np.arange(0.0, 180.5, 1.0)
    grid = ctx.grid(site)
    exact = _exact_directions(grid.dx, grid.dy)
    runs = (("as_configured", "legacy", LEGACY, "legacy"), ("as_configured", "new", NEW, "new"),
            ("no_angle_search", "new", ACCENT, "new, without the search"))
    for config, code, color, label in runs:
        vdt = ctx.vdt(config, code, site)
        if vdt is not None:
            ax.hist(np.degrees(vdt["XS_Angle"]) % 180.0, bins=bins, histtype="step", color=color, lw=1.1,
                    label=label)
    for e in exact:
        ax.axvline(e, color=GROUND, lw=0.7, ls=":")
    ax.set_xticks(range(0, 181, 30))
    ax.set_xlabel("final cross-section direction (XS_Angle, degrees); dotted: rows, columns and cell diagonals")
    ax.set_ylabel("stream cells")
    ax.set_title(ctx.site_label(site))
    ax.legend(loc="upper right")
    ax = axes[1]
    stats = {}
    for config, code, color, label in runs:
        values = []
        for s in ctx.sites:
            vdt = ctx.vdt(config, code, s)
            if vdt is None:
                continue
            g = ctx.grid(s) if s in ctx.detail_sites else None
            dx, dy = (g.dx, g.dy) if g is not None else _cell_size(ctx, s)
            values.append(_to_nearest(np.degrees(vdt["XS_Angle"].to_numpy()), _exact_directions(dx, dy)))
        values = np.concatenate(values)
        counts, edges = np.histogram(values, bins=np.arange(0.0, 30.5, 0.5))
        ax.step(edges[:-1], counts / values.size, where="post", color=color, lw=1.2,
                label=f"{label}: {np.mean(values < 0.5):.0%} within 0.5°")
        stats[label] = dict(within_half_degree=rounded(np.mean(values < 0.5), 3), cells=int(values.size))
    ax.set_xlabel("degrees from the nearest row, column or cell diagonal")
    ax.set_ylabel("share of stream cells")
    ax.set_title(f"all {len(ctx.sites)} sites")
    ax.legend(loc="upper right")
    return fig, dict(caption=(
        "The directions the angle search settles on. Along the rows, the columns and the cells' diagonals every "
        "ordinate lands on a cell centre, so the sampled ground isn't averaged between cells. The narrowest water "
        "0.5 m deep, inside a channel a cell or two wide, was then often found exactly there, and a 0.5 m search's "
        "directions piled up on them (15% within 0.5°, D7), an artefact of the sampling rather than of the channels "
        "(legacy's own sampling has its own favoured directions). The new search is 5 m deep, where the water spans "
        "the floodplain, and its directions spread out nearly as far as without the search."), stats=stats)


def _cell_size(ctx, site):
    """A site's cell size in metres without reading its DEM's values."""
    capture = ctx.capture("as_configured", "new", site)
    if capture is not None:
        return capture["dx"], capture["dy"]
    grid = ctx.grid(site)
    return grid.dx, grid.dy


# --- The angle search's test depth ------------------------------------------------------------------------------------

TEST_DEPTHS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)


def _valley_errors(amplitude, noise, count=6):
    """Each rule's error from the valley's and the channel's own cross-section directions, on made-up valleys."""
    import vc_search
    from arc.xsection.orientation import angle_offsets, stream_direction
    dx, dy = SITE_CELL
    offsets = angle_offsets(90.0, 2.5)
    depths = np.array(TEST_DEPTHS)
    rng = np.random.default_rng(int(amplitude) + int(noise * 100))
    errors = {name: ([], []) for name in ["none", *TEST_DEPTHS]}
    for _ in range(count):
        angle = float(rng.uniform(0, math.pi))
        dem, streams, heading = vc_search.made_up_valley(angle, float(rng.uniform(150, 700)), amplitude, noise,
                                                         int(rng.integers(1 << 30)), dx, dy)
        size = dem.shape[0]
        rows, cols = np.nonzero(streams)
        inner = (np.abs(rows - size / 2) < size * 0.3) & (np.abs(cols - size / 2) < size * 0.3)
        for r, c in zip(rows[inner], cols[inner]):
            d0 = stream_direction(streams, int(r), int(c), 5, dx, dy)
            tried, widths, _ = vc_search.candidate_metrics(dem, int(r), int(c), d0, LENGTH, dx, dy, offsets, depths)
            for name in errors:
                k = 0 if name == "none" else vc_search.best_candidate(
                    np.ascontiguousarray(widths[:, [TEST_DEPTHS.index(name)]]))
                chosen = d0 + tried[k]
                errors[name][0].append(abs(_fold(math.degrees(chosen - angle))))
                errors[name][1].append(abs(_fold(math.degrees(chosen - heading[(int(r), int(c))]))))
    return {name: (np.array(v), np.array(ch)) for name, (v, ch) in errors.items()}


@figure("D6", "Which test depth: made-up valleys whose directions are known", "Stream direction and the angle search")
def test_depth_synthetic(ctx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
    stats = {}
    for ax, (amplitude, title) in zip(axes, ((0.0, "a straight channel down a straight valley"),
                                             (150.0, "a channel meandering 150 m either side of the valley's axis"))):
        for noise, style in ((0.25, "-"), (0.75, "--")):
            errors = _valley_errors(amplitude, noise)
            depths = np.array(TEST_DEPTHS)
            valley = [np.median(errors[d][0]) for d in TEST_DEPTHS]
            ax.plot(depths, valley, style, color=NEW, marker="o", ms=3.5, lw=1.4,
                    label=f"from the valley's own, DEM noise {noise:g} m")
            ax.axhline(np.median(errors["none"][0]), color=GROUND, ls=style, lw=0.9,
                       label=f"no search, from the valley's ({noise:g} m)")
            key = f"amplitude {amplitude:g}, noise {noise:g}"
            stats[key] = dict(no_search_valley=rounded(np.median(errors["none"][0]), 2),
                              no_search_channel=rounded(np.median(errors["none"][1]), 2),
                              **{f"{d:g}_valley": rounded(np.median(errors[d][0]), 2) for d in TEST_DEPTHS},
                              **{f"{d:g}_channel": rounded(np.median(errors[d][1]), 2) for d in TEST_DEPTHS})
            if amplitude > 0:
                channel = [np.median(errors[d][1]) for d in TEST_DEPTHS]
                ax.plot(depths, channel, style, color=ACCENT, marker="s", ms=3.5, lw=1.2,
                        label=f"from the channel's own ({noise:g} m)")
                ax.axhline(np.median(errors["none"][1]), color=ACCENT, ls=style, lw=0.9, alpha=0.8,
                           label=f"no search, from the channel's ({noise:g} m)")
        ax.axvline(0.5, color=LEGACY, lw=0.8, ls=":")
        ax.axvline(5.0, color=NEW, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(TEST_DEPTHS)
        ax.set_xticklabels([f"{d:g}" for d in TEST_DEPTHS])
        ax.set_xlabel("test depth above the stream cell (m); legacy's is 0.5 m, the new code's 5 m (dotted)")
        ax.set_title(title, fontsize=8.5)
    handles, labels = axes[1].get_legend_handles_labels()  # the meandering panel's lines include the straight one's
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7.5)
    axes[0].set_ylabel("median error in the chosen cross section's direction (°)")
    return fig, dict(caption=(
        "Made-up valleys at random angles on Cuyahoga's cells (six of each kind): a 36 m channel whose DEM water is "
        "1 m below a flat floodplain 300 to 1400 m wide, walls rising 30 m, and smoothed DEM noise. Each stream "
        "cell's search (±45° in 2.5° steps, as the sites) is run at each test depth, and its choice is compared with "
        "the cross section square to the valley and square to the channel there. At 0.5 m the water is inside a "
        "channel one or two cells wide, whose rasterised edges decide the width, and the search turns cross sections "
        "about 20° from both; not searching is better. From 2 to 3 m (more with more noise) the water spans the "
        "floodplain and the search finds the valley. Where the channel meanders, no search follows the channel, and a "
        "deep search the valley."), stats=stats)


def _search_table(ctx, site):
    """Every sampled stream cell's candidates' widths and areas at the test depths, and at its own bankfull and flood
    depths (vc_search), with what the metrics need."""
    key = ("search_table", site)
    if key in ctx._cache:
        return ctx._cache[key]
    import vc_search
    from arc.xsection.orientation import angle_offsets
    capture = ctx.capture("as_configured", "new", site)
    vdt = ctx.vdt("no_bathymetry", "new", site)
    if capture is None or vdt is None:
        return None
    grid = ctx.grid(site)
    configs = ctx.configs(site)
    offsets = angle_offsets(configs.degree_manip, configs.degree_interval)
    top = max(int(c[4:]) for c in vdt.columns if c.startswith("wse_"))
    flood = {(int(r), int(c)): (w - e, t) for r, c, w, e, t in
             zip(vdt.Row, vdt.Col, vdt[f"wse_{top}"], vdt.Elev, vdt[f"t_{top}"])}
    cells = [c for c in capture["cells"] if c is not None]
    n, k, m = len(cells), offsets.size, vc_search.DEPTHS.size + 2
    table = dict(widths=np.full((n, k, m), np.nan), areas=np.full((n, k, m), np.nan), tried=np.full((n, k), np.nan),
                 rows=np.empty(n, np.int64), cols=np.empty(n, np.int64), reaches=np.empty(n, np.int64),
                 direction=np.empty(n), flood_width=np.empty(n), final=np.empty(n), dx=grid.dx, dy=grid.dy)
    for i, cell in enumerate(cells):
        r, c = cell["center"]
        table["rows"][i], table["cols"][i], table["reaches"][i] = r, c, cell["reach"]
        table["direction"][i], table["final"][i] = cell["initial_direction"], cell["direction"]
        depth, width = flood.get((cell["row"], cell["col"]), (math.nan, math.nan))
        table["flood_width"][i] = width
        bankfull = cell["target_depth"] if math.isfinite(cell["target_depth"]) else 1.0
        depths = np.r_[vc_search.DEPTHS, bankfull, depth if math.isfinite(depth) and depth > 0 else 3.0]
        t, w, a = vc_search.candidate_metrics(grid.dem, int(r), int(c), float(cell["initial_direction"]),
                                              float(configs.x_section_dist), grid.dx, grid.dy, offsets, depths)
        table["tried"][i, :t.size], table["widths"][i, :t.size], table["areas"][i, :t.size] = t, w, a
    ctx._cache[key] = table
    return table


def _chosen(table, columns, kind="widths"):
    """Each cell's chosen candidate for a rule: None for no search, else the columns (depths) it's scored at."""
    if columns is None:
        return np.zeros(table["tried"].shape[0], dtype=int)
    metric = table[kind][:, :, columns].copy()
    metric[~np.isfinite(table["tried"])] = np.inf
    if len(columns) == 1:
        return np.argmin(metric[:, :, 0], axis=1)
    lows = np.maximum(np.min(metric, axis=1, keepdims=True), 1e-12)
    return np.argmin(np.sum(metric / lows, axis=2), axis=1)


def _rule_metrics(tables, columns, kind="widths"):
    """A rule's metrics pooled over the sites: on the grid's directions, neighbours' jitter and crossings, and how
    much wider (or bigger) the chosen cross section is than the narrowest candidate at the flood depth."""
    import vc_search
    on_grid, jitter, crossing, excess_width, excess_area = [], [], [], [], []
    for table in tables:
        pick = _chosen(table, columns, kind)
        n = pick.size
        offset = table["tried"][np.arange(n), pick]
        angle = np.degrees(table["direction"] + offset - math.pi / 2) % 180.0
        exact = _exact_directions(table["dx"], table["dy"])
        on_grid.append(np.min(np.abs(angle[:, None] - exact[None, :]), axis=1) < 0.5)
        for store, kind_ in ((excess_width, "widths"), (excess_area, "areas")):
            m = table[kind_][:, :, vc_search.FLOOD].copy()
            m[~np.isfinite(table["tried"])] = np.inf
            low, chosen = np.min(m, axis=1), m[np.arange(n), pick]
            ok = np.isfinite(chosen) & (low > 0)
            store.append(chosen[ok] / low[ok] - 1.0)
        x, y = table["cols"] * table["dx"], table["rows"] * table["dy"]
        phi = np.radians(angle)
        ux, uy = np.cos(phi), np.sin(phi)
        half = np.where(np.isfinite(table["flood_width"]), 0.5 * table["flood_width"], 250.0)
        for reach in np.unique(table["reaches"]):
            ks = np.flatnonzero(table["reaches"] == reach)
            rr, cc = table["rows"][ks], table["cols"][ks]
            for a in range(ks.size):
                near = np.flatnonzero((np.abs(rr - rr[a]) <= 3) & (np.abs(cc - cc[a]) <= 3))
                near = near[near > a]
                for b in near:
                    i, j = ks[a], ks[b]
                    if abs(rr[a] - rr[b]) <= 1 and abs(cc[a] - cc[b]) <= 1:
                        jitter.append(abs(_fold(angle[i] - angle[j])))
                    det = -ux[i] * uy[j] + uy[i] * ux[j]
                    if abs(det) < 1e-9:
                        crossing.append(False)
                        continue
                    ex, ey = x[j] - x[i], y[j] - y[i]
                    t = (-ex * uy[j] + ey * ux[j]) / det
                    s = (ux[i] * ey - uy[i] * ex) / det
                    reach_ = min(half[i], half[j])
                    crossing.append(abs(t) <= reach_ and abs(s) <= reach_)
    return dict(on_grid=float(np.mean(np.concatenate(on_grid))), jitter=float(np.median(jitter)),
                crossing=float(np.mean(crossing)), excess_width=float(np.median(np.concatenate(excess_width))),
                excess_area=float(np.median(np.concatenate(excess_area))))


@figure("D7", "Which test depth: the real sites", "Stream direction and the angle search")
def test_depth_real(ctx):
    import vc_search
    from arc.xsection.orientation import TEST_DEPTH
    tables = [t for t in (_search_table(ctx, s) for s in ctx.sites) if t is not None]
    own = [list(vc_search.DEPTHS).index(TEST_DEPTH)]  # the pipeline's own depth, which the runs used
    same = sum(int(np.sum(np.abs(t["direction"] + t["tried"][np.arange(t["tried"].shape[0]),
                                                                _chosen(t, own)] - t["final"]) < 1e-12))
               for t in tables)
    cells = sum(t["tried"].shape[0] for t in tables)
    columns = [list(vc_search.DEPTHS).index(d) for d in TEST_DEPTHS]
    rules = {d: _rule_metrics(tables, [k]) for d, k in zip(TEST_DEPTHS, columns)}
    none = _rule_metrics(tables, None)
    extra = {"the cell's flood depth": _rule_metrics(tables, [vc_search.FLOOD]),
             "widths at 0.5 to 10 m together": _rule_metrics(tables, list(range(2, vc_search.DEPTHS.size))),
             "flow area at 5 m": _rule_metrics(tables, [list(vc_search.DEPTHS).index(5.0)], "areas")}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0), sharex=True)
    depths = np.array(TEST_DEPTHS)
    panels = (("crossing", "neighbouring cross sections that cross\nwithin the flood's reach (%)", 100),
              ("jitter", "median angle between neighbours (°)", 1),
              ("on_grid", "on a row, column or cell diagonal (%)", 100),
              ("excess_width", "wider than the narrowest candidate\nat the flood depth, median (%)", 100))
    markers = ("s", "D", "^")
    for ax, (key, label, scale) in zip(axes.flat, panels):
        ax.plot(depths, [rules[d][key] * scale for d in TEST_DEPTHS], "o-", color=NEW, ms=4, lw=1.4,
                label="the search at one test depth")
        ax.axhline(none[key] * scale, color=GROUND, lw=1.0, ls="--", label="no search")
        for (name, values), marker in zip(extra.items(), markers):
            ax.plot([12.5], [values[key] * scale], marker, color=ACCENT, ms=5)
            ax.annotate(name, (12.5, values[key] * scale), xytext=(4, 0), textcoords="offset points", fontsize=6,
                        va="center", color=ACCENT)
        ax.axvline(0.5, color=LEGACY, lw=0.8, ls=":")
        ax.axvline(5.0, color=NEW, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(TEST_DEPTHS)
        ax.set_xticklabels([f"{d:g}" for d in TEST_DEPTHS])
        ax.set_xlim(0.2, 40)
        ax.set_ylabel(label, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("test depth above the stream cell (m); legacy's is 0.5 m, the new code's 5 m")
    axes[0, 0].legend(loc="lower left", fontsize=6.5)
    stats = dict(cells=cells, reproduces_pipeline=same, no_search={k: rounded(v, 4) for k, v in none.items()},
                 **{f"{d:g} m": {k: rounded(v, 4) for k, v in rules[d].items()} for d in TEST_DEPTHS},
                 **{name: {k: rounded(v, 4) for k, v in values.items()} for name, values in extra.items()})
    return fig, dict(caption=(
        f"The same measures on the {len(tables)} sites' {cells:,} sampled stream cells, for the search at each test "
        f"depth and three combinations (at the right). At 5 m the table reproduces the pipeline's own choice at "
        f"{same:,} of them. Neighbours are cross sections of the same reach within three cells; a crossing is two that "
        "cross within the flood's half top width of both stream cells, and the flood depth is each cell's top water "
        "surface above its stream cell (without bathymetry). Legacy's 0.5 m more than doubles the crossings and "
        "multiplies the jitter by six, and at the flood depth its cross sections are wider than not searching "
        "leaves them. Deeper test depths are better on every measure, which is why the new code's is 5 m (since "
        "2026-09-26), but the search at any depth crosses more than not searching."), stats=stats)


def _valley(channel_col, size=41, cell=10.0):
    return np.tile(100.0 + 0.2 * cell * np.abs(np.arange(size) - channel_col), (size, 1))


@figure("L1", "Low_Spot_Range: how far out the low spot is looked for", "Low spot")
def low_spot_range(ctx):
    from arc.cross_section import CrossSection
    from arc.xsection.low_spot import low_spot_cell
    cell = 10.0
    dem = _valley(25)
    params = {"d_x_section_distance": 200.0, "dx": cell, "dy": cell, "d_degree_manipulation": 0.0,
              "d_degree_interval": 0.0, "i_boundary_number": 0, "nrows": 41, "ncols": 41,
              "b_FindBanksBasedOnLandCover": False, "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": 0.2,
              "b_bathy_use_banks": False, "s_output_bathymetry_path": ""}

    def legacy_move(low_spot_range):
        old = CrossSection(cell, cell, dem, np.zeros(dem.shape, dtype=np.uint8), None, params)
        old.associate_with_precomputed_index_arrays(*CrossSection.create_cross_section_ordinates(params))
        old.set_cross_section(20, 20, 0, 0.0)
        old.adjust_cross_section_to_lowest_point(low_spot_range)
        return int(old.get_row_col()[1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    stats = {}
    columns = np.arange(41) - 20
    for ax, n in zip(axes, (5, 1)):
        ax.plot(columns * cell, dem[20], color=GROUND, lw=1.5, marker="o", ms=3, label="ground at the ordinates")
        ax.axvspan(-(n - 1) * cell - 3, (n - 1) * cell + 3, color=LEGACY, alpha=0.12, lw=0,
                   label=f"legacy looked {n - 1} ordinate{'s' if n - 1 != 1 else ''} out")
        ax.axvspan(-n * cell - 3, n * cell + 3, color=NEW, alpha=0.10, lw=0,
                   label=f"new looks {n} ordinate{'s' if n != 1 else ''} out")
        old = legacy_move(n) - 20
        new = low_spot_cell(dem, 20, 20, math.pi / 2, 200.0, cell, cell, n)[1] - 20
        ax.plot([0], [dem[20, 20]], "k^", ms=8, label="stream cell")
        ax.plot([old * cell], [dem[20, 20 + old] + 0.6], "v", color=LEGACY, ms=9, label=f"legacy moves to {old:+d}")
        ax.plot([new * cell], [dem[20, 20 + new] + 0.6], "v", color=NEW, ms=9, label=f"new moves to {new:+d}")
        ax.set_xlim(-100, 100)
        ax.set_xlabel("metres from the stream cell along the cross section")
        ax.set_title(f"Low_Spot_Range = {n}, the channel 5 cells away")
        ax.legend(loc="upper left", fontsize=6.8)
        stats[f"range_{n}"] = dict(legacy=old, new=new)
    axes[0].set_ylabel("elevation (m)")
    return fig, dict(caption=(
        "A V-shaped valley whose low point is 5 cells from the stream cell, on 10 m cells. Legacy counted the stream "
        "cell as the first of its Low_Spot_Range ordinates, so it looked one ordinate less far than asked, and with "
        "a range of 1 didn't look at all. The new code looks the full range either side."), stats=stats)


@figure("L2", "The low spot on a real site", "Low spot")
def low_spot_real(ctx):
    from arc.Automated_Rating_Curve_Generator import get_stream_direction_information
    from arc.xsection.low_spot import low_spot_cell
    from arc.xsection.orientation import stream_direction
    site = ctx.detail_sites[0]
    grid = ctx.grid(site)
    configs = ctx.configs(site)
    n = 3  # the sites don't use the low spot; this is what it would do
    dem32 = grid.dem.astype(np.float32)
    legacy = legacy_sampler(dem32, grid.dx, grid.dy)
    padded_streams = np.pad(grid.streams, PAD)
    rows, cols = np.nonzero(grid.streams > 0)
    moves = []
    same_axis = {"agree": 0, "cells": 0}
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)
        _, xs_direction = get_stream_direction_information(r + PAD, c + PAD, padded_streams, configs.gen_dir_dist)
        legacy.set_cross_section(r + PAD, c + PAD, legacy_index(xs_direction), xs_direction)
        legacy.adjust_cross_section_to_lowest_point(n)
        lr, lc = (int(v) - PAD for v in legacy.get_row_col())
        direction = stream_direction(grid.streams, r, c, configs.gen_dir_dist, grid.dx, grid.dy)
        nr, nc = low_spot_cell(grid.dem, r, c, direction, LENGTH, grid.dx, grid.dy, n)
        moves.append((r, c, lr, lc, nr, nc))
        # along rows and columns, legacy with a range of n + 1 should move exactly as the new code with n
        for j, xs in ((0, 0.0), (15, math.pi / 2)):
            legacy.set_cross_section(r + PAD, c + PAD, j, j * LEGACY_STEP)
            legacy.adjust_cross_section_to_lowest_point(n + 1)
            old = tuple(int(v) - PAD for v in legacy.get_row_col())
            same_axis["cells"] += 1
            same_axis["agree"] += int(old == low_spot_cell(grid.dem, r, c, xs + math.pi / 2, LENGTH, grid.dx,
                                                            grid.dy, n))
    moves = np.array(moves)
    legacy_distance = np.hypot((moves[:, 2] - moves[:, 0]) * grid.dy, (moves[:, 3] - moves[:, 1]) * grid.dx)
    new_distance = np.hypot((moves[:, 4] - moves[:, 0]) * grid.dy, (moves[:, 5] - moves[:, 1]) * grid.dx)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    moved = (legacy_distance > 0) | (new_distance > 0)
    half_r, half_c = int(450 / grid.dy), int(450 / grid.dx)
    best = max(np.flatnonzero(moved), key=lambda k: np.sum(moved & (np.abs(moves[:, 0] - moves[k, 0]) <= half_r)
                                                            & (np.abs(moves[:, 1] - moves[k, 1]) <= half_c)))
    window = Window(moves[best, 0] - half_r, moves[best, 0] + half_r, moves[best, 1] - half_c,
                    moves[best, 1] + half_c, grid.dx, grid.dy)
    ax = axes[0]
    hillshade(ax, window, grid.dem, alpha=0.6)
    inside = window.contains(moves[:, 0], moves[:, 1])
    x0, y0 = window.xy(moves[inside, 0], moves[inside, 1])
    ax.plot(x0, y0, "s", color="#333333", ms=2.5, alpha=0.6, lw=0, label="stream cells")
    for (a, b), color, label in (((2, 3), LEGACY, "legacy's move"), ((4, 5), NEW, "new move")):
        x1, y1 = window.xy(moves[inside, a], moves[inside, b])
        moved = (x1 != x0) | (y1 != y0)
        ax.quiver(x0[moved], y0[moved], (x1 - x0)[moved], (y1 - y0)[moved], angles="xy", scale_units="xy", scale=1,
                  color=color, width=0.006, headwidth=4, label=label, alpha=0.85)
    ax.set_xlim(window.extent[0], window.extent[1])
    ax.set_ylim(window.extent[2], window.extent[3])
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"moves with Low_Spot_Range = {n}")
    ax = axes[1]
    bins = np.arange(0, max(legacy_distance.max(), new_distance.max()) + 20, 10)
    ax.hist([legacy_distance, new_distance], bins=bins, color=[LEGACY, NEW], label=["legacy", "new"])
    ax.set_yscale("log")
    ax.set_xlabel("metres the cross section moved")
    ax.set_ylabel("stream cells")
    note(ax, f"moved: legacy {np.mean(legacy_distance > 0):.0%}, new {np.mean(new_distance > 0):.0%}\n"
             f"along rows and columns, legacy with {n + 1}\nmoves as the new code with {n}: "
             f"{same_axis['agree']} of {same_axis['cells']}", loc="upper right")
    ax.legend(loc="center right")
    return fig, dict(caption=(
        f"Every stream cell of {ctx.site_label(site)} with a Low_Spot_Range of {n} (the sites don't use the low spot; "
        "this shows what it does). The arrows go from the stream cell to where each code moved its cross section. "
        "The new code looks one ordinate further, finds the nearest cell to the low ordinate, and on oblique cross "
        "sections looks where the ordinates really are. The sanity check: on cross sections along rows and "
        f"columns, legacy given {n + 1} moves every cell exactly as the new code given {n}."),
                 stats=dict(site=site, cells=int(moves.shape[0]), legacy_moved=int((legacy_distance > 0).sum()),
                            new_moved=int((new_distance > 0).sum()), axis_agree=same_axis["agree"],
                            axis_cells=same_axis["cells"]))


def _meander(amplitude, wavelength, dx, dy, size=121):
    """Stream cells along y = amplitude * sin(2 pi x / wavelength), and each one's distance along the curve."""
    center = size // 2
    t = np.linspace(-center * dx * 1.2, center * dx * 1.2, 40000)
    x, y = t, amplitude * np.sin(2 * math.pi * t / wavelength)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    cols = np.rint(center + x / dx).astype(int)
    rows = np.rint(center + y / dy).astype(int)
    keep = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    cells = {}
    for r, c, s in zip(rows[keep], cols[keep], arc[keep]):
        cells.setdefault((r, c), s)
    return cells, center


def _slopes_at_center(cells, center, dx, dy, slope, distance=19, size=121):
    """Legacy's and the new local average slope at the centre cell of a reach whose ground falls at `slope` per
    metre along its cells' stations."""
    from arc.Automated_Rating_Curve_Generator import get_local_average_stream_slope_information
    from arc.xsection.slope import local_average_slopes
    from arc.xsection.stream_path import along_stream_stations
    streams = np.zeros((size, size), dtype=np.int64)
    dem = np.full((size, size), 500.0)
    rows = np.array([rc[0] for rc in cells])
    cols = np.array([rc[1] for rc in cells])
    arc = np.array(list(cells.values()))
    streams[rows, cols] = 1
    dem[rows, cols] = 500.0 - slope * arc
    k = int(np.argmin(np.hypot(rows - center, cols - center)))
    r0, c0 = int(rows[k]), int(cols[k])
    old = get_local_average_stream_slope_information(r0, c0, dem, streams, dx, dy, distance)
    stations = along_stream_stations(rows, cols, dx, dy)
    new = local_average_slopes(dem[rows, cols], rows, cols, stations, dx, dy, distance)[k]
    return old, new


@figure("SL1", "Stream slopes on straight and winding streams", "Stream slopes")
def slopes_synthetic(ctx):
    dx = dy = 30.0
    slope = 0.002
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    angles = np.arange(0.0, 90.01, 1.5)
    old, new = [], []
    for degrees in angles:
        streams, center = rasterise_line(math.radians(degrees), dx, dy, size=121)
        rows, cols = np.nonzero(streams)
        # distance along the straight line from one end
        along = (cols - center) * dx * math.cos(math.radians(degrees)) + (rows - center) * dy * math.sin(
            math.radians(degrees))
        cells = {(int(r), int(c)): float(a - along.min()) for r, c, a in zip(rows, cols, along)}
        o, n = _slopes_at_center(cells, center, dx, dy, slope)
        old.append(o / slope)
        new.append(n / slope)
    ax.plot(angles, old, color=LEGACY, lw=1.3, label="legacy (straight-line distance)")
    ax.plot(angles, new, color=NEW, lw=1.3, label="new (along the stream's cells)")
    ax.axhline(1.0, color=GROUND, lw=0.8)
    ax.set_xticks(range(0, 91, 15))
    ax.set_xlabel("direction of a straight stream (degrees from the rows)")
    ax.set_ylabel("local average slope / true slope")
    ax.set_title("straight streams, square cells")
    ax.legend(loc="lower left")
    ax = axes[1]
    wavelength = 900.0
    sinuosity, old, new = [], [], []
    for amplitude in np.linspace(0.0, 300.0, 16):
        cells, center = _meander(amplitude, wavelength, dx, dy)
        t = np.linspace(0, wavelength, 20001)
        length = np.sum(np.hypot(np.diff(t), np.diff(amplitude * np.sin(2 * math.pi * t / wavelength))))
        sinuosity.append(length / wavelength)
        o, n = _slopes_at_center(cells, center, dx, dy, slope)
        old.append(o / slope)
        new.append(n / slope)
    ax.plot(sinuosity, old, "o-", color=LEGACY, ms=3, lw=1.2, label="legacy")
    ax.plot(sinuosity, new, "o-", color=NEW, ms=3, lw=1.2, label="new")
    ax.plot(sinuosity, sinuosity, color=GROUND, lw=0.8, ls=":", label="the valley's slope (true × sinuosity)")
    ax.axhline(1.0, color=GROUND, lw=0.8)
    ax.set_xlabel("sinuosity (stream length / valley length)")
    ax.set_ylabel("local average slope / true slope")
    ax.set_title(f"meanders {wavelength:g} m long, square 30 m cells")
    ax.legend(loc="upper left")
    return fig, dict(caption=(
        "The local average slope (Gen_Slope_Dist 19 cells) at the middle of a stream whose ground falls 0.2% per "
        "metre along it. Left: straight streams. Legacy's straight-line distances are exact; the new distances along "
        "the stream's cells are the same along rows, columns and diagonals, but up to 8% longer in between (a path of "
        "whole cells zigzags), so its slope is up to 7.6% low there. Right: meanders. Legacy's straight lines cut "
        "across the bends, so its slope grows with the sinuosity towards the valley's; the new code measures along "
        "the stream and stays near the true slope."),
                 stats=dict(straight_new_min=rounded(min(new), 3)))


@figure("SL2", "Stream slopes on the real sites", "Stream slopes")
def slopes_real(ctx):
    old, new = [], []
    for site in ctx.sites:
        legacy, current = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
        if legacy is None or current is None:
            continue
        for l, n in matched_cells(legacy, current):
            old.append(l["slope"])
            new.append(n["slope"])
    old, new = np.array(old), np.array(new)
    ratio = new / old
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.loglog(old, new, ".", ms=1.5, color=NEW, alpha=0.3)
    lims = [min(old.min(), new.min()), max(old.max(), new.max())]
    ax.plot(lims, lims, color=GROUND, lw=0.8)
    ax.set_xlabel("legacy's slope")
    ax.set_ylabel("new slope")
    ax.set_title(f"each stream cell of the {len(ctx.sites)} sites (local_average_corrected)")
    ax = axes[1]
    bins = np.linspace(0.4, 1.6, 61)
    ax.hist(np.clip(ratio, bins[0], bins[-1]), bins=bins, color=NEW, alpha=0.8)
    ax.axvline(1.0, color=GROUND, lw=0.8)
    ax.set_xlabel("new slope / legacy's slope")
    ax.set_ylabel("stream cells")
    q = np.percentile(ratio, [10, 50, 90])
    note(ax, f"{ratio.size} cells\nmedian {q[1]:.3f}, p10 {q[0]:.3f}, p90 {q[2]:.3f}\n"
             f"exactly equal: {np.mean(ratio == 1.0):.1%}\nlower: {np.mean(ratio < 1.0):.0%}", loc="upper left")
    return fig, dict(caption=(
        "The slope each code gave each stream cell (local_average_corrected: the local average over 19 cells, limited "
        "to the reach's 20th to 50th percentiles). The new slopes are mostly a little lower, as the distances along "
        "the stream are longer than the straight lines wherever it bends; where both are held at the same reach "
        "percentile they can be equal."),
                 stats=dict(cells=int(ratio.size), median_ratio=rounded(q[1], 4), p10=rounded(q[0], 4),
                            p90=rounded(q[2], 4), equal=rounded(np.mean(ratio == 1.0), 4),
                            lower=rounded(np.mean(ratio < 1.0), 4)))


@figure("SL3", "Distances along a real stream, and the local slope's box", "Stream slopes")
def stations_map(ctx):
    from arc.xsection.stream_path import along_stream_stations, stream_cells_by_reach
    site = ctx.detail_sites[0]
    grid = ctx.grid(site)
    reaches = stream_cells_by_reach(grid.streams)
    reach = max(reaches, key=lambda k: reaches[k][0].size)
    rows, cols = reaches[reach]
    stations = along_stream_stations(rows, cols, grid.dx, grid.dy)
    distance = ctx.configs(site).gen_slope_dist
    # the bendiest stretch: where the stations between cells a box apart most exceed the straight lines
    k = int(np.argmax([np.max(np.abs(stations[(np.abs(rows - r) <= distance) & (np.abs(cols - c) <= distance)] - s)
                              / np.maximum(np.hypot((rows[(np.abs(rows - r) <= distance) & (np.abs(cols - c) <= distance)] - r) * grid.dy,
                                                    (cols[(np.abs(rows - r) <= distance) & (np.abs(cols - c) <= distance)] - c) * grid.dx), 1.0))
                       for r, c, s in zip(rows[::5], cols[::5], stations[::5])])) * 5
    r0, c0 = int(rows[k]), int(cols[k])
    window = Window(r0 - distance - 6, r0 + distance + 6, c0 - distance - 6, c0 + distance + 6, grid.dx, grid.dy)
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    hillshade(ax, window, grid.dem, alpha=0.5)
    inside = window.contains(rows, cols)
    along = np.full(grid.streams.shape, np.nan)
    along[rows, cols] = np.abs(stations - stations[k])
    image = show_raster(ax, window, along, cmap="viridis_r")
    fig.colorbar(image, ax=ax, shrink=0.7, label="metres along the stream from the starred cell, either way")
    from matplotlib.patches import Rectangle
    xk, yk = window.xy(r0, c0)
    ax.add_patch(Rectangle((xk - (distance + 0.5) * grid.dx, yk - (distance + 0.5) * grid.dy),
                           (2 * distance + 1) * grid.dx, (2 * distance + 1) * grid.dy, fill=False, ec=NEW, lw=1.5,
                           label=f"new box: {distance} cells every way"))
    ax.add_patch(Rectangle((xk - (distance + 0.5) * grid.dx, yk - (distance - 0.5) * grid.dy),
                           (2 * distance) * grid.dx, (2 * distance) * grid.dy, fill=False, ec=LEGACY, lw=1.2,
                           ls="--", label=f"legacy's box: {distance} before, {distance - 1} after"))
    ax.plot([xk], [yk], "k*", ms=12)
    # the pair whose along-stream distance most exceeds its straight line
    box = inside & (np.abs(rows - r0) <= distance) & (np.abs(cols - c0) <= distance)
    straight = np.hypot((rows - r0) * grid.dy, (cols - c0) * grid.dx)
    excess = np.where(box, np.abs(stations - stations[k]) / np.maximum(straight, 1.0), 0)
    j = int(np.argmax(excess))
    xj, yj = window.xy(rows[j], cols[j])
    ax.plot([xk, xj], [yk, yj], color="k", lw=1.2, ls=":")
    ax.text((xk + xj) / 2, (yk + yj) / 2,
            f"straight {straight[j]:.0f} m\nalong the stream {abs(stations[j] - stations[k]):.0f} m", fontsize=7.5,
            bbox=dict(fc="white", ec="#aaaaaa", alpha=0.9))
    ax.set_xlim(window.extent[0], window.extent[1])
    ax.set_ylim(window.extent[2], window.extent[3])
    ax.legend(loc="upper right", fontsize=7)
    return fig, dict(caption=(
        f"A bend of the longest reach of {ctx.site_label(site)}, drawn to scale, with its cells coloured by their "
        "distance along the stream from the starred cell (the shortest path through the reach's cells). The local "
        f"average slope at the starred cell uses the reach's cells within Gen_Slope_Dist ({distance}) cells along "
        "each axis: the box is a rectangle on the ground, since the cells aren't square, and legacy's missed its last "
        "row and column. Across a bend the drop between two cells is spread over the stream's length between them, "
        "not the straight line."),
                 stats=dict(site=site, reach=int(reach), straight_m=rounded(straight[j], 1),
                            along_m=rounded(abs(stations[j] - stations[k]), 1)))
