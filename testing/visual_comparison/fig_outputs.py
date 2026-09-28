"""The outputs as a whole: the VDT databases compared, what each difference contributes, the speed, and sanity
checks of what both codes should share."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from vc_plot import (ACCENT, GROUND, LEGACY, NEW, PINK, Window, cdf, figure, hillshade, increments, note, rounded,
                     show_raster)

ABLATION = (("as_configured", "as configured", LEGACY),
            ("no_bathymetry", "without bathymetry", ACCENT),
            ("no_angle_search", "… and without the angle search", PINK),
            ("constant_n", "… and n not varying with depth", NEW),
            ("uniform_n", "… and every class the same n", GROUND))


def _pooled_differences(ctx, config):
    """Pooled differences over the shared VDT rows of every site: |Δwse| in metres and |Δ|/legacy for t, v and q,
    increment by increment."""
    out = {key: [] for key in ("wse", "t", "v", "q")}
    rows = [0, 0, 0]
    for site in ctx.sites:
        legacy, new = ctx.vdt(config, "legacy", site), ctx.vdt(config, "new", site)
        if legacy is None or new is None:
            continue
        m = ctx.merged_vdt(config, site)
        rows[0] += len(legacy)
        rows[1] += len(new)
        rows[2] += len(m)
        count = increments(legacy)
        for key in out:
            old = np.concatenate([m[f"{key}_{i}_l"].to_numpy() for i in range(1, count + 1)])
            cur = np.concatenate([m[f"{key}_{i}_n"].to_numpy() for i in range(1, count + 1)])
            out[key].append(np.abs(cur - old) if key == "wse" else np.abs(cur - old) / np.maximum(np.abs(old), 1e-6))
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}, rows


@figure("O1", "Where the rating curves' differences come from", "The outputs")
def ablation(ctx):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    stats = {}
    labels = (("wse", "|Δ water surface| (m)", (1e-4, 10)), ("t", "|Δ top width| / legacy's", (1e-4, 10)),
              ("v", "|Δ velocity| / legacy's", (1e-4, 10)), ("q", "|Δ discharge| / legacy's", (1e-4, 10)))
    for config, label, color in ABLATION:
        pooled, rows = _pooled_differences(ctx, config)
        if not pooled["wse"].size:
            continue
        stats[config] = dict(rows_legacy=rows[0], rows_new=rows[1], rows_shared=rows[2])
        for ax, (key, name, lims) in zip(axes, labels):
            values = pooled[key]
            cdf(ax, np.maximum(values, lims[0]), color, f"{label}: median {np.median(values):.3g}", lw=1.4)
            stats[config][key] = dict(median=rounded(np.median(values), 4), p90=rounded(np.percentile(values, 90), 4))
    for ax, (key, name, lims) in zip(axes, labels):
        ax.set_xscale("log")
        ax.set_xlim(*lims)
        ax.set_xlabel(name)
        ax.legend(loc="upper left", fontsize=6)
    axes[0].set_ylabel("share of increments (shared cells)")
    return fig, dict(caption=(
        f"How far the new VDT databases are from legacy's on the {len(ctx.sites)} sites, pooled over the increments "
        "of the cells both wrote, as each difference is taken away in turn: as the sites are configured; without "
        "bathymetry; also without the angle search (Degree_Manip 0); also with n not varying with depth; and also "
        "with every land cover class the same n (where both codes take the same segment roughness). What's left in "
        "the last is the sampling (legacy's oblique geometry and directions), the slopes, the banks the hydraulics "
        "divide at, and the exact top."), stats=stats)


def _wse_at(q_values, wse_values, q):
    """A rating curve's water surface for a discharge, by linear interpolation; NaN outside it."""
    q_values, wse_values = np.asarray(q_values, float), np.asarray(wse_values, float)
    keep = np.isfinite(q_values) & np.isfinite(wse_values) & (wse_values > 0)
    q_values, wse_values = q_values[keep], wse_values[keep]
    if q_values.size < 2 or not q_values[0] <= q <= q_values[-1]:
        return math.nan
    q_values = np.maximum.accumulate(q_values)
    unique = np.r_[True, np.diff(q_values) > 0]
    return float(np.interp(q, q_values[unique], wse_values[unique]))


def _same_discharge(ctx, config, site, share=0.5):
    """(row, col, legacy's water surface, the new one) for each shared cell at share of its maximum flow."""
    m = ctx.merged_vdt(config, site)
    capture = ctx.capture("as_configured", "new", site)
    qmax = {(c["comid"], *c["center"]): c["qmax"] for c in capture["cells"] if c is not None}
    count = increments(m, "_l")
    out = []
    for row in m.itertuples(index=False):
        key = (int(row.COMID), int(row.Row), int(row.Col))
        if key not in qmax or not qmax[key] > 0:
            continue
        q = share * qmax[key]
        old = _wse_at([getattr(row, f"q_{i}_l") for i in range(1, count + 1)],
                      [getattr(row, f"wse_{i}_l") for i in range(1, count + 1)], q)
        new = _wse_at([getattr(row, f"q_{i}_n") for i in range(1, count + 1)],
                      [getattr(row, f"wse_{i}_n") for i in range(1, count + 1)], q)
        out.append((key[1], key[2], old, new))
    return np.array(out, dtype=float)


def _densest_window(rows, cols, dx, dy, half_metres, shape):
    """A window half_metres either way around the cell with the most of the cells near it."""
    half_r, half_c = int(half_metres / dy), int(half_metres / dx)
    best = max(range(0, rows.size, 3), key=lambda k: np.sum((np.abs(rows - rows[k]) <= half_r)
                                                            & (np.abs(cols - cols[k]) <= half_c)))
    r, c = int(rows[best]), int(cols[best])
    return Window(max(r - half_r, 0), min(r + half_r, shape[0] - 1), max(c - half_c, 0), min(c + half_c, shape[1] - 1),
                  dx, dy)


@figure("O2", "Water surfaces for the same discharge", "The outputs")
def same_discharge(ctx):
    site = ctx.detail_sites[0]
    grid = ctx.grid(site)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    stats = {}
    for ax, (config, label) in zip(axes[:2], (("as_configured", "as configured"),
                                              ("no_bathymetry", "without bathymetry"))):
        values = _same_discharge(ctx, config, site)
        difference = values[:, 3] - values[:, 2]
        rows, cols = values[:, 0].astype(int), values[:, 1].astype(int)
        window = _densest_window(rows, cols, grid.dx, grid.dy, 800.0, grid.dem.shape)
        hillshade(ax, window, grid.dem, alpha=0.5)
        raster = np.full(grid.dem.shape, np.nan)
        raster[rows, cols] = difference
        image = show_raster(ax, window, raster, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax.set_title(f"{label}: new − legacy water surface\nat half each cell's maximum flow", fontsize=8.5)
        fig.colorbar(image, ax=ax, shrink=0.75, label="metres")
    for config, label, color in (("as_configured", "as configured", LEGACY), ("no_bathymetry", "without bathymetry",
                                                                            ACCENT)):
        pooled = np.concatenate([d[:, 3] - d[:, 2] for d in (_same_discharge(ctx, config, s) for s in ctx.sites)
                                 if d.size])
        pooled = pooled[np.isfinite(pooled)]
        axes[2].hist(np.clip(pooled, -2, 2), bins=np.linspace(-2, 2, 81), histtype="step", color=color, lw=1.3,
                     label=f"{label}: median {np.median(pooled):+.2f} m, |Δ| median {np.median(np.abs(pooled)):.2f} m",
                     density=True)
        stats[config] = dict(cells=int(pooled.size), median=rounded(np.median(pooled), 3),
                             abs_median=rounded(np.median(np.abs(pooled)), 3),
                             abs_p90=rounded(np.percentile(np.abs(pooled), 90), 3))
    axes[2].set_xlabel("new − legacy water surface (m)")
    axes[2].set_ylabel("density")
    axes[2].set_title(f"all {len(ctx.sites)} sites", fontsize=9)
    axes[2].legend(loc="upper left", fontsize=6.5)
    return fig, dict(caption=(
        f"The water surface each code's rating curve gives for half of each cell's maximum flow (interpolated "
        f"between increments), new minus legacy: the stream cells of {ctx.site_label(site)} drawn to scale, and the "
        "differences pooled over every site. Comparing at the same discharge takes out the increments' different "
        "depths. With bathymetry the new channels, carved to their own shapes, change the low flows most; red is "
        "where the new water surface is higher."), stats=stats)


@figure("O3", "The first increment's top width", "The outputs")
def first_top_width(ctx):
    fig, ax = plt.subplots(figsize=(8, 3.9))
    stats = {}
    bins = np.logspace(-1, 3.5, 91)
    for config, style in (("as_configured", "-"), ("no_bathymetry", "--")):
        for code, color in (("legacy", LEGACY), ("new", NEW)):
            values = np.concatenate([v["t_1"].to_numpy() for v in (ctx.vdt(config, code, s) for s in ctx.sites)
                                     if v is not None])
            values = values[values > 0]
            ax.hist(np.clip(values, bins[0], bins[-1]), bins=bins, histtype="step", color=color, ls=style, lw=1.3,
                    label=f"{code}, {config.replace('_', ' ')}: median {np.median(values):.1f} m")
            stats[f"{code}_{config}"] = rounded(np.median(values), 2)
    ax.set_xscale("log")
    ax.set_xlabel("top width at the first increment (m)")
    ax.set_ylabel("VDT rows")
    ax.legend(loc="upper left", fontsize=7)
    return fig, dict(caption=(
        "The top width of every VDT row's first increment, a thirtieth of the way from the stream cell to the "
        "maximum flow's water surface. With bathymetry the new channels are trapezoids with flat beds, as wide as "
        "their banks at the top, where legacy's single cells were Vs to the neighbouring ordinates; so the first "
        "increment is wider. Without bathymetry the two are close."), stats=stats)


@figure("O4", "Speed", "The outputs", needs_timing=True)
def speed(ctx):
    old = np.array([ctx.seconds("timing", "legacy", s) for s in ctx.sites])
    new = np.array([ctx.seconds("timing", "new", s) for s in ctx.sites])
    keep = np.isfinite(old) & np.isfinite(new)
    old, new = old[keep], new[keep]
    ratio = old / new
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    ax.loglog(old, new, "o", color=NEW, ms=4)
    lims = [min(old.min(), new.min()) * 0.8, max(old.max(), new.max()) * 1.2]
    ax.plot(lims, lims, color=GROUND, lw=0.8, label="as fast")
    ax.plot(lims, np.array(lims) / 2, color=GROUND, lw=0.8, ls=":", label="twice as fast")
    ax.set_xlabel("legacy (s)")
    ax.set_ylabel("new (s)")
    ax.set_title(f"each site, one process each, VDT and bathymetry ({keep.sum()} sites)", fontsize=8.5)
    ax.legend(loc="upper left")
    note(ax, f"totals: legacy {old.sum():.0f} s, new {new.sum():.0f} s", loc="lower right")
    ax = axes[1]
    ax.hist(ratio, bins=np.linspace(0.5, 3.0, 26), color=NEW, alpha=0.8)
    ax.axvline(1.0, color=GROUND, lw=0.8)
    ax.set_xlabel("legacy's time / the new time")
    ax.set_ylabel("sites")
    note(ax, f"median {np.median(ratio):.2f}×, {ratio.min():.2f}× to {ratio.max():.2f}×\n"
             f"new faster on {int((ratio > 1).sum())} of {ratio.size}", loc="upper right")
    return fig, dict(caption=(
        "Each code's run time on each site, in a fresh process of its own, one at a time, writing what the site's "
        "input file asks for (the VDT database and bathymetry); the time excludes Python's imports but includes "
        "numba loading its compiled functions."),
                 stats=dict(sites=int(keep.sum()), legacy_total=rounded(old.sum(), 1), new_total=rounded(new.sum(), 1),
                            median_speedup=rounded(np.median(ratio), 3), min_speedup=rounded(ratio.min(), 3),
                            max_speedup=rounded(ratio.max(), 3), faster=int((ratio > 1).sum())))


@figure("O5", "Sanity check: the same grid and the same stream cells", "The outputs")
def same_inputs(ctx):
    rows = []
    for site in ctx.sites:
        legacy, new = ctx.capture("as_configured", "legacy", site), ctx.capture("as_configured", "new", site)
        if legacy is None or new is None:
            continue
        old_cells = {(c["row"], c["col"], c["comid"]) for c in legacy["cells"] if c is not None}
        new_cells = {(c["row"], c["col"], c["comid"]) for c in new["cells"] if c is not None}
        old_vdt, new_vdt = ctx.vdt("as_configured", "legacy", site), ctx.vdt("as_configured", "new", site)
        rows.append((legacy["dx"], new["dx"], legacy["dy"], new["dy"], len(old_cells), len(new_cells),
                     len(old_cells & new_cells), len(old_vdt), len(new_vdt),
                     list(old_vdt.columns) == list(new_vdt.columns)))
    rows = np.array(rows, dtype=object)
    dx = rows[:, :4].astype(float)
    counts = rows[:, 4:9].astype(float)
    same_columns = bool(np.all(rows[:, 9]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    ax.plot(dx[:, 0], dx[:, 1], "o", color=NEW, ms=4, label="dx")
    ax.plot(dx[:, 2], dx[:, 3], "s", color=ACCENT, ms=4, label="dy")
    lims = [dx.min() - 0.5, dx.max() + 0.5]
    ax.plot(lims, lims, color=GROUND, lw=0.8)
    ax.set_xlabel("legacy's cell size (convert_cell_size, m)")
    ax.set_ylabel("new cell size (Raster.cell_size_in_metres, m)")
    ax.legend(loc="upper left")
    largest = max(np.abs(dx[:, 0] - dx[:, 1]).max(), np.abs(dx[:, 2] - dx[:, 3]).max())
    note(ax, f"{len(rows)} sites\nlargest difference {largest:.1e} m", loc="lower right")
    ax = axes[1]
    ax.loglog(counts[:, 0], counts[:, 1], "o", color=NEW, ms=4)
    lims = [counts[:, :2].min() * 0.8, counts[:, :2].max() * 1.2]
    ax.plot(lims, lims, color=GROUND, lw=0.8)
    ax.set_xlabel("stream cells with a cross section, legacy")
    ax.set_ylabel("stream cells with a cross section, new")
    note(ax, f"the same cells at every site: {bool(np.all(counts[:, 2] == counts[:, 0]) and np.all(counts[:, 2] == counts[:, 1]))}",
         loc="lower right")
    ax = axes[2]
    ax.loglog(counts[:, 3], counts[:, 4], "o", color=NEW, ms=4)
    lims = [counts[:, 3:].min() * 0.8, counts[:, 3:].max() * 1.2]
    ax.plot(lims, lims, color=GROUND, lw=0.8)
    ax.set_xlabel("VDT rows, legacy")
    ax.set_ylabel("VDT rows, new")
    note(ax, f"totals: legacy {int(counts[:, 3].sum()):,}, new {int(counts[:, 4].sum()):,}\n"
             f"same columns in the same order: {same_columns}", loc="lower right")
    return fig, dict(caption=(
        "Things both codes should share, on every site as configured. Left: the cell size in metres of the "
        "geographic rasters. Middle: the stream cells each worked on (those with the flow file's IDs, less the "
        "reaches whose slope couldn't be resolved) that got a cross section. Right: the VDT rows written, which "
        "differ only where one code accepted a rating curve and the other didn't, and the VDT's columns. The "
        "writers themselves are byte-identical to legacy's on the same data (tests)."),
                 stats=dict(sites=len(rows), largest_cell_size_difference=float(largest),
                            same_cells=bool(np.all(counts[:, 2] == counts[:, 0]) and np.all(counts[:, 2] == counts[:, 1])),
                            vdt_rows_legacy=int(counts[:, 3].sum()), vdt_rows_new=int(counts[:, 4].sum()),
                            same_columns=same_columns))


@figure("O6", "Sanity check: representative cross sections from the same cross sections", "The outputs")
def representative(ctx):
    from arc.hydraulic_data import build_representative_cross_section_dataframe
    from arc.outputs import RepresentativeSample, representative_cross_section_dataframe
    from fig_hydraulics import _legacy_section
    site = ctx.detail_sites[0]
    capture = ctx.capture("uniform_n", "legacy", site)
    cells = [c for c in capture["cells"] if c is not None]
    records, samples = [], []
    for cell in cells:
        (side1, side2), (n1, n2) = cell["final"], cell["n"]
        b1, b2 = cell["hydraulic_banks"]
        records.append({"COMID": cell["comid"], "XS1_Profile": side1, "XS2_Profile": side2, "Manning_N_Raster1": n1,
                        "Manning_N_Raster2": n2, "Ordinate_Dist": cell["spacing"], "Slope": cell["slope"],
                        "Thalweg": float(side1[0]), "Bank_Index1": int(b1), "Bank_Index2": int(b2)})
        samples.append(RepresentativeSample(cell["comid"], _legacy_section(cell), float(side1[0]), cell["slope"]))
    old = build_representative_cross_section_dataframe(records, 6.0, 1.0, 1.0, 1.0)
    new = representative_cross_section_dataframe(samples, roughness=None, slope_factor=1.0)
    merged = old.merge(new, on=["COMID", "Depth_Stage_Index"], suffixes=("_l", "_n"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    stats = {}
    for ax, column, unit in zip(axes, ("Mean_Top_Width", "Mean_Cross_Sectional_Area", "Mean_Discharge"),
                                ("m", "m²", "m³/s")):
        x, y = merged[f"{column}_l"].to_numpy(float), merged[f"{column}_n"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ax.loglog(x[ok], y[ok], ".", ms=2, color=NEW, alpha=0.5)
        lims = [min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())]
        ax.plot(lims, lims, color=GROUND, lw=0.8)
        relative = np.abs(y[ok] / x[ok] - 1)
        note(ax, f"|Δ|/legacy median {np.median(relative):.1e}\np99 {np.percentile(relative, 99):.1e}")
        ax.set_xlabel(f"legacy {column.replace('_', ' ').lower()} ({unit})")
        ax.set_ylabel(f"new ({unit})")
        stats[column] = dict(median=float(np.median(relative)), p99=float(np.percentile(relative, 99)))
    return fig, dict(caption=(
        f"Each reach's representative cross section (every 0.1 m of stage up to 25 m, averaged over the reach's "
        f"cross sections), built by legacy's build_representative_cross_section_dataframe and by the new "
        f"representative module from the same cross sections: legacy's own, from its run of "
        f"{ctx.site_label(site)} with one n everywhere. They agree to about legacy's rounding: it rounded area, "
        "perimeter, top width, discharge and velocity to 3 decimals and each water surface to the millimetre."),
                 stats=stats)
