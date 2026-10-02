"""Another branch of ARC run as a third code (make_figures --other NAME=SRC), against legacy and the new code: Joseph
Gutenson's varying_roughness_and_slope (MikeFHS/automated-rating-curve, 187eb40), whose bank smoothing is a profile
falling downstream through the reach's banks, with an outlier filter and each reach no higher than its inflows. His
smoothing is also run inside the new code (vc_variants.joseph_smoothing, checked against his functions)."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from vc_plot import GROUND, LEGACY, NEW, PINK, figure, note, rounded

SECTION = "Joseph's branch: varying_roughness_and_slope"
OTHER = PINK


def _other(ctx):
    return next(iter(ctx.others), None)


def _label(name):
    return f"{name.capitalize()}'s branch"


def _failed(ctx, name):
    """Each site the other ARC failed on, and why."""
    import vc_runs
    reasons = {}
    for site in ctx.sites:
        path = vc_runs.run_directory(ctx.runs, "as_configured", name, site) / "failed.txt"
        if path.exists():
            reasons[site] = path.read_text().strip()
    return reasons


def _legacy_like(capture):
    """A legacy-style capture's smoothed bank elevation less each cross section's stream cell (in the ground's frame,
    fig_bathymetry.legacy_offset), and how far its carve moved the stream cell, where its bathymetry applied."""
    from fig_bathymetry import legacy_offset
    cells = [c for c in capture["cells"] if c is not None]
    stream = np.array([c["found"][0][0] for c in cells])
    raise_ = legacy_offset(capture)
    bank = np.array([c["banks"].get("smoothed_bank_elevation", np.nan) for c in cells], float) - raise_ - stream
    applies = np.array([bool(c["banks"].get("bathymetry_should_apply", False)) for c in cells])
    carved = np.array([c["final"][0][0] for c in cells], float) - stream
    return bank, np.where(applies, carved, np.nan)


def _new_way(ctx, site, key):
    """The new code's site with its own bank smoothing ("new") or Joseph's ("joseph"), rebuilt from its capture:
    (capture, smoothing, channels), cached."""
    def make():
        import vc_variants
        from arc.bathymetry import smooth_bank_elevations
        from fig_bathymetry import _channels_with
        smooth = smooth_bank_elevations if key == "new" else vc_variants.joseph_smoothing(smooth_bank_elevations)
        return _channels_with(ctx, site, smooth)
    return ctx._cached(("new_way", site, key), make)


def _new_like(ctx, site, key):
    """The new code's bank elevation less each cross section's stream cell, with either smoothing, and how far its
    carve moves the stream cell where it carves: the capture's own for "new", and for "joseph" each cross section
    carved again with his bank elevation (the stream cell of a lopsided channel is on its side, not its bed)."""
    from fig_bathymetry import new_carve
    new, smoothed, channels = _new_way(ctx, site, key)
    banks, carved = [], []
    for reach, result in smoothed.items():
        cells = [new["cells"][k] for k in new["reaches"][reach]["cells"]]
        stream = np.array([c["found"][c["found"].size // 2] for c in cells])
        bank = np.asarray(result.bank_elevations, float)
        banks.append(bank - stream)
        for i, c in enumerate(cells):
            m = c["found"].size // 2
            if not np.isfinite(c["carve_depth"]) or reach not in channels or not np.isfinite(bank[i]):
                carved.append(np.nan)
            elif key == "new":
                carved.append(c["final"][m] - c["found"][m])
            else:
                xs, _ = new_carve(c["found"], c["spacing"], c["hydraulic_banks"], channels[reach].depths[i],
                                  bank_elevation=bank[i], trapezoid_height=new["trapezoid_height"])
                carved.append(xs.elevations[m] - c["found"][m])
    return (np.concatenate(banks) if banks else np.empty(0), np.array(carved, float))


def _ran(ctx, name):
    return [s for s in ctx.sites if ctx.capture("as_configured", name, s) is not None
            and ctx.capture("as_configured", "legacy", s) is not None
            and ctx.capture("as_configured", "new", s) is not None]


def _ways(name):
    return (("legacy", LEGACY, "-", "legacy"), (name, OTHER, "-", _label(name)),
            ("new_joseph", OTHER, "--", "the new code with his smoothing"), ("new", NEW, "-", "the new code"))


@figure("JG1", "Bank elevations: legacy, Joseph's branch and the new code", SECTION, needs_other=True)
def other_banks(ctx):
    name = _other(ctx)
    ran, failed = _ran(ctx, name), _failed(ctx, name)
    heights = {key: [] for key, *_ in _ways(name)}
    shares = []
    for site in ran:
        values = {"legacy": _legacy_like(ctx.capture("as_configured", "legacy", site))[0],
                  name: _legacy_like(ctx.capture("as_configured", name, site))[0],
                  "new": _new_like(ctx, site, "new")[0], "new_joseph": _new_like(ctx, site, "joseph")[0]}
        for key, v in values.items():
            heights[key].append(v[np.isfinite(v)])
        shares.append((site, *(float(np.mean(values[k][np.isfinite(values[k])] < 0)) if np.isfinite(values[k]).any()
                               else np.nan for k in ("legacy", name, "new_joseph"))))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.05, 1.4]))
    ax = axes[0]
    bins = np.arange(-8, 8.01, 0.25)
    stats = dict(sites_run=len(ran), sites=len(ctx.sites), failed={s: r for s, r in failed.items()})
    for key, color, style, label in _ways(name):
        v = np.concatenate(heights[key]) if heights[key] else np.empty(0)
        stats[key] = dict(below=rounded(float(np.mean(v < 0)), 3), median=rounded(float(np.median(v)), 2))
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", color=color, ls=style, lw=1.3,
                label=f"{label}: below the stream cell at {np.mean(v < 0):.0%}")
    ax.axvline(0, color=GROUND, lw=0.8)
    ax.set_xlabel("smoothed bank elevation − the stream cell (m)")
    ax.set_ylabel(f"cross sections, the {len(ran)} sites all three ran")
    ax.legend(loc="upper left", fontsize=6.3)
    ax = axes[1]
    shares.sort(key=lambda r: -r[1])
    y = np.arange(len(shares))
    ax.barh(y, [r[1] for r in shares], color=LEGACY, height=0.8, alpha=0.8, label="legacy")
    ax.plot([r[2] for r in shares], y, "|", color=OTHER, ms=7, mew=1.8, label=_label(name))
    ax.plot([r[3] for r in shares], y, ".", color=OTHER, ms=4, label="the new code with his smoothing")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0].replace("_", " ")[:28] for r in shares], fontsize=4.8)
    ax.invert_yaxis()
    ax.set_xlabel("share of the site's cross sections whose bank elevation\nis below the stream cell "
                  "(the new code's: none)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax = axes[2]
    site = next((s for s in ran if s.startswith("Du_Page")), None)
    if site is not None:
        from fig_bathymetry import _new_stations
        reach = 760524857
        new = ctx.capture("as_configured", "new", site)
        by_cell, stations = _new_stations(new, reach)
        order = np.argsort(stations)
        cells = [new["cells"][k] for k in new["reaches"][reach]["cells"]]
        stream = np.array([c["found"][c["found"].size // 2] for c in cells])
        ax.plot(stations[order], stream[order], color=GROUND, lw=0.9, label="the stream cells")
        for key, color, style, label in _ways(name):
            if key in ("legacy", name):
                points = sorted((by_cell[(c["row"], c["col"])], c["banks"].get("smoothed_bank_elevation", np.nan))
                                for c in ctx.capture("as_configured", key, site)["cells"]
                                if c is not None and (c["row"], c["col"]) in by_cell)
                points = np.array(points)
                ax.plot(points[:, 0], points[:, 1], color=color, ls=style, lw=1.6, label=label)
                stats.setdefault("du_page", {})[key] = rounded(float(np.nanmedian(
                    points[:, 1] - np.interp(points[:, 0], stations[order], stream[order]))), 2)
            else:
                _, smoothed, _ = _new_way(ctx, site, "new" if key == "new" else "joseph")
                bank = np.asarray(smoothed[reach].bank_elevations, float)
                ax.plot(stations[order], bank[order], color=color, ls=style, lw=1.6, label=label)
                stats.setdefault("du_page", {})[key] = rounded(float(np.nanmedian(bank - stream)), 2)
        ax.set_xlabel("metres along the reach from its upstream end")
        ax.set_ylabel("elevation (m)")
        ax.set_title(f"{ctx.site_label(site).split(' (')[0]}, reach {reach}: mild, then steep, below an inflow "
                     "with no cross sections", fontsize=8)
        ax.legend(loc="upper right", fontsize=6.3)
    # the sites the new code and legacy ran where his stops, and why; and those no code ran (no flow file)
    stops = {site: reason for site, reason in failed.items()
             if ctx.capture("as_configured", "legacy", site) is not None}
    reasons = {}
    for reason in stops.values():
        reasons[reason] = reasons.get(reason, 0) + 1
    stats["failure_reasons"] = reasons
    stats["stops"] = sorted(stops)
    stats["no_code_ran"] = sorted(set(failed) - set(stops))
    runnable = len(ran) + len(stops)
    return fig, dict(caption=(
        f"The smoothed bank elevation, the level each channel is carved below with Bathy_Use_Banks, less its stream "
        f"cell, at the {len(ran)} of the {runnable} sites legacy and the new code run where {_label(name)} "
        "(MikeFHS/automated-rating-curve varying_roughness_and_slope, 187eb40) runs too; at the other "
        f"{len(stops)} it stops with "
        + ("; ".join(f"{reason!r}" if len(reasons) == 1 else f"{count} with {reason!r}"
                     for reason, count in reasons.items())) + ". His smoothing takes each "
        "cross section's lower bank, replaces those outside the reach's 10th to 90th percentiles with the next kept "
        "one downstream (his notes say the 25th and 75th; the code keeps the 10th to 90th), starts each reach no higher "
        "than the lowest outlet flowing into it, and falls from bank to lower bank downstream, no faster than 0.5 m a "
        "metre and at 1e-4 where none is lower; the stream cell doesn't bound it. Left: every cross section. Middle: "
        "the share below the stream cell at each site. Right: the Du Page reach where legacy's smoothing ran metres "
        "under the water (BS3). His smoothing inside the new code (dashed) is the same functions, checked against "
        "his on random reaches (JG2), on the new code's banks."), stats=stats)


def _port_check(src, trials=4000, seed=3):
    """vc_variants' port of his smoothing against his own two functions, taken from his source, on random reaches:
    how many were compared, and whether any differed."""
    import ast
    from pathlib import Path

    import vc_variants
    path = Path(src) / "arc" / "Automated_Rating_Curve_Generator.py"
    wanted = {"_build_downstream_monotone_bank_profile", "_replace_reach_bank_outliers_with_downstream"}
    namespace = {"np": np, "MIN_SLOPE": vc_variants.JOSEPH_MIN_SLOPE, "MAX_SLOPE": vc_variants.JOSEPH_MAX_SLOPE}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    if not wanted <= set(namespace):
        return dict(compared=0, missing=sorted(wanted - set(namespace)))
    profile, outliers = namespace["_build_downstream_monotone_bank_profile"],         namespace["_replace_reach_bank_outliers_with_downstream"]
    rng = np.random.default_rng(seed)
    compared, differ, largest = 0, 0, 0.0
    for _ in range(trials):
        n = int(rng.integers(1, 60))
        stations = np.cumsum(rng.uniform(5, 60, n)) - 5
        banks = 100 - 0.002 * stations + rng.normal(0, rng.choice([0.2, 1.0, 3.0]), n)
        if rng.random() < 0.5:
            banks[rng.random(n) < 0.15] = np.nan
        if rng.random() < 0.3:
            banks[rng.integers(0, n)] -= rng.uniform(1, 8)
            banks = np.round(banks, 1)
        control = None
        if rng.random() < 0.5 and np.isfinite(banks).any():
            control = float(np.nanmax(banks) - rng.uniform(-2, 4))
        try:
            theirs, _ = outliers(banks, 1)
        except ValueError:
            continue
        differ += not np.array_equal(theirs, vc_variants.joseph_outliers(banks), equal_nan=True)
        if not np.isfinite(theirs[0]) and control is None:
            continue
        largest = max(largest, float(np.nanmax(np.abs(profile(theirs, stations, control)[0]
                                                      - vc_variants.joseph_profile(theirs, stations, control)))))
        compared += 1
    return dict(compared=compared, outliers_differ=int(differ), largest_profile_difference_m=largest)


@figure("JG2", "Joseph's bank smoothing on made-up reaches", SECTION, needs_other=True)
def other_made_up(ctx):
    import vc_variants
    from arc.bathymetry import smooth_bank_elevations
    from fig_bathymetry import _smooth_made_up
    name = _other(ctx)
    cell, count = 30.0, 300
    x = np.arange(count) * cell
    steady = 100.0 - 0.001 * x
    mild_then_steep = np.where(x < 6000, 165.0 - 0.0001 * x, 164.4 - 8.0 * (x - 6000) / 3000)
    cases = (("a steady reach falling 0.1%", steady, False),
             ("mild, then steep (as the Du Page reach)", mild_then_steep, False),
             ("the steady reach, below an inflow with no cross sections", steady, True))
    methods = (("legacy", LEGACY, "legacy's"), (vc_variants.joseph_smoothing(smooth_bank_elevations), OTHER, "his"),
               ("water_plus_height", NEW, "the new"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    stats = dict(port_check=_port_check(ctx.others[name]))
    for ax, (title, water, phantom) in zip(axes, cases):
        rng = np.random.default_rng(21)
        stream = water + rng.normal(0.0, 0.15, count)
        tops = water + 1.5
        observed = tops + rng.normal(0.0, 0.3, count)
        walls = rng.random(count) < 0.4
        observed[walls] += rng.uniform(1.0, 4.0, walls.sum())
        ax.plot(x, stream, color=GROUND, lw=0.8, label="the stream cells (the DEM's water)")
        ax.plot(x, tops, color=GROUND, lw=1.0, ls="--", label="the true bank tops")
        ax.plot(x, observed, "o", color=NEW, ms=2.0, mfc="none", alpha=0.4, label="each cross section's observation")
        lines = []
        for method, color, label in methods:
            result = _smooth_made_up(stream, observed, cell, phantom=phantom, method=method)
            bank = np.asarray(result.bank_elevations)[result.order]
            ax.plot(x, bank, color=color, lw=1.8, label=f"{label} smoothing")
            under, below = float(np.median(tops - bank)), float(np.mean(bank < stream))
            lines.append(f"{label}: {under:+.2f} m under the banks, below the water at {below:.0%}")
            stats.setdefault(title, {})[label] = dict(median_under_banks=rounded(under, 2),
                                                      below_stream=rounded(below, 3))
        note(ax, "\n".join(lines), loc="lower left")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("metres along the reach")
    axes[0].set_ylabel("elevation (m)")
    axes[0].legend(loc="upper right", fontsize=6.3)
    return fig, dict(caption=(
        "BS2's made-up reaches (true banks 1.5 m above the DEM's water, observations 0.3 m about them, 40% single "
        "cells whose observations are 1 to 4 m higher), smoothed three ways. His smoothing falls from bank to lower "
        "bank, so it follows a reach that runs mild and then steep, as ours does, and it takes the reach's shape from "
        "its own banks where legacy's lower envelope took it from its outlet; an inflow without cross sections has no "
        "outlet of its own to cap the reach below it. But its outlier band is on the banks' elevations, so where a "
        "reach falls, its lowest tenth, at its downstream end, is outside the band and takes the last kept bank "
        "upstream: the profile holds level over the reach's last stretch while the banks go on falling (the right of "
        "each panel), and the carve fills there."), stats=stats)


@figure("JG3", "What each carve does to the stream cell: legacy, Joseph's branch and the new code", SECTION,
        needs_other=True)
def other_beds(ctx):
    name = _other(ctx)
    ran = _ran(ctx, name)
    beds = {key: [] for key, *_ in _ways(name)}
    for site in ran:
        values = {"legacy": _legacy_like(ctx.capture("as_configured", "legacy", site))[1],
                  name: _legacy_like(ctx.capture("as_configured", name, site))[1],
                  "new": _new_like(ctx, site, "new")[1], "new_joseph": _new_like(ctx, site, "joseph")[1]}
        for key, v in values.items():
            beds[key].append(v[np.isfinite(v)])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bins = np.arange(-8, 8.01, 0.25)
    stats = {}
    for key, color, style, label in _ways(name):
        v = np.concatenate(beds[key]) if beds[key] else np.empty(0)
        stats[key] = dict(cross_sections=int(v.size), median=rounded(float(np.median(v)), 2),
                          above=rounded(float(np.mean(v > 0)), 3), above_2m=rounded(float(np.mean(v > 2)), 3),
                          below_2m=rounded(float(np.mean(v < -2)), 3))
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", color=color, ls=style, lw=1.3,
                label=f"{label}: raised at {np.mean(v > 0):.0%} (by more than 2 m at {np.mean(v > 2):.0%}), "
                      f"lowered by more than 2 m at {np.mean(v < -2):.0%}")
    ax.axvline(0, color=GROUND, lw=0.8)
    ax.set_xlabel("the stream cell after the carve − before (m); positive: raised")
    ax.set_ylabel(f"carved cross sections, the {len(ran)} sites")
    ax.legend(loc="upper left", fontsize=7)
    lab = {key: label for key, _, _, label in _ways(name)}
    lowered = ", ".join(f"{lab[k]} {v['below_2m']:.0%}" for k, v in stats.items())
    raised = ", ".join(f"{lab[k]} {v['above']:.0%}" for k, v in stats.items())
    return fig, dict(caption=(
        "How far each code's carve moves the stream cell, the DEM's water, at every carved cross section of the "
        f"{len(ran)} sites all three codes ran: legacy's and his from their own runs, the new code's from its run, and "
        "the new code with his smoothing carved again with his bank elevation. Where a bank elevation runs under the "
        "water the carve digs the channel out below it, and where it is high above the water the carve fills. "
        f"Lowered by more than 2 m: {lowered}. Raised: {raised}."), stats=stats)


@figure("JG4", "Flood maps: Joseph's branch against legacy and the new code", SECTION, needs_other=True,
        needs_fim=True)
def other_maps(ctx):
    from fig_fim import paired, site_bootstrap
    name = _other(ctx)
    comparisons = ((f"{_label(name)} − legacy", "legacy", name), (f"the new code − {_label(name)}", name, "new"),
                   ("the new code with his smoothing − the new code", "new", "new_joseph_smoothing"))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    stats = {}
    for ax, (title, base, other) in zip(axes, comparisons):
        j = paired(ctx, base, other)
        if j is None or not len(j):
            ax.set_visible(False)
            continue
        per_site = j["d"].groupby(level=0).mean().sort_values()
        lo, hi = site_bootstrap(j["d"])
        ax.bar(np.arange(per_site.size), per_site.to_numpy(), width=0.8,
               color=[LEGACY if v < 0 else NEW for v in per_site])
        ax.axhline(0, color=GROUND, lw=0.8)
        for k in list(range(2)) + list(range(per_site.size - 2, per_site.size)):
            ax.text(k, per_site.iloc[k], " " + per_site.index[k].replace("_", " ")[:24], rotation=90, fontsize=5.6,
                    va="bottom" if per_site.iloc[k] > 0 else "top", ha="center")
        ax.set_xticks([])
        ax.set_xlabel(f"the {per_site.size} sites both have, sorted")
        ax.set_ylabel("the site's mean change in MCC")
        ax.set_title(title, fontsize=8.5)
        note(ax, f"mean {j['d'].mean():+.4f} (95% {lo:+.4f} to {hi:+.4f})\nsites better "
                 f"{np.mean(per_site > 0):.0%}; median bias {j['bias'].median():.2f} → {j['bias_o'].median():.2f}",
             loc="upper left")
        stats[other if base != name else "new_vs_other"] = dict(
            mean=rounded(j["d"].mean(), 4), interval=[rounded(lo, 4), rounded(hi, 4)], sites=int(per_site.size),
            pairs=int(len(j)), sites_better=rounded(float(np.mean(per_site > 0)), 3),
            bias=[rounded(j["bias"].median(), 3), rounded(j["bias_o"].median(), 3)],
            worst={s: rounded(v, 3) for s, v in per_site.head(3).items()},
            best={s: rounded(v, 3) for s, v in per_site.tail(3)[::-1].items()})
    # the new code against legacy on the same sites as his, for scale
    j = paired(ctx, "legacy", "new")
    theirs = paired(ctx, "legacy", name)
    if j is not None and theirs is not None:
        same = j[j.index.get_level_values(0).isin(theirs.index.get_level_values(0).unique())]
        lo, hi = site_bootstrap(same["d"])
        stats["new_vs_legacy_same_sites"] = dict(mean=rounded(same["d"].mean(), 4), interval=[rounded(lo, 4),
                                                                                                rounded(hi, 4)])
    mine = stats.get(name, {})
    same = stats.get("new_vs_legacy_same_sites", {})
    return fig, dict(caption=(
        f"The benchmark's flood maps (F1) with {_label(name)}, pair by pair on the sites where it runs, against legacy "
        f"(left: {mine.get('mean', math.nan):+.4f}, where the new code gains "
        f"{same.get('mean', math.nan):+.4f} over legacy on the same sites) and against the new code (middle); and his "
        "bank smoothing inside the new code, on every site (right). Each bar is a site's mean change in MCC; the "
        "intervals resample sites."), stats=stats)
