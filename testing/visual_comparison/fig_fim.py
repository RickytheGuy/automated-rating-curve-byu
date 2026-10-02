"""Flood maps on the FIM benchmark (fim_benchmark.py): the new code against legacy, and which changes matter."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vc_plot import ACCENT, GROUND, LEGACY, NEW, PINK, WATER, Window, figure, note, rounded, show_raster

SECTION = "Flood maps on the FIM benchmark"
# the variants of F2, against the new code as it is, in groups
GROUPS = (
    ("the angle search (the new code's is 5 m)",
     (("new_td05", "test depth 0.5 m (legacy's)"), ("new_td1", "test depth 1 m"), ("new_td2", "test depth 2 m"),
      ("new_td10", "test depth 10 m"), ("new_nosearch", "no search (Degree_Manip 0)"))),
    ("the bed cap (the new code's is MAX_SLOPE)", (("new_cap001", "1 cm per metre (legacy's)"), ("new_nocap", "no cap"))),
    # tested on 2026-10-01 with an option (Segment_Roughness) that was removed afterwards; the row shows the scores
    # kept from that run
    ("a segment's roughness (the new code's: its inner end's n)",
     (("new_legacy_segment_n", "a wholly wet segment's outer end (legacy's)"),)),
    ("the new smoothing's constraint: never rising downstream",
     (("new_wh10_bankfree", "the bank elevation free to rise"),
      ("new_wh10_free", "the water surface and the bank elevation free to rise"))),
    ("other levels to carve the channel below",
     (("new_wh25", "water surface + 25th-percentile height"),
      ("new_fq25", "falling fit of the banks' 25th percentile"),
      ("new_local", "each cross section's own banks"),
      ("new_observed_clamp", "legacy's smoothing, observed inflows, never below the stream cell"),
      ("new_legacy_smoothing", "legacy's smoothing (the new code's until 2026-09-26)"),
      ("new_wse", "the stream cell (Bathy_Use_Banks 0)"))),
    ("a test of the fills, not a proposal", (("new_nofill", "no bed above the stream cell"),)),
    ("the cross section's direction as the water rises (PV1)",
     (("new_pivot", "pivoting at every increment"), ("new_pivot_top", "the direction narrowest at the top, throughout"))),
    ("Joseph's branch, varying_roughness_and_slope (JG4)",
     (("joseph", "his ARC, on the sites it runs"), ("new_joseph_smoothing", "his bank smoothing in the new code"))),
    ("for scale", (("legacy", "legacy ARC"),)),
)
# the new code as it is, for the captions
AS_IT_IS = ("the search 5 m deep, the bed cap at MAX_SLOPE, and the bank elevation the DEM's water surface plus the 10th "
            "percentile of the nearby banks' heights, never rising downstream")


def scores(ctx, name: str) -> pd.DataFrame | None:
    """A configuration's scores, one row per site and stage with a score."""
    key = ("fim", name)
    if key not in ctx._cache:
        path = ctx.fim / "results" / name / "scores.csv" if ctx.fim is not None else None
        if path is None or not path.exists():
            ctx._cache[key] = None
        else:
            df = pd.read_csv(path)
            if "error" in df:
                df = df[df["error"].isna()]
            df = df.dropna(subset=["mcc"]).copy()
            df["stage"] = df["stage"].astype(str)
            ctx._cache[key] = df.set_index(["site", "stage"])
    return ctx._cache[key]


def paired(ctx, base: str, other: str):
    """The two configurations' scores for the pairs both have, and each pair's stage third within its site."""
    a, b = scores(ctx, base), scores(ctx, other)
    if a is None or b is None:
        return None
    joined = a.join(b, rsuffix="_o", how="inner")
    stage = np.array([float(s.replace("_", ".")) for s in joined.index.get_level_values(1)])
    joined["rank"] = pd.Series(stage, index=joined.index).groupby(level=0).rank(pct=True)
    joined["d"] = joined["mcc_o"] - joined["mcc"]
    return joined


def site_bootstrap(values: pd.Series, reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    """A 95% interval of the mean, resampling sites (the pairs of a site move together)."""
    rng = np.random.default_rng(seed)
    sites = values.index.get_level_values(0)
    groups = [values[sites == s].to_numpy() for s in pd.unique(sites)]
    means = [np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))]).mean()
             for _ in range(reps)]
    return tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))


@figure("F1", "Flood maps: the new code against legacy", SECTION, needs_fim=True)
def new_against_legacy(ctx):
    j = paired(ctx, "legacy", "new")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1.3, 1]))
    ax = axes[0]
    thirds = np.digitize(j["rank"], [1 / 3, 2 / 3], right=True)
    for k, (color, label) in enumerate(((WATER, "lowest third of a site's stages"), (NEW, "middle"),
                                        ("#003f63", "highest"))):
        sel = thirds == k
        ax.plot(j["mcc"][sel], j["mcc_o"][sel], "o", ms=2.5, color=color, alpha=0.7, label=label)
    ax.plot([0, 1], [0, 1], color=GROUND, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("legacy's MCC")
    ax.set_ylabel("the new code's MCC")
    ax.set_title(f"{len(j)} site and stage pairs", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=6.5)
    ax = axes[1]
    per_site = j["d"].groupby(level=0).mean().sort_values()
    colors = [LEGACY if v < 0 else NEW for v in per_site]
    ax.bar(np.arange(per_site.size), per_site.to_numpy(), color=colors, width=0.8)
    ax.axhline(0, color=GROUND, lw=0.8)
    for k in list(range(3)) + list(range(per_site.size - 3, per_site.size)):
        ax.text(k, per_site.iloc[k], " " + per_site.index[k].replace("_", " ")[:26], rotation=90, fontsize=5.8,
                va="bottom" if per_site.iloc[k] > 0 else "top", ha="center")
    ax.set_xticks([])
    ax.set_xlabel(f"the {per_site.size} sites, sorted")
    ax.set_ylabel("new − legacy, the site's mean MCC")
    lo, hi = site_bootstrap(j["d"])
    note(ax, f"mean {j['d'].mean():+.4f} (95%, resampling sites: {lo:+.4f} to {hi:+.4f})\n"
             f"median {j['d'].median():+.4f}; pairs better {np.mean(j['d'] > 0):.0%}, worse {np.mean(j['d'] < 0):.0%}\n"
             f"sites better {np.mean(per_site > 0):.0%}", loc="upper left")
    ax = axes[2]
    ax.loglog(j["bias"], j["bias_o"], "o", ms=2.5, color=NEW, alpha=0.5)
    ax.plot([0.02, 30], [0.02, 30], color=GROUND, lw=0.8)
    ax.axhline(1, color=GROUND, lw=0.5, ls=":")
    ax.axvline(1, color=GROUND, lw=0.5, ls=":")
    ax.set_xlabel("legacy's frequency bias (mapped wet / reference wet)")
    ax.set_ylabel("the new code's")
    ax.set_title(f"median bias: legacy {j['bias'].median():.2f}, new {j['bias_o'].median():.2f}", fontsize=8.5)
    def named(items):
        return ", ".join(f"{site.split('_(')[0].replace('_', ' ')} {value:+.2f}" for site, value in items)
    verdict = ("better" if lo > 0 else "worse" if hi < 0 else
               f"{'better' if j['d'].mean() > 0 else 'worse'} on the mean, within the sites' noise")
    bigger = j["bias_o"].median() > j["bias"].median()
    return fig, dict(caption=(
        "Every site's flood maps at every stage of the thesis's benchmark, built by nencarta's pipeline with each "
        "ARC in turn and scored against the USGS reference extents (fim_benchmark.py); the new code as it is: "
        f"{AS_IT_IS}. Left: each pair's MCC. Middle: each site's mean change, which is what the pairs' spread really "
        f"is: they move together within a site. The new code is {verdict}; it does much better at "
        f"{named(per_site.tail(3)[::-1].items())} and worse at {named(per_site.head(3).items())}. With legacy's bank "
        "smoothing, until 2026-09-26, it was worse on the mean, losing most at Flint, East Fork White and South "
        "Platte, where the carve dug whole incised channels out below a bank elevation metres under their banks "
        f"(F2, F3, BS3). Right: the new maps are {'bigger' if bigger else 'smaller'} than legacy's at the median."),
                 stats=dict(pairs=int(len(j)), sites=int(per_site.size), legacy_median=rounded(j["mcc"].median(), 4),
                            new_median=rounded(j["mcc_o"].median(), 4), mean_difference=rounded(j["d"].mean(), 4),
                            interval=[rounded(lo, 4), rounded(hi, 4)], median_difference=rounded(j["d"].median(), 4),
                            sites_better=rounded(np.mean(per_site > 0), 3),
                            worst={s: rounded(v, 3) for s, v in per_site.head(3).items()},
                            best={s: rounded(v, 3) for s, v in per_site.tail(3).items()},
                            bias=[rounded(j["bias"].median(), 3), rounded(j["bias_o"].median(), 3)]))


@figure("F2", "Flood maps: which change matters", SECTION, needs_fim=True)
def variants(ctx):
    rows = []
    for group, members in GROUPS:
        for name, label in members:
            j = paired(ctx, "new", name)
            if j is None:
                continue
            lo, hi = site_bootstrap(j["d"])
            per_site = j["d"].groupby(level=0).mean()
            rows.append(dict(group=group, name=name, label=label, mean=j["d"].mean(), lo=lo, hi=hi,
                             median=j["d"].median(), bias=j["bias_o"].median(), mcc=j["mcc_o"].median(),
                             changed=float(np.mean(np.abs(j["d"]) > 1e-9)), sites_better=float(np.mean(per_site > 0)),
                             spread=(float(per_site.min()), float(per_site.max())),
                             worst=per_site.sort_values().head(3), best=per_site.sort_values().tail(3)[::-1],
                             sites=int(per_site.size)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.36 * len(rows) + 1.8), gridspec_kw=dict(width_ratios=[1.3, 1]),
                             sharey=True)
    y = np.arange(len(rows))[::-1].astype(float)
    ax = axes[0]
    groups = [r["group"] for r in rows]
    for k, r in enumerate(rows):
        color = LEGACY if r["name"] == "legacy" else NEW
        ax.plot([r["lo"], r["hi"]], [y[k], y[k]], color=color, lw=2.0, alpha=0.6)
        ax.plot(r["mean"], y[k], "o", color=color, ms=5)
        ax.plot(r["median"], y[k], "|", color=GROUND, ms=9, mew=1.5)
    ax.axvline(0, color=GROUND, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5)
    for k in range(1, len(rows)):
        if groups[k] != groups[k - 1]:
            ax.axhline((y[k] + y[k - 1]) / 2, color="#cccccc", lw=0.8)
    for k, r in enumerate(rows):
        if k == 0 or groups[k] != groups[k - 1]:
            ax.text(ax.get_xlim()[0], y[k] + 0.42, r["group"], fontsize=7, color=GROUND, style="italic")
    ax.set_xlabel("change in MCC against the new code as it is: mean (dot), 95% interval resampling sites, "
                  "median (bar)", fontsize=7.5)
    ax = axes[1]
    for k, r in enumerate(rows):
        lo, hi = r["spread"]
        ax.plot([lo, hi], [y[k], y[k]], color=GROUND, lw=1.0)
        j = paired(ctx, "new", r["name"])
        per_site = j["d"].groupby(level=0).mean()
        ax.plot(per_site.to_numpy(), np.full(per_site.size, y[k]), "|", color=LEGACY if r["name"] == "legacy" else NEW,
                ms=6, alpha=0.6)
        ax.text(1.02, y[k], f"bias {r['bias']:.2f}", transform=ax.get_yaxis_transform(), fontsize=6.5, va="center")
    ax.axvline(0, color=GROUND, lw=0.8)
    ax.set_xlabel("each site's mean change in MCC", fontsize=7.5)
    by = {r["name"]: r for r in rows}

    def change(name):
        r = by.get(name)
        return "(not run)" if r is None else f"{r['mean']:+.4f} ({r['lo']:+.4f} to {r['hi']:+.4f})"

    def bias(name):
        return f"{by[name]['bias']:.2f}" if name in by else "?"
    search = [by[n]["mean"] for n in ("new_td05", "new_td1", "new_td2", "new_nosearch") if n in by]
    return fig, dict(caption=(
        f"Each variant's flood maps against the new code's as it is ({AS_IT_IS}), pair by pair, as a mean change in "
        "MCC with a 95% interval resampling sites (right: each site's mean, and the median frequency bias; the new "
        f"code's is {scores(ctx, 'new')['bias'].median():.2f}). The angle search's test "
        "depth from 0.5 to 2 m, or no search at all, moves the maps by less than the sites' noise "
        f"({min(search):+.4f} to {max(search):+.4f}); 10 m is worse, {change('new_td10')}. Legacy's bed cap changes "
        f"{by['new_cap001']['changed'] if 'new_cap001' in by else float('nan'):.0%} of the pairs, by next to nothing. "
        "Taking a wholly wet segment's n from its outer end, as legacy did (H2; tested with an option since "
        f"removed), gives {change('new_legacy_segment_n')}, bias {bias('new_legacy_segment_n')}. "
        "Relaxing the new smoothing's one constraint, that the bank elevation never rises downstream, changes the "
        f"mean by next to nothing too: the bank elevation free to rise, {change('new_wh10_bankfree')}, and the water "
        "surface too, "
        f"{change('new_wh10_free')}, with the maps a little bigger (bias {bias('new_wh10_bankfree')} and "
        f"{bias('new_wh10_free')}), though single sites move. What moves the maps is the level the channel is "
        f"carved below: legacy's smoothing changes them by {change('new_legacy_smoothing')}, most at the sites of F3, "
        f"and with its two small fixes by {change('new_observed_clamp')}. The 25th percentile of the heights, "
        f"{change('new_wh25')}, and the falling fit of the banks' 25th percentile, {change('new_fq25')}, are within "
        "the noise of the new code's 10th percentile, with bigger maps; its percentile was chosen on this benchmark, "
        "from three, so it is a little flattered. The new smoothing leaves the carved bed above the stream cell at "
        "a third of the cross sections (BS5); carving every channel down to at least its stream cell, a test only "
        f"(bank-based bathymetry may raise the DEM, as decided on 2025-09-18), gives {change('new_nofill')}, bias "
        f"{bias('new_nofill')}. Letting the cross section pivot to where the water is narrowest at every increment of "
        f"its rating curve (PV1) gives {change('new_pivot')}, bias {bias('new_pivot')}; taking the direction narrowest "
        f"at the top for the whole curve, {change('new_pivot_top')}, bias {bias('new_pivot_top')}. Joseph's branch "
        f"(JG4), on the {by['joseph']['sites'] if 'joseph' in by else 0} sites it runs, gives {change('joseph')}, bias "
        f"{bias('joseph')}; his bank smoothing in the new code, {change('new_joseph_smoothing')}, bias "
        f"{bias('new_joseph_smoothing')}."),
                 stats={"new": dict(median_mcc=rounded(scores(ctx, "new")["mcc"].median(), 4),
                                    bias=rounded(scores(ctx, "new")["bias"].median(), 3)),
                        **{r["name"]: dict(mean=rounded(r["mean"], 4), interval=[rounded(r["lo"], 4), rounded(r["hi"], 4)],
                                           median=rounded(r["median"], 4), median_mcc=rounded(r["mcc"], 4),
                                           bias=rounded(r["bias"], 3), changed=rounded(r["changed"], 3),
                                           sites_better=rounded(r["sites_better"], 3),
                                           site_range=[rounded(v, 3) for v in r["spread"]],
                                           worst={s: rounded(v, 3) for s, v in r["worst"].items()},
                                           best={s: rounded(v, 3) for s, v in r["best"].items()}) for r in rows}})


MAP_CONFIGS = (("legacy", "legacy"), ("new_legacy_smoothing", "new, with legacy's bank smoothing"),
               ("new_joseph_smoothing", "new, with Joseph's bank smoothing"), ("new", "new, as it is"))


@figure("F3", "Flood maps at four sites", SECTION, needs_fim=True)
def maps(ctx):
    from matplotlib.colors import ListedColormap
    folder = ctx.fim / "results"
    sites = sorted({p.stem for p in (folder / "new" / "maps").glob("*.npz")}) if (folder / "new" / "maps").exists() \
        else []
    order = ("Flint", "East_Fork", "South_Platte", "Cuyahoga")
    sites.sort(key=lambda s: next((k for k, o in enumerate(order) if s.startswith(o)), 9))
    configs = MAP_CONFIGS
    fig, axes = plt.subplots(len(sites), len(configs), figsize=(14.5, 3.6 * len(sites)), squeeze=False)
    cmap = ListedColormap(["#eeeeee", WATER, ACCENT, PINK])  # dry, both wet, mapped only, reference only
    stats = {}
    for row, stem in enumerate(sites):
        site, stage = stem.rsplit("_", 1)
        grid = ctx.grid(site)
        panels = []
        for name, _ in configs:
            path = folder / name / "maps" / f"{stem}.npz"
            panels.append(dict(np.load(path)) if path.exists() else None)
        union = np.zeros_like(next(p for p in panels if p is not None)["reference"])
        for p in panels:
            if p is not None:
                union |= (p["flood"] | p["reference"]) & p["domain"]
        rr, cc = np.nonzero(union)
        window = Window.around(rr, cc, grid.dx, grid.dy, pad_metres=300.0, shape=union.shape)
        for column, ((name, label), p) in enumerate(zip(configs, panels)):
            ax = axes[row, column]
            if p is None:
                ax.set_visible(False)
                continue
            classes = np.full(p["flood"].shape, np.nan)
            domain = p["domain"]
            classes[domain] = 0
            classes[domain & p["flood"] & p["reference"]] = 1
            classes[domain & p["flood"] & ~p["reference"]] = 2
            classes[domain & ~p["flood"] & p["reference"]] = 3
            show_raster(ax, window, classes, cmap=cmap, vmin=-0.5, vmax=3.5)
            tp = int(np.sum(classes == 1))
            fp = int(np.sum(classes == 2))
            fn = int(np.sum(classes == 3))
            tn = int(np.sum(classes == 0))
            den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = (tp * tn - fp * fn) / den if den > 0 else np.nan
            ax.set_title(f"{label}\nMCC {mcc:.2f}, bias {(tp + fp) / max(tp + fn, 1):.2f}", fontsize=7.8)
            stats.setdefault(stem, {})[name] = dict(mcc=rounded(mcc, 3), bias=rounded((tp + fp) / max(tp + fn, 1), 3))
            if column:
                ax.set_ylabel("")
        axes[row, 0].set_ylabel(f"{site.replace('_', ' ')[:30]}\nstage {stage}\nmetres north", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WATER, ACCENT, PINK, "#eeeeee")]
    fig.legend(handles, ["wet in both", "mapped wet, the reference dry", "the reference wet, mapped dry", "dry in both"],
               loc="outside upper center", ncol=4, fontsize=8, frameon=False)
    scored = ", ".join(
        f"{stem.rsplit('_', 1)[0].split('_(')[0].split(',')[0].replace('_', ' ')} "
        + " / ".join(f"{stats[stem][name]['mcc']:.2f}" if name in stats[stem] else "–" for name, _ in configs)
        for stem in stats)
    return fig, dict(caption=(
        "The benchmark's flood maps at one stage of four sites, each drawn to scale and scored within the benchmark's "
        "domain (water land cover left out, as the benchmark scores). With legacy's bank smoothing the new code's "
        "water surfaces at Flint, East Fork White and South Platte were metres lower than legacy's, because its carve "
        "dug out incised channels below a bank elevation far under their banks (C4b, BS3), and the maps shrank. The "
        "new smoothing (BS4) carves below the DEM's water plus a low percentile of the banks' heights, and the maps "
        "grow back; Joseph's smoothing (JG1) in the new code does better than legacy's at some of these sites and "
        "worse at others (below both at East Fork White). Their MCC, " + " / ".join(label for _, label in configs) + f": {scored}. Cuyahoga is one of the "
        "sites the new code already did better on than legacy, whose map there is more than twice the reference's."),
        stats=stats)
