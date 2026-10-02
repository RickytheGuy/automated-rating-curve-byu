"""Figures comparing the overhauled ARC (arc.pipeline and the modules it uses) with legacy ARC
(Automated_Rating_Curve_Generator.main).

For each change there's a figure showing it, on small made-up cases where that shows the change most plainly and on
the real sites where it matters, and there are sanity checks of things that should be the same in both. Maps are
drawn in metres, so cells that aren't square (as on these geographic rasters) are drawn their true shape. Cross
sections are drawn between their banks, with a little to either side.

    python testing/visual_comparison/make_figures.py --out FOLDER

runs both codes on every site with a GEOGLOWS_ARC_Input_fabdem_Bathy.yaml under --sites-root (the first time; the
runs are kept under FOLDER/runs and reused), then writes the figures to FOLDER/figures and what each one shows, with
the numbers it found, to FOLDER/figures/manifest.json. --only draws just the figures whose ids start with the given
prefixes. The runs write only into FOLDER; the sites' own folders are only read. The first time takes about 15
minutes and 1.5 GB (most of it the captured intermediate results); later, --no-runs uses the runs already made.

Timing runs each code on each site in a fresh process, one at a time, so leave the machine otherwise idle for them
(about 6 minutes for the sites), or pass --no-timing.

The flood-map figures (F) need the FIM benchmark's results, which fim_benchmark.py makes (about 90 s a
configuration on 10 workers, where the thesis's nencarta, curve2flood and fim_audit harness are); --fim names its
folder (default FOLDER/fim), and without it those figures are skipped:

    python testing/visual_comparison/fim_benchmark.py --out FOLDER/fim legacy '{"code": "legacy"}' new '{"code": "new"}' ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import vc_runs  # noqa: E402

FIGURE_MODULES = ("fig_sampling", "fig_direction", "fig_hydraulics", "fig_bathymetry", "fig_outputs", "fig_pivot",
                  "fig_other", "fig_fim")
DEFAULT_DETAIL_SITES = ("Cuyahoga_River_near_Independence,_OH_(2024)", "Salt_Creek_at_Wood_Dale_(2010)",
                        "South_Fork_Peachtree_at_Casa_Dr,_nr_Clarkston_(2015)",
                        "Boise_River_at_Glenwood_Bridge_nr_Boise_(2015)", "Hohokus_Brook_at_Ho-Ho-Kus_(2014)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="the folder for the runs and the figures")
    parser.add_argument("--sites-root", type=Path, default=vc_runs.DEFAULT_SITES_ROOT)
    parser.add_argument("--sites", nargs="*", help="only these sites (default: every site with the input file)")
    parser.add_argument("--detail-sites", nargs="*", default=list(DEFAULT_DETAIL_SITES),
                        help="the sites the maps and single cross sections come from")
    parser.add_argument("--only", nargs="*", help="draw only the figures whose ids start with these")
    parser.add_argument("--no-runs", action="store_true", help="use the runs already made, and make no more")
    parser.add_argument("--no-timing", action="store_true", help="skip the timing runs and the timing figure")
    parser.add_argument("--fim", type=Path, help="the FIM benchmark's folder (fim_benchmark.py --out; default "
                        "OUT/fim), for the flood-map figures, which are skipped without it")
    parser.add_argument("--other", nargs="*", default=[], metavar="NAME=SRC",
                        help="another ARC's legacy code to run as a third code, such as a maintainer's branch: its "
                             "name and its src folder (figures JG)")
    parser.add_argument("--run-one", nargs=3, metavar=("CODE", "SITE", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--run-other", nargs=4, metavar=("NAME", "SRC", "SITE", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.run_one:
        code, site, out = args.run_one
        vc_runs.run_one(code, site, Path(out), args.sites_root)
        return
    if args.run_other:
        _, src, site, out = args.run_other
        vc_runs.run_other(Path(src), site, Path(out), args.sites_root)
        return
    others = dict(item.split("=", 1) for item in args.other)
    if args.out is None:
        parser.error("--out is required")

    import vc_plot
    sites = args.sites or vc_runs.site_names(args.sites_root)
    detail = [s for s in args.detail_sites if s in sites]
    runs = args.out / "runs"
    started = time.perf_counter()
    if not args.no_runs:
        print(f"Running {len(sites)} sites in {len(vc_runs.CONFIGS)} configurations (skipping runs already made)")
        vc_runs.ensure_runs(runs, args.sites_root, sites)
        import vc_pivot
        vc_pivot.ensure_pivot_runs(runs, args.sites_root, sites)
        for name, src in others.items():
            print(f"Running {name}'s ARC ({src}) on {len(sites)} sites")
            vc_runs.ensure_other_runs(runs, args.sites_root, sites, name, Path(src), Path(__file__).resolve())
        if not args.no_timing:
            print("Timing each code on each site, one process at a time")
            vc_runs.ensure_timings(runs, args.sites_root, sites, Path(__file__).resolve())
    print(f"Runs ready after {time.perf_counter() - started:.0f} s")

    fim = args.fim or args.out / "fim"
    context = vc_plot.Context(out=args.out, runs=runs, sites_root=args.sites_root, sites=sites, detail_sites=detail,
                              timing=not args.no_timing, fim=fim if (fim / "results").exists() else None,
                              others=others)
    for name in FIGURE_MODULES:
        __import__(name)
    manifest_path = context.figures / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() and args.only else {}
    for entry in vc_plot.REGISTRY:
        if args.only and not any(entry["id"].startswith(prefix) for prefix in args.only):
            continue
        if entry["needs_timing"] and not context.timing:
            continue
        if entry.get("needs_other") and not context.others:
            print(f"{entry['id']}: skipped, no other ARC run (--other NAME=SRC)")
            continue
        if entry["needs_fim"] and context.fim is None:
            print(f"{entry['id']}: skipped, no FIM benchmark results (fim_benchmark.py)")
            continue
        t = time.perf_counter()
        try:
            record = vc_plot.draw(entry, context)
        except Exception:
            print(f"{entry['id']}: FAILED")
            traceback.print_exc()
            continue
        manifest[entry["id"]] = record
        print(f"{entry['id']:8s} {time.perf_counter() - t:5.1f} s  {entry['title']}")
    ordered = {e["id"]: manifest[e["id"]] for e in vc_plot.REGISTRY if e["id"] in manifest}
    manifest_path.write_text(json.dumps(ordered, indent=1, default=vc_plot.json_default))
    print(f"{len(ordered)} figures in {context.figures} after {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
