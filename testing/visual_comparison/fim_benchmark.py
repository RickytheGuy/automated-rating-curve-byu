"""Score configurations of ARC on the FIM benchmark: flood maps against the USGS reference extents.

The benchmark is the thesis's: 51 sites, each with reference inundation at several stages (814 site/stage pairs),
built by nencarta's pipeline (ARC, curve2flood's bathymetry pass, the FLDPLN library and maps) and scored by
fim_audit's harness (MCC, CSI and frequency bias, the harness reproducing the pipeline's own maps bit for bit). For
each configuration a copy of the benchmark's output tree has its ARC-dependent products purged and rebuilt with
nencarta, as fim_audit/rerun_arc.py does, but with ARC swapped for the chosen code: legacy, or the new arc.pipeline
with the settings under test (vc_variants). Then every stage of every site is scored.

    python fim_benchmark.py --out FOLDER NAME SETTINGS [NAME SETTINGS ...]
    e.g. python fim_benchmark.py --out runs/fim legacy '{"code": "legacy"}' new '{"code": "new"}'

SETTINGS is JSON: "code" (legacy or new), any vc_variants settings, "yaml" (changes to the ARC input file nencarta
writes, for either code; the mapper reads the same file) and "objective" (the benchmark's own switches). Each
configuration's scores, VDTs and ARC bathymetry go to FOLDER/results/NAME/, and a configuration already scored is
skipped. The benchmark's tree is only read: it is copied to FOLDER/tree_template once (1.1 GB). Each configuration
takes about 90 s on 10 workers. ARC's roughness defaults are used (rerun_arc's arguments pass 1s for them, which
nencarta now forwards, so those keys are dropped), as the benchmark's relaxed tree was built.

It needs the thesis's nencarta, curve2flood and the fim_audit harness, so it runs only where they are.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")  # before numpy: nencarta's BLAS calls die without it

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = Path(r"C:\Users\lrr43\Documents\masters\fim_sites_nencarta")
AUDIT = Path(r"C:\Users\lrr43\Documents\masters\fim_audit")
THESIS = Path(r"C:\Users\lrr43\Documents\masters\thesis")
DEMS = r"C:\Users\lrr43\Documents\masters\fim_sites\*\DEMs\fabdem.tif"
PURGE = ("*/VDT/*.parquet", "*/VDT/*.csv", "*/Bathymetry/*.tif", "*/FloodMap/*.tif", "*/ARC_InputFiles/*")
ROUGHNESS_KEYS = ("shallow_factor", "deep_factor", "k_decay", "slope_adjustment_factor")
MAPS = (("Flint_River_at_Albany_(2007)", "35"), ("East_Fork_White_River_near_Bedford_(2014)", "30"),
        ("South_Platte_River_at_Fort_Morgan_(2017)", "20"), ("Cuyahoga_River_near_Independence,_OH_(2024)", "20"))
# every configuration the flood-map figures (fig_fim.py) use, for --all. "new" is the code as it is: since
# 2026-09-26 the angle search 5 m deep, the bed cap at MAX_SLOPE, and the bank elevation the water surface plus the
# banks' 10th-percentile height; the rest change one thing from it.
CONFIGURATIONS = {
    "legacy": {"code": "legacy"},
    "new": {"code": "new"},
    "legacy_nosearch": {"code": "legacy", "yaml": {"Degree_Manip": 0}},
    "legacy_wse": {"code": "legacy", "yaml": {"Bathy_Use_Banks": 0}},
    "new_td05": {"code": "new", "test_depth": 0.5},
    "new_td1": {"code": "new", "test_depth": 1.0},
    "new_td2": {"code": "new", "test_depth": 2.0},
    "new_td10": {"code": "new", "test_depth": 10.0},
    "new_nosearch": {"code": "new", "yaml": {"Degree_Manip": 0}},
    "new_cap001": {"code": "new", "bed_grade": 0.01},
    "new_nocap": {"code": "new", "bed_grade": None},
    # the bank elevation: legacy's smoothing, alone and with fixes, and the other smoothings tried
    "new_legacy_smoothing": {"code": "new", "bank_smoothing": "legacy"},
    "new_observed_clamp": {"code": "new", "bank_ceiling": "observed", "bank_reference": "clamp"},
    "new_local": {"code": "new", "bank_reference": "local"},
    "new_fq25": {"code": "new", "bank_reference": "falling_quantile", "bank_quantile": 0.25},
    "new_wh25": {"code": "new", "bank_reference": "water_plus_height", "bank_quantile": 0.25},
    "new_wh10_bankfree": {"code": "new", "bank_reference": "water_plus_height", "bank_quantile": 0.1,
                          "bank_falling": "water"},
    "new_wh10_free": {"code": "new", "bank_reference": "water_plus_height", "bank_quantile": 0.1,
                      "bank_falling": "none"},
    "new_wse": {"code": "new", "yaml": {"Bathy_Use_Banks": 0}},
    # a test of the fills the new smoothing makes, not a proposal: no bed above the stream cell
    "new_nofill": {"code": "new", "bed_at_most_stream": True},
    # the cross section pivoting as the rating curve rises (vc_pivot): at every increment, or once at the top
    "new_pivot": {"code": "new", "pivot": "each"},
    "new_pivot_top": {"code": "new", "pivot": "top"},
}


# --- ARC, as the worker processes run it -----------------------------------------------------------------------------


def run_arc(config_path: str, settings: dict, quiet: bool = True) -> None:
    """Run the chosen ARC on a nencarta-written ARC input file."""
    Path(config_path + ".arc_code").write_text(json.dumps(settings))  # which ARC wrote the outputs
    if settings.get("yaml"):
        import yaml
        with open(config_path) as f:
            inputs = yaml.safe_load(f)
        for key, value in settings["yaml"].items():
            if value is None:
                inputs.pop(key, None)
            else:
                inputs[key] = value
        with open(config_path, "w") as f:
            yaml.safe_dump(inputs, f, sort_keys=False)
    if settings.get("code", "new") == "legacy":
        from arc import Arc
        Arc(config_path, quiet=quiet).run()
        return
    import vc_variants
    from arc import pipeline
    from arc.config import Configs
    configs = Configs.from_file(config_path)
    with vc_variants.applied(settings, configs):
        pipeline.run(configs, quiet=quiet)


def _worker_init(settings_json: str) -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    settings = json.loads(settings_json)
    import nencarta.tasks.run_models as run_models

    def _run_arc(config, model_config):
        run_arc(str(config), settings, quiet=True)

    run_models._run_arc = _run_arc


# --- Scoring ---------------------------------------------------------------------------------------------------------


def _harness():
    if str(AUDIT) not in sys.path:
        sys.path.insert(0, str(AUDIT))
    import harness  # reads FIM_OUT at import, from the parent's environment
    return harness


def _score_site(site: str) -> list[dict]:
    H = _harness()
    if not H.has_outputs(site):
        return [dict(site=site, stage=None, error="missing outputs")]
    rows = []
    for stage in H.stages_for(site):
        if H.reference_for(site, stage) is None:
            continue
        try:
            rows.append(dict(site=site, stage=str(stage), **H.score(site, stage, H.flood_map(site, stage, 0.0))))
        except Exception as error:  # noqa: BLE001 - one bad stage shouldn't lose the site
            rows.append(dict(site=site, stage=str(stage), error=f"{type(error).__name__}: {error}"))
    return rows


def _save_map(site: str, stage: str, path: Path) -> None:
    import numpy as np
    from osgeo import gdal
    H = _harness()
    if not H.has_outputs(site):
        return
    st = H.load_static(site)
    reference = H.reference_for(site, stage)
    if reference is None:
        return
    np.savez_compressed(path, flood=H.flood_map(site, stage, 0.0), reference=gdal.Open(str(reference)).ReadAsArray() > 0,
                        domain=st["domain"], dem=st["dem"])


# --- One configuration -----------------------------------------------------------------------------------------------


def prepare(out: Path, tree: Path) -> None:
    template = out / "tree_template"
    if not template.exists():
        print(f"copying {SOURCE} to {template} (once)", flush=True)
        shutil.copytree(SOURCE, template)
    if not tree.exists():
        shutil.copytree(template, tree)
    for pattern in PURGE:
        for f in tree.glob(pattern):
            f.unlink()


def evaluate(out: Path, name: str, settings: dict, workers: int = 10, dem_glob: str = DEMS, maps=MAPS) -> Path:
    import pandas as pd
    out = out.resolve()  # the objective runs from inside it
    results = out / "results" / name
    if (results / "scores.csv").exists():
        print(f"{name}: already scored", flush=True)
        return results
    results.mkdir(parents=True, exist_ok=True)
    (results / "settings.json").write_text(json.dumps(settings, indent=1))
    tree = out / "tree_work"
    prepare(out, tree)

    sys.path.insert(0, str(THESIS))
    sys.path.insert(0, str(AUDIT))
    from optimieze_linux_based import _objective
    from rerun_arc import ARGS
    args = dict(ARGS)
    args["bathy_args"] = {k: v for k, v in ARGS["bathy_args"].items() if k not in ROUGHNESS_KEYS}
    args.update(settings.get("objective", {}))

    started = time.time()
    # the objective writes its own summary, <tree>.csv, into the working directory: keep it in the work folder
    with (contextlib.chdir(out),
          ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(json.dumps(settings),)) as pool):
        objective = _objective(output_dir=str(tree), executor=pool, dem_glob=dem_glob, **args)
    built = time.time() - started

    os.environ["FIM_OUT"] = str(tree)
    sites = sorted(p.name for p in tree.iterdir() if p.is_dir())
    rows = []
    started = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(_score_site, s) for s in sites]):
            rows.extend(future.result())
        (results / "maps").mkdir(exist_ok=True)
        list(pool.map(_save_map, [m[0] for m in maps], [m[1] for m in maps],
                      [results / "maps" / f"{m[0]}_{m[1]}.npz" for m in maps]))
    scored = time.time() - started
    df = pd.DataFrame(rows)
    ok = df[df["error"].isna()] if "error" in df else df
    print(f"{name}: built in {built:.0f} s (objective {objective}), scored {len(ok)} pairs in {scored:.0f} s; "
          f"median MCC {ok['mcc'].median():.4f}, CSI {ok['csi'].median():.4f}, bias {ok['bias'].median():.3f}",
          flush=True)
    for folder, pattern in (("vdt", "VDT/GEOGLOWS_*_VDT_Database_Bathy.parquet"),
                            ("bathymetry", "Bathymetry/GEOGLOWS_*_ARC_Bathy.tif")):
        (results / folder).mkdir(exist_ok=True)
        for site in sites:
            for f in (tree / site).glob(pattern):
                shutil.copy2(f, results / folder / f"{site}{f.suffix}")
    (results / "timing.json").write_text(json.dumps(dict(build_seconds=built, score_seconds=scored)))
    df.to_csv(results / "scores.csv", index=False)  # last, so an interrupted run is redone
    return results


def save_maps(out: Path, name: str, settings: dict, workers: int = 4, maps=MAPS) -> None:
    """Rebuild just the MAPS sites with a configuration and keep their flood maps, for the figures (the whole
    benchmark's run keeps them too)."""
    out = out.resolve()
    results = out / "results" / name / "maps"
    results.mkdir(parents=True, exist_ok=True)
    tree = out / "tree_maps"
    prepare(out, tree)
    sys.path.insert(0, str(THESIS))
    sys.path.insert(0, str(AUDIT))
    from optimieze_linux_based import _objective
    from rerun_arc import ARGS
    args = dict(ARGS)
    args["bathy_args"] = {k: v for k, v in ARGS["bathy_args"].items() if k not in ROUGHNESS_KEYS}
    args.update(settings.get("objective", {}))
    for site, _ in maps:
        with (contextlib.chdir(out),
              ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(json.dumps(settings),)) as pool):
            _objective(output_dir=str(tree), executor=pool, dem_glob=DEMS.replace("*", site, 1), **args)
    os.environ["FIM_OUT"] = str(tree)
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_save_map, [m[0] for m in maps], [m[1] for m in maps],
                      [results / f"{m[0]}_{m[1]}.npz" for m in maps]))
    print(f"{name}: maps kept for {len(maps)} sites", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the FIM work folder")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dems", default=DEMS, help="a glob of the sites' DEMs, to run fewer")
    parser.add_argument("--maps-only", action="store_true", help="only rebuild the mapped sites and keep their maps")
    parser.add_argument("--all", action="store_true", help="every configuration the figures use (CONFIGURATIONS)")
    parser.add_argument("runs", nargs="*", help="NAME SETTINGS pairs")
    args = parser.parse_args(argv)
    if len(args.runs) % 2 or not (args.runs or args.all):
        parser.error("give NAME SETTINGS pairs, or --all")
    runs = [(name, json.loads(settings)) for name, settings in zip(args.runs[::2], args.runs[1::2])]
    if args.all:
        runs += list(CONFIGURATIONS.items())
    for name, settings in runs:
        if args.maps_only:
            save_maps(args.out, name, settings)
        else:
            evaluate(args.out, name, settings, args.workers, args.dems)


if __name__ == "__main__":
    main()
