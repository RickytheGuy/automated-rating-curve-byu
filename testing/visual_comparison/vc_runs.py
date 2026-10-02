"""Running legacy ARC and the new pipeline on the real sites, and saving what each worked out along the way.

Each site is run with its GEOGLOWS_ARC_Input_fabdem_Bathy.yaml, its outputs sent to this script's output folder, in
a few configurations (CONFIGS). Each configuration after the first takes one more of the differences away, so that
comparing them shows what each difference does to the rating curves. The first configuration also records each
code's intermediate results (a "capture"): the cross sections as sampled, their banks, the smoothed bank elevations,
depths and beds, and the carved cross sections.

Legacy ARC keeps its state in module globals, so its intermediate results are read by wrapping the function that
finishes its cross sections (_finalize_cross_section_records). The new pipeline's are read by wrapping the functions
arc.pipeline calls.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

YAML_NAME = "GEOGLOWS_ARC_Input_fabdem_Bathy.yaml"
DEFAULT_SITES_ROOT = Path(r"C:\Users\lrr43\Documents\masters\fim_sites_nencarta")
# curve2flood's keys, and a mask neither ARC reads, which would otherwise point at the site's own folders
DROPPED_KEYS = ("FSOutBATHY", "BathyWaterMask", "Comid_Flow_File")
NO_BATHYMETRY = {"BATHY_Out_File": None, "AROutBATHY": None}

# name: (overrides of the site's inputs, the outputs to write, the codes whose intermediate results to capture)
CONFIGS = {
    "as_configured": ({}, ("vdt", "bathy", "xs"), ("legacy", "new")),
    "no_bathymetry": (NO_BATHYMETRY, ("vdt",), ()),
    "no_angle_search": ({**NO_BATHYMETRY, "Degree_Manip": 0}, ("vdt",), ()),
    "constant_n": ({**NO_BATHYMETRY, "Degree_Manip": 0, "shallow_factor": 1.0, "deep_factor": 1.0}, ("vdt",),
                   ("legacy",)),
    # every land cover class the same n, so the codes' conventions for which ordinate's n a segment takes agree
    "uniform_n": ({**NO_BATHYMETRY, "Degree_Manip": 0, "shallow_factor": 1.0, "deep_factor": 1.0,
                   "LU_Manning_n": ("uniform", 0.035)}, ("vdt",), ("legacy",)),
}
CODES = ("legacy", "new")


def quiet_logs() -> None:
    from arc import LOG
    LOG.setLevel(logging.ERROR)


def site_names(sites_root: Path) -> list[str]:
    return sorted(p.name for p in sites_root.iterdir() if (p / "ARC_InputFiles" / YAML_NAME).exists())


def site_inputs(sites_root: Path, site: str, out_dir: Path, overrides: dict, outputs) -> dict:
    """The site's ARC inputs with its outputs sent to out_dir, only those named in outputs, and the overrides (a
    value of None removes the key)."""
    with open(sites_root / site / "ARC_InputFiles" / YAML_NAME) as f:
        inputs = yaml.safe_load(f)
    for key in DROPPED_KEYS + ("Print_Curve_File", "Print_AP_Database", "XS_Out_File", "Print_VDT_Database"):
        inputs.pop(key, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in overrides.items():
        if value is None:
            inputs.pop(key, None)
        elif isinstance(value, tuple) and value[0] == "uniform":
            # a Manning's n table giving every land cover class (0 to 255) the same n
            path = out_dir / "manning_uniform.txt"
            lines = ["LC_ID\tDescription\tManning_n"] + [f"{c}\tclass {c}\t{value[1]}" for c in range(256)]
            path.write_text("\n".join(lines) + "\n")
            inputs[key] = str(path)
        else:
            inputs[key] = value
    if "vdt" in outputs:
        inputs["Print_VDT_Database"] = str(out_dir / "vdt.parquet")
    if "xs" in outputs:
        inputs["XS_Out_File"] = str(out_dir / "xs.txt")
    if "bathy" in outputs and "BATHY_Out_File" in inputs:
        inputs["BATHY_Out_File"] = inputs["AROutBATHY"] = str(out_dir / "bathy.tif")
    else:
        inputs.pop("BATHY_Out_File", None)
        inputs.pop("AROutBATHY", None)
    return inputs


# --- Legacy ----------------------------------------------------------------------------------------------------------

_LEGACY_BANK_KEYS = ("function_used", "i_bank_1_index", "i_bank_2_index", "i_total_bank_cells", "bank_elev_1",
                     "bank_elev_2", "is_valid", "smoothed_bank_elevation", "raw_bank_elevation",
                     "along_stream_coordinate", "reach_order_index", "bathymetry_depth",
                     "bathymetry_depth_raw_un-smoothed", "smoothed_bed_elevation", "bathymetry_should_apply",
                     "bathymetry_depth_source", "reach_top_width_filter_applied",
                     "reach_top_width_filter_median_top_width", "reach_top_width_filter_q25",
                     "reach_top_width_filter_q75", "reach_top_width_filter_observed_top_width",
                     "reach_top_width_filter_final_top_width", "reach_top_width_filter_reach_id",
                     "local_function_used", "local_i_bank_1_index", "local_i_bank_2_index",
                     "local_i_total_bank_cells", "observed_cell_minimum_bank_elevation",
                     "reach_median_bank_fill_applied")


def _bank_subset(result) -> dict:
    if not isinstance(result, dict):
        return {}
    return {key: copy.copy(result[key]) for key in _LEGACY_BANK_KEYS if key in result}


def run_legacy(inputs: dict, capture: bool) -> tuple[float, dict | None]:
    """Run legacy ARC, returning its time and, if asked, its intermediate results."""
    import arc.Automated_Rating_Curve_Generator as legacy
    quiet_logs()
    store: dict = {}
    original = legacy._finalize_cross_section_records

    def spy(sampled_records, params, quiet, collect_cross_section_data=False):
        pad = int(params["i_boundary_number"])
        offset = 100.0 if params.get("b_modified_dem") else 0.0
        found = [None if r is None else (r["xs1_profile"].astype(np.float64) - offset,
                                         r["xs2_profile"].astype(np.float64) - offset,
                                         _bank_subset(r.get("bank_search_result"))) for r in sampled_records]
        result = original(sampled_records, params, quiet, collect_cross_section_data)
        finals = result[1]
        n_raster = legacy._MANNINGS_N
        cells = []
        for i, final in enumerate(finals):
            row, col = int(legacy._CELL_ROWS[i]), int(legacy._CELL_COLS[i])
            if final is None:
                cells.append(None)
                continue
            baseflow, slope, target_depth, target_width = legacy._get_cell_bathymetry_inputs(i, row, col, params)
            banks = _bank_subset(sampled_records[i].get("bank_search_result"))
            n1 = n_raster[final["xs1_row"], final["xs1_col"]].astype(np.float64)
            n2 = n_raster[final["xs2_row"], final["xs2_col"]].astype(np.float64)
            left, right = int(banks.get("i_bank_1_index", -1)), int(banks.get("i_bank_2_index", -1))
            hydraulic = (left, right) if banks.get("is_valid") and 0 < left < n1.size and 0 < right < n2.size \
                else (-1, -1)
            _, initial_xs = legacy.get_stream_direction_information(row, col, legacy._STREAMS,
                                                                     params["i_general_direction_distance"])
            cells.append(dict(
                row=row - pad, col=col - pad, comid=int(legacy._CELL_COMIDS[i]),
                center=(int(final["row"]) - pad, int(final["col"]) - pad),
                xs_angle=float(final["xs_angle"]), initial_xs_direction=float(initial_xs),
                spacing=float(final["ordinate_dist"]),
                found=found[i][:2], found_banks=found[i][2], banks=banks, hydraulic_banks=hydraulic,
                final=(final["xs1_profile"].astype(np.float64) - offset,
                       final["xs2_profile"].astype(np.float64) - offset),
                cells1=(final["xs1_row"] - pad, final["xs1_col"] - pad),
                cells2=(final["xs2_row"] - pad, final["xs2_col"] - pad),
                n=(n1, n2), baseflow=float(baseflow), slope=float(slope),
                target_depth=math.nan if target_depth is None else float(target_depth),
                target_width=math.nan if target_width is None else float(target_width),
                qmax=float(legacy._CELL_QMAX[i]),
                slope_bounds=(float(legacy._CELL_SLOPE_25[i]), float(legacy._CELL_SLOPE_75[i]))
                if legacy._CELL_SLOPE_25 is not None else (math.nan, math.nan)))
        store.update(cells=cells, pad=pad, dx=float(params["dx"]), dy=float(params["dy"]),
                     modified_dem=bool(params.get("b_modified_dem")),
                     index_arrays=np.array(legacy._INDEX_ARRAYS).copy(),
                     distances=np.array(legacy._Z_DISTANCE_ARRAY).copy(),
                     fractions=np.array(legacy._INDEX_FRACT_ARRAYS).copy())
        return result

    if capture:
        legacy._finalize_cross_section_records = spy
    try:
        started = time.perf_counter()
        legacy.main("", inputs, quiet=True, processes=1)
        seconds = time.perf_counter() - started
    finally:
        legacy._finalize_cross_section_records = original
    return seconds, (store if capture else None)


# --- New -------------------------------------------------------------------------------------------------------------


def run_new(inputs: dict, capture: bool) -> tuple[float, dict | None]:
    """Run the new pipeline, returning its time and, if asked, its intermediate results."""
    from arc import pipeline
    from arc.config import Configs
    quiet_logs()
    configs = Configs.from_mapping(inputs)
    if not capture:
        started = time.perf_counter()
        pipeline.run(configs, quiet=True)
        return time.perf_counter() - started, None

    store: dict = {"initial": {}, "carve": {}}
    originals = {name: getattr(pipeline, name) for name in
                 ("stream_direction", "apply_bathymetry", "smooth_bank_elevations", "smooth_channel_depths",
                  "carve_channel")}

    def stream_direction(streams, row, col, distance, dx, dy):
        direction = originals["stream_direction"](streams, row, col, distance, dx, dy)
        store["initial"][(int(row), int(col))] = float(direction)
        return direction

    def apply_bathymetry(configs, grid, cells, sections, network, slopes, baseflow, target_depth, target_width,
                         water_n):
        store["target_depth"], store["target_width"] = np.array(target_depth), np.array(target_width)
        return originals["apply_bathymetry"](configs, grid, cells, sections, network, slopes, baseflow,
                                             target_depth, target_width, water_n)

    def smooth_bank_elevations(network, reaches, dx, dy):
        result = originals["smooth_bank_elevations"](network, reaches, dx, dy)
        store["reaches"] = {reach: (np.array(r.rows), np.array(r.cols), list(r.banks)) for reach, r in reaches.items()}
        store["smoothed"] = dict(result)
        return result

    def smooth_channel_depths(network, reaches, smoothed, depths, dx, dy, **kwargs):
        result = originals["smooth_channel_depths"](network, reaches, smoothed, depths, dx, dy, **kwargs)
        store["depths"] = {reach: np.array(values, dtype=np.float64) for reach, values in depths.items()}
        store["channels"] = dict(result)
        return result

    def carve_channel(xs, banks, depth, **kwargs):
        store["carve"][id(xs)] = (xs.elevations.copy(), float(depth))
        return originals["carve_channel"](xs, banks, depth, **kwargs)

    for name, wrapper in (("stream_direction", stream_direction), ("apply_bathymetry", apply_bathymetry),
                          ("smooth_bank_elevations", smooth_bank_elevations),
                          ("smooth_channel_depths", smooth_channel_depths), ("carve_channel", carve_channel)):
        setattr(pipeline, name, wrapper)
    try:
        started = time.perf_counter()
        results = pipeline.run(configs, quiet=True)
        seconds = time.perf_counter() - started
    finally:
        for name, function in originals.items():
            setattr(pipeline, name, function)

    grid = pipeline.read_grid(configs)
    cells = []
    for k, section in enumerate(results.sections):
        row, col = int(results.cells.rows[k]), int(results.cells.cols[k])
        if section is None:
            cells.append(None)
            continue
        carved = store["carve"].get(id(section.xs))
        profile = section.xs.profile
        cells.append(dict(
            row=row, col=col, comid=int(results.cells.comids[k]), reach=int(results.cells.reaches[k]),
            center=(section.row, section.col), direction=section.direction, xs_angle=section.xs_angle,
            initial_direction=store["initial"].get((row, col), math.nan), spacing=float(section.xs.ordinate_distance),
            side_one=section.side_one,
            found=(section.xs.elevations.copy() if carved is None else carved[0]).astype(np.float64),
            final=section.xs.elevations.astype(np.float64).copy(), n=np.asarray(section.xs.mannings_n, np.float64),
            profile=None if profile is None else tuple(np.array(a) if isinstance(a, np.ndarray) else a
                                                       for a in profile),
            banks=section.banks, hydraulic_banks=section.hydraulic_banks, bank_elevation=section.bank_elevation,
            usable=section.usable, carve_depth=math.nan if carved is None else carved[1],
            ordinate_rows=np.array(section.rows), ordinate_cols=np.array(section.cols),
            slope=float(results.slopes[k]), baseflow=float(results.baseflow[k]), qmax=float(results.max_flow[k]),
            target_depth=float(store["target_depth"][k]) if "target_depth" in store else math.nan,
            target_width=float(store["target_width"][k]) if "target_width" in store else math.nan,
            increments=None if results.curves is None else results.curves.increments[k].copy()))
    by_reach: dict[int, list[int]] = {}
    for k, section in enumerate(results.sections):
        if section is not None:
            by_reach.setdefault(int(results.cells.reaches[k]), []).append(k)
    reaches = {}
    for reach, ks in by_reach.items():
        smoothed = store.get("smoothed", {}).get(reach)
        channel = store.get("channels", {}).get(reach)
        found = store.get("reaches", {}).get(reach)
        reaches[reach] = dict(
            cells=ks,
            found_banks=None if found is None else found[2],
            smoothed=None if smoothed is None else smoothed._asdict(),
            depths=store.get("depths", {}).get(reach),
            channel=None if channel is None else channel._asdict())
    return seconds, dict(cells=cells, reaches=reaches, dx=grid.dx, dy=grid.dy, shape=grid.dem.shape,
                         bathy_bed_cap=configs.bathy_bed_cap, bathy_use_banks=configs.bathy_use_banks,
                         trapezoid_height=configs.bathy_trap_h)


# --- Running every site ----------------------------------------------------------------------------------------------


def run_directory(runs: Path, config: str, code: str, site: str) -> Path:
    return runs / config / code / site


def ensure_runs(runs: Path, sites_root: Path, sites: list[str], configs=CONFIGS, codes=CODES, log=print) -> None:
    """Run each site in each configuration with each code, unless it already has been."""
    for config, (overrides, outputs, capture) in configs.items():
        for code in codes:
            for site in sites:
                out = run_directory(runs, config, code, site)
                done = out / "done.json"
                if done.exists():
                    continue
                inputs = site_inputs(sites_root, site, out, overrides, outputs)
                try:
                    seconds, captured = (run_legacy if code == "legacy" else run_new)(inputs, code in capture)
                except Exception as error:  # a failed site is left out of the figures, and said so
                    log(f"  {config} {code} {site}: FAILED {type(error).__name__}: {error}")
                    (out / "failed.txt").write_text(f"{type(error).__name__}: {error}\n")
                    continue
                if captured is not None:
                    with open(out / "capture.pkl", "wb") as f:
                        pickle.dump(captured, f, protocol=pickle.HIGHEST_PROTOCOL)
                done.write_text(json.dumps({"seconds": seconds}))
                log(f"  {config} {code} {site}: {seconds:.1f} s")


def ensure_timings(runs: Path, sites_root: Path, sites: list[str], script: Path, log=print) -> None:
    """Time each code on each site in a fresh process of its own, as a user would run it, with only the outputs
    the site's input file asks for (its VDT database and bathymetry). One site at a time, so the times are fair."""
    for site in sites:
        for code in CODES:
            out = run_directory(runs, "timing", code, site)
            if (out / "done.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            process = subprocess.run([sys.executable, str(script), "--run-one", code, site, str(out),
                                      "--sites-root", str(sites_root)], capture_output=True, text=True)
            if process.returncode != 0:
                log(f"  timing {code} {site}: FAILED\n{process.stderr[-2000:]}")
                (out / "failed.txt").write_text(process.stderr[-5000:])
                continue
            log(f"  timing {code} {site}: {json.loads((out / 'done.json').read_text())['seconds']:.1f} s")


def run_one(code: str, site: str, out: Path, sites_root: Path) -> None:
    """The --run-one entry point: one timed run in this process (see ensure_timings)."""
    inputs = site_inputs(sites_root, site, out, {}, ("vdt", "bathy"))
    seconds, _ = (run_legacy if code == "legacy" else run_new)(inputs, False)
    (out / "done.json").write_text(json.dumps({"seconds": seconds}))


def ensure_other_runs(runs: Path, sites_root: Path, sites: list[str], name: str, src: Path, script: Path,
                      log=print) -> None:
    """Another ARC's legacy code, such as a maintainer's branch (src is its src folder), on every site as configured,
    captured as legacy's is, into runs/as_configured/<name>: each site in a process of its own with src ahead of this
    repository's (run_other). A site it fails on says why in failed.txt, and isn't tried again."""
    for site in sites:
        out = run_directory(runs, "as_configured", name, site)
        if (out / "done.json").exists() or (out / "failed.txt").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        process = subprocess.run([sys.executable, str(script), "--run-other", name, str(src), site, str(out),
                                  "--sites-root", str(sites_root)], capture_output=True, text=True)
        if process.returncode != 0 or not (out / "done.json").exists():
            reason = (process.stderr.strip().splitlines() or ["no error message"])[-1]
            (out / "failed.txt").write_text(reason + "\n")
            log(f"  as_configured {name} {site}: FAILED {reason}")
            continue
        log(f"  as_configured {name} {site}: {json.loads((out / 'done.json').read_text())['seconds']:.1f} s")


def run_other(src: Path, site: str, out: Path, sites_root: Path) -> None:
    """The --run-other entry point: run_legacy, captured, with another ARC's src ahead of this repository's."""
    sys.path.insert(0, str(src))
    import arc
    if Path(src).resolve() not in Path(arc.__file__).resolve().parents:
        raise RuntimeError(f"arc was imported from {arc.__file__}, not from {src}")
    inputs = site_inputs(sites_root, site, out, {}, ("vdt", "bathy", "xs"))
    seconds, captured = run_legacy(inputs, True)
    with open(out / "capture.pkl", "wb") as f:
        pickle.dump(captured, f, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "done.json").write_text(json.dumps({"seconds": seconds, "arc": str(arc.__file__)}))


def load_capture(runs: Path, config: str, code: str, site: str) -> dict | None:
    path = run_directory(runs, config, code, site) / "capture.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
