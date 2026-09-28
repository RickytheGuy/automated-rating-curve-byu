"""The cross section pivoting as its rating curve rises: an experiment, not in the source.

The pipeline turns each cross section once, to the candidate direction whose water is narrowest 5 m above the stream
cell (arc.xsection.orientation), and takes the whole rating curve on it. Here the cross section can pivot at every
water surface the rating curve samples: at each increment it takes the candidate whose water is narrowest at that
increment's water surface, over the angle search's candidates (the stream's direction turned by angle_offsets), and
the increment's discharge, velocity and top width are that candidate's. The top, the lowest water surface carrying
the maximum flow, is the lowest at which the cross section pivoted there carries it.

Each candidate is the cross section the pipeline would have made in that direction: sampled from the DEM and from the
Manning's n raster (with the water's n between every cross section's banks), with the water's n between the fixed
cross section's banks, carved with the fixed cross section's channel (its banks as distances, its depth after the bed
smoothing and its bank elevation), and divided at the same banks. So up to the bank elevation every candidate holds
the same channel, and the fixed direction is kept (of candidates equally narrow, it wins); above it the cross section
turns to where the floodplain is narrowest. The widths are compared as the angle search compares them, each side no
further out than every candidate reaches.

mode "each" pivots at every increment; mode "top" chooses once, the direction narrowest at the pivoted top, and takes
the whole rating curve on it. Candidate 0 is always the fixed direction, and a rating curve pivoting over it alone is
the pipeline's own, bit for bit (checked on every cell by run_pivot).
"""
from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numba import njit

from arc.hydraulics import _check_slope_factor, _parameters, hydraulic_profile, wse_for_profile_conveyance
from arc.rating_curve import (ACCEPTABLE, BASEFLOW_MARGIN, DISCHARGE_CAP, FIX_UP_STEP, P, Q, T, V, WSE, RatingCurve,
                              _increments, rating_curve, section_hydraulics)
from arc.xsection.orientation import _SAME_DIRECTION, _reach, angle_offsets, stream_direction
from arc.xsection.sampling import sample_elevations
from arc.xsection.xsection import XSection

TIE = 1e-6  # metres: a candidate must be narrower than this to replace one before it
MODES = ("each", "top")


class Candidates(NamedTuple):
    """A cell's candidate cross sections, the fixed one first: their offsets from the stream's direction, their
    profiles padded into arrays (sizes gives each one's length), the banks they're divided at, and how far each
    reaches on each side."""
    offsets: np.ndarray
    stations: np.ndarray
    elevations: np.ndarray
    mannings_n: np.ndarray
    sizes: np.ndarray
    centers: np.ndarray
    left_bank: float
    right_bank: float
    left_reach: np.ndarray
    right_reach: np.ndarray


# --- The kernels ---------------------------------------------------------------------------------------------------


@njit(cache=False, error_model="numpy")
def _side_width(stations, elevations, center, step, wse):
    """The distance along a profile from its centre out to one water edge, given water above the centre. The water
    passes ground no higher than it, as wse_for_profile_conveyance's does, so at a high point's own elevation a
    cross section that spills there is as wide as the water beyond it."""
    end = elevations.size - 1 if step > 0 else 0
    origin = stations[center]
    j = center
    while j != end:
        z_out = elevations[j + step]
        if z_out > wse:
            z_in = elevations[j]
            return abs(stations[j] - origin) + (wse - z_in) / (z_out - z_in) * abs(stations[j + step] - stations[j])
        j += step
    return abs(stations[end] - origin)


@njit(cache=False, error_model="numpy")
def _choose(stations, elevations, sizes, centers, left_reach, right_reach, wse):
    """The candidate whose water is narrowest at a water surface elevation, each side no further out than every
    candidate reaches; of candidates within TIE of each other, the first."""
    left_limit, right_limit = left_reach.min(), right_reach.min()
    best, narrowest = 0, np.inf
    for j in range(sizes.size):
        s, e, c = stations[j, :sizes[j]], elevations[j, :sizes[j]], centers[j]
        width = 0.0
        if wse > e[c]:
            width = min(_side_width(s, e, c, -1, wse), left_limit) + min(_side_width(s, e, c, 1, wse), right_limit)
        if width < narrowest - TIE:
            best, narrowest = j, width
    return best


@njit(cache=False, error_model="numpy")
def _hydraulics(stations, elevations, mannings_n, sizes, centers, left_bank, right_bank, j, wse, k_decay,
                shallow_factor, deep_factor):
    size = sizes[j]
    return section_hydraulics(stations[j, :size], elevations[j, :size], mannings_n[j, :size], centers[j], left_bank,
                              right_bank, wse, k_decay, shallow_factor, deep_factor)


@njit(cache=False, error_model="numpy")
def _pivot_top(stations, elevations, mannings_n, sizes, centers, left_bank, right_bank, left_reach, right_reach,
               target, k_decay, shallow_factor, deep_factor):
    """The lowest water surface at or above the top of the candidate narrowest there, and that candidate: (NaN, -1)
    if there is none. A candidate's top is the lowest water surface at which it carries the target conveyance, or
    the high point it spills over there (wse_for_profile_conveyance, as the pipeline's). The pivoted top lies at a
    candidate's top, or between two, where the choice changes, found by bisection."""
    m = sizes.size
    tops = np.empty(m)
    keys = np.empty(m)
    for j in range(m):
        size = sizes[j]
        tops[j] = wse_for_profile_conveyance(stations[j, :size], elevations[j, :size], mannings_n[j, :size],
                                             centers[j], left_bank, right_bank, target, k_decay, shallow_factor,
                                             deep_factor)
        keys[j] = tops[j] if math.isfinite(tops[j]) else np.inf
    below = -np.inf
    for index in np.argsort(keys):
        w = tops[index]
        if not math.isfinite(w):
            break
        j = _choose(stations, elevations, sizes, centers, left_reach, right_reach, w)
        if tops[j] <= w:
            if not math.isfinite(below):
                return w, j
            lo, hi, j_hi = below, w, j
            for _ in range(60):
                if hi - lo <= 1e-9:
                    break
                mid = 0.5 * (lo + hi)
                j_mid = _choose(stations, elevations, sizes, centers, left_reach, right_reach, mid)
                if tops[j_mid] <= mid:
                    hi, j_hi = mid, j_mid
                else:
                    lo = mid
            return hi, j_hi
        below = w
    return math.nan, -1


@njit(cache=False, error_model="numpy")
def _pivot_increments(stations, elevations, mannings_n, sizes, centers, left_bank, right_bank, left_reach,
                      right_reach, k_decay, shallow_factor, deep_factor, thalweg, max_wse, count, discharge_scale,
                      q_top, out, chosen):
    """arc.rating_curve._increments with the cross section pivoting at every water surface it tries, the fix-up's
    included: fills out and chosen (each increment's candidate), and returns start and last as it does."""
    step = (max_wse - thalweg) / count
    start = 0
    last = 0
    prev_t = prev_a = prev_p = prev_q = prev_v = prev_wse = 0.0
    prev_j = 0
    for i in range(1, count + 1):
        wse = thalweg + step * i
        j = _choose(stations, elevations, sizes, centers, left_reach, right_reach, wse)
        area, perimeter, top_width, k, _, _ = _hydraulics(stations, elevations, mannings_n, sizes, centers, left_bank,
                                                          right_bank, j, wse, k_decay, shallow_factor, deep_factor)
        row = out[i - 1]
        if top_width > 0.0 and area > 0.0 and perimeter > 0.0:
            q = k * discharge_scale
            v = q / area
            if q < prev_q:
                candidate = wse + FIX_UP_STEP
                upper = thalweg + step * (i + 1)
                while candidate < upper:
                    jc = _choose(stations, elevations, sizes, centers, left_reach, right_reach, candidate)
                    area, perimeter, top_width, k, _, _ = _hydraulics(stations, elevations, mannings_n, sizes, centers,
                                                                      left_bank, right_bank, jc, candidate, k_decay,
                                                                      shallow_factor, deep_factor)
                    q_candidate = k * discharge_scale
                    if area > prev_a and perimeter > prev_p and q_candidate > prev_q and q_candidate <= q_top:
                        wse = candidate
                        q = q_candidate
                        v = q_candidate / area
                        j = jc
                        break
                    candidate += FIX_UP_STEP
            if q <= prev_q or q > q_top * DISCHARGE_CAP:
                row[Q], row[V], row[T], row[WSE], row[P] = prev_q, prev_v, prev_t, prev_wse, prev_p
                chosen[i - 1] = prev_j
                continue
            row[Q], row[V], row[T], row[WSE], row[P] = q, v, top_width, wse, perimeter
            prev_t, prev_a, prev_p, prev_q, prev_v, prev_wse = top_width, area, perimeter, q, v, wse
            prev_j = j
            chosen[i - 1] = j
            last = i
        else:
            start = i
            row[Q] = row[V] = row[T] = row[WSE] = row[P] = 0.0
            chosen[i - 1] = j
    return start, last


# --- Candidates and rating curves ----------------------------------------------------------------------------------


def candidate_offsets(offsets: np.ndarray) -> np.ndarray:
    """The distinct offsets the angle search tries, in its order (as _narrowest_direction dedupes them)."""
    tried: list[float] = []
    for offset in np.asarray(offsets, dtype=np.float64).ravel():
        offset = offset % np.pi
        if offset > np.pi / 2:
            offset -= np.pi
        if not any(abs(t - offset) < _SAME_DIRECTION for t in tried):
            tried.append(float(offset))
    return np.array(tried if tried else [0.0])


def section_candidates(grid, section, stream: float, offsets: np.ndarray, length: float, mannings_n: np.ndarray,
                       water_n: float, carve: tuple | None, only_fixed: bool = False) -> Candidates:
    """A section's candidate cross sections (see the notes above), the section itself first. stream is the stream's
    direction at its cell, offsets the angle search's distinct offsets from it, carve the fixed section's (depth,
    trapezoid height, bank elevation), or None if it wasn't carved."""
    from arc.bathymetry import carve_channel, set_bank_distances, set_in_bank_roughness
    fixed = float(section.direction)
    own = int(np.argmin(np.abs(((stream + offsets) - fixed + np.pi / 2) % np.pi - np.pi / 2)))
    order = [own] + [j for j in range(offsets.size) if j != own]
    if only_fixed:
        order = order[:1]
    profiles, reaches = [], []
    for j in order:
        if j == own:
            xs = section.xs
        else:
            direction = stream + offsets[j]
            elevations, spacing = sample_elevations(grid.dem, section.row, section.col, direction, length, grid.dx,
                                                    grid.dy)
            n, _ = sample_elevations(mannings_n, section.row, section.col, direction, length, grid.dx, grid.dy)
            xs = XSection(elevations, n, float(spacing))
            set_in_bank_roughness(xs, section.banks, water_n)
            if carve is not None:
                depth, trapezoid_height, bank_elevation = carve
                carve_channel(xs, section.hydraulic_banks, depth, trapezoid_height=trapezoid_height,
                              bank_elevation=bank_elevation)
            set_bank_distances(xs, section.hydraulic_banks)
        profile = hydraulic_profile(xs)
        profiles.append(profile)
        center = xs.elevations.size // 2
        reaches.append((_reach(xs.elevations, center, -1, float(xs.ordinate_distance)),
                        _reach(xs.elevations, center, 1, float(xs.ordinate_distance))))
    size = max(p.elevations.size for p in profiles)
    m = len(profiles)
    stations, elevations, n = (np.full((m, size), np.nan) for _ in range(3))
    sizes, centers = np.empty(m, dtype=np.int64), np.empty(m, dtype=np.int64)
    for j, p in enumerate(profiles):
        k = p.elevations.size
        stations[j, :k], elevations[j, :k], n[j, :k] = p.stations, p.elevations, p.mannings_n
        sizes[j], centers[j] = k, p.center
    return Candidates(offsets[order] - offsets[own], stations, elevations, n, sizes, centers,
                      float(section.xs.left_bank_distance), float(section.xs.right_bank_distance),
                      np.array([r[0] for r in reaches]), np.array([r[1] for r in reaches]))


def pivot_rating_curve(c: Candidates, q_max: float, slope: float, increments: int, *, baseflow: float = 0.0,
                       roughness=None, slope_factor: float = 1.0, mode: str = "each"):
    """arc.rating_curve.rating_curve on the candidates, pivoting (mode "each") or chosen once at the top ("top"):
    (the RatingCurve or None, each increment's candidate, the top's candidate)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    count = max(int(increments), 0)
    out = np.full((count, 5), np.nan)
    chosen = np.full(count, -1, dtype=np.int64)
    if not q_max > 0.0:
        return RatingCurve(out, math.nan, -1, 0), chosen, -1
    if not slope > 0.0:
        return None, chosen, -1
    parameters = _parameters(roughness)
    discharge_scale = math.sqrt(slope) * _check_slope_factor(slope_factor)
    arrays = (c.stations, c.elevations, c.mannings_n, c.sizes, c.centers, c.left_bank, c.right_bank)
    wse, top = _pivot_top(*arrays, c.left_reach, c.right_reach, q_max / discharge_scale, *parameters)
    if not math.isfinite(wse):
        return None, chosen, -1
    q_top = _hydraulics(*arrays, top, wse, *parameters)[3] * discharge_scale
    if not ACCEPTABLE[0] * q_max <= q_top <= ACCEPTABLE[1] * q_max:
        return None, chosen, int(top)
    if count == 0:
        return RatingCurve(out, wse, -1, 0), chosen, int(top)
    thalweg = float(c.elevations[0, c.centers[0]])
    if mode == "each":
        start, last = _pivot_increments(*arrays, c.left_reach, c.right_reach, *parameters, thalweg, wse, count,
                                        discharge_scale, float(q_top), out, chosen)
    else:
        size = int(c.sizes[top])
        start, last = _increments(c.stations[top, :size], c.elevations[top, :size], c.mannings_n[top, :size],
                                  int(c.centers[top]), c.left_bank, c.right_bank, *parameters, thalweg, wse, count,
                                  discharge_scale, float(q_top), out)
        chosen[:] = top
    if last > start and baseflow > BASEFLOW_MARGIN and out[start, Q] >= baseflow:
        out[start, Q] = baseflow - BASEFLOW_MARGIN
    return RatingCurve(out, wse, int(start), int(last)), chosen, int(top)


# --- In the pipeline -------------------------------------------------------------------------------------------------


def patches(pipeline, mode: str, store: dict | None = None):
    """(owner, name, replacement) for arc.pipeline to pivot its rating curves (see the notes above); its
    rating_curves, but for the curve itself, as the pipeline's. With a store, each cell's fixed and pivoted curves and
    choices go into store["cells"], and both ways' times into store["seconds"]."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    state: dict = {"carve": {}}
    originals = {name: getattr(pipeline, name) for name in ("mannings_n_raster", "carve_channel", "rating_curves")}
    but_baseflow = pipeline._METADATA_BUT_BASEFLOW

    def mannings_n_raster(configs, land_cover):
        state["mannings_n"], state["water_n"] = originals["mannings_n_raster"](configs, land_cover)
        return state["mannings_n"], state["water_n"]

    def carve_channel(xs, banks, depth, **kwargs):
        state["carve"][id(xs)] = (float(depth), float(kwargs["trapezoid_height"]), kwargs.get("bank_elevation"))
        return originals["carve_channel"](xs, banks, depth, **kwargs)

    def rating_curves(configs, grid, cells, sections, slopes, baseflow, max_flow, roughness, quiet=True):
        from arc.outputs import RatingCurves
        if store is not None:
            started = time.perf_counter()
            originals["rating_curves"](configs, grid, cells, sections, slopes, baseflow, max_flow, roughness, quiet)
            store["seconds"] = dict(fixed=time.perf_counter() - started)
        pivot_seconds = 0.0
        increments = int(configs.vdt_database_numiterations)
        curves = RatingCurves.empty(cells.count, max(increments, 0))
        reach_average = configs.reach_average_curve_file
        offsets = candidate_offsets(angle_offsets(configs.degree_manip, configs.degree_interval))
        length = float(configs.x_section_dist)
        kept = []
        for k in range(cells.count):
            section = sections[k]
            comid = float(cells.comids[k])
            if section is None or not section.usable:
                if reach_average:
                    row, col = (int(cells.rows[k]), int(cells.cols[k])) if section is None \
                        else (section.row, section.col)
                    elevation = float(grid.dem[row, col])
                    curves.metadata[k, but_baseflow] = \
                        [comid, row, col, elevation, 0.0, np.nan if section is None else section.xs_angle, elevation]
                continue
            elevation = float(grid.dem[section.row, section.col])
            thalweg = float(section.xs.elevations[section.xs.elevations.size // 2])
            arguments = (float(max_flow[k]), float(slopes[k]), increments)
            options = dict(baseflow=float(baseflow[k]), roughness=roughness,
                           slope_factor=configs.slope_adjustment_factor)
            if section.manual:
                curve, chosen, top, c = rating_curve(section.xs, *arguments, **options), None, 0, None
            else:
                stream = float(stream_direction(grid.streams, int(cells.rows[k]), int(cells.cols[k]),
                                                configs.gen_dir_dist, grid.dx, grid.dy))
                carve = state["carve"].get(id(section.xs))
                started = time.perf_counter()
                c = section_candidates(grid, section, stream, offsets, length, state["mannings_n"], state["water_n"],
                                       carve)
                curve, chosen, top = pivot_rating_curve(c, *arguments, **options, mode=mode)
                pivot_seconds += time.perf_counter() - started
            if store is not None and c is not None:
                fixed = rating_curve(section.xs, *arguments, **options)
                alone = pivot_rating_curve(section_candidates(grid, section, stream, offsets, length,
                                                              state["mannings_n"], state["water_n"], carve,
                                                              only_fixed=True), *arguments, **options)[0]
                same = (fixed is None) == (alone is None) and (fixed is None or (
                    np.array_equal(fixed.increments, alone.increments, equal_nan=True)
                    and fixed.max_wse == alone.max_wse and (fixed.start, fixed.last) == (alone.start, alone.last)))
                kept.append(dict(
                    k=k, row=section.row, col=section.col, reach=int(cells.reaches[k]), stream=stream,
                    direction=float(section.direction), offsets=c.offsets, thalweg=thalweg,
                    bank_elevation=float(section.bank_elevation), q_max=float(max_flow[k]),
                    baseflow=float(baseflow[k]), slope=float(slopes[k]),
                    fixed=None if fixed is None else fixed._asdict(),
                    pivot=None if curve is None else curve._asdict(), chosen=chosen, top=top,
                    top_curve=None if curve is None else pivot_rating_curve(c, *arguments, **options,
                                                                            mode="top")[0]._asdict(),
                    reproduces_fixed=bool(same)))
            if curve is None:
                if reach_average:
                    curves.metadata[k, but_baseflow] = \
                        [comid, section.row, section.col, elevation, slopes[k], section.xs_angle, elevation]
                continue
            if curve.valid:
                curves.increments[k] = curve.increments
                curves.metadata[k] = [comid, section.row, section.col, elevation, baseflow[k], slopes[k],
                                      section.xs_angle, thalweg]
            if reach_average or (configs.print_curve_file and curve.start >= 0 and curve.last > curve.start + 1):
                curves.metadata[k, but_baseflow] = \
                    [comid, section.row, section.col, section.dem_low_point, slopes[k], section.xs_angle, thalweg]
        if store is not None:
            store["cells"] = kept
            store["seconds"]["pivot"] = pivot_seconds
        return curves

    return [(pipeline, "mannings_n_raster", mannings_n_raster), (pipeline, "carve_channel", carve_channel),
            (pipeline, "rating_curves", rating_curves)]


# --- The figures' runs ---------------------------------------------------------------------------------------------


def run_pivot(inputs: dict) -> dict:
    """Run the new pipeline on a site with the cross sections pivoting, and return each cell's fixed and pivoted
    rating curves and choices, and the rating curves' times."""
    from arc import pipeline
    from arc.config import Configs

    import vc_runs
    vc_runs.quiet_logs()
    configs = Configs.from_mapping(inputs)
    store: dict = {}
    replaced = patches(pipeline, "each", store)
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in replaced]
    for owner, name, value in replaced:
        setattr(owner, name, value)
    try:
        started = time.perf_counter()
        pipeline.run(configs, quiet=True, write=False)
        store["run_seconds"] = time.perf_counter() - started
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)
    store["increments"] = int(configs.vdt_database_numiterations)
    return store


def ensure_pivot_runs(runs: Path, sites_root: Path, sites: list[str], log=print) -> None:
    """run_pivot on every site not yet run, into runs/pivot/new/<site>/capture.pkl."""
    import vc_runs
    for site in sites:
        out = vc_runs.run_directory(runs, "pivot", "new", site)
        if (out / "done.json").exists():
            continue
        inputs = vc_runs.site_inputs(sites_root, site, out, {}, ("bathy",))
        try:
            store = run_pivot(inputs)
        except Exception as error:  # a failed site is left out of the figures, and said so
            log(f"  pivot new {site}: FAILED {type(error).__name__}: {error}")
            (out / "failed.txt").write_text(f"{type(error).__name__}: {error}\n")
            continue
        with open(out / "capture.pkl", "wb") as f:
            pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
        (out / "done.json").write_text(json.dumps({"seconds": store["run_seconds"], **store["seconds"]}))
        log(f"  pivot new {site}: rating curves {store['seconds']['fixed']:.1f} s fixed, "
            f"{store['seconds']['pivot']:.1f} s pivoting")
