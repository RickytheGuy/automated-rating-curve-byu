"""Smoothing the bank elevations along the reaches of a stream network.

This is legacy ARC's reach and network pass between finding each cross section's banks and working out its
bathymetry (_smooth_reach_bank_elevations and the functions it calls). The smoothed bank elevation is the level the
channel is carved below with Bathy_Use_Banks, the bank_elevation that bathymetry_depth and carve_channel take.
smooth_bank_elevations does it all:

1. Each reach's widths are filtered (filter_reach_widths). A valid channel narrower than the reach's 25th percentile
   width or wider than its 75th gets banks for the reach's median width instead, and so does a cross section
   without valid banks.
2. Each cross section's lower bank above its stream cell is an observation of the reach's bank elevation, leaving
   out those below the reach's 2nd percentile or above its 97th (reach_bank_observations). With four or more, that
   always leaves out the lowest and the highest, unless another ties with them.
3. The reach's cross sections are put in order from upstream to downstream (order_reach).
4. Each reach's lowest observation is its bank elevation at its outlet, and the network fills in reaches without one
   and makes every outlet lower than the ones upstream of it (network_outlet_elevations).
5. Along each reach the bank elevation falls in a straight line to its outlet, and wherever an observation is lower
   than the line it is refitted through it (reach_bank_surface).

Where legacy ARC used bank indices
----------------------------------
- A channel's width is the distance between its banks, so a single-cell channel is two spacings wide (its banks are
  the ordinates either side), where legacy counted one. Banks for a width are exactly that width, where legacy
  rounded to whole spacings, so a rebuilt width is the reach's median. It only needs widening where a cross section
  can't hold it: a median between one and two of its spacings, or one wider than it reaches.
- An observation is the ground's elevation at the lower bank, not its bank ordinate's.

Errors in the legacy code, not repeated here
--------------------------------------------
- A bank level with the stream cell was ignored and the other one used, but a bank below the stream cell took the
  cross section's observation away, even when the other bank was above it. Here a bank at or below the stream cell
  is ignored either way. Legacy also ignored banks within np.isclose's tolerance above the stream cell (about 1 mm
  at 100 m, growing with the elevation), and banks at exactly 0 m, its placeholder for none.
- Cross sections whose banks weren't valid still gave an observation, from the ordinates their failed search
  stopped at. Here they give none.
- A reach with no reach next to it that had cross sections was put in order along a straight line through its
  cells, and got stations 0, 1, 2 and so on, not metres. Its own comments say a straight line can reverse or scramble
  a curved reach. Here it's in order along its cells from one end, with stations in metres. Which end is upstream is
  decided as legacy did.
- Cells a reach's path couldn't reach, across a gap in its cells, got their straight-line distance to its downstream
  end, which can put them out of order. Here the path jumps the gap (see arc.xsection.stream_path).
- A reach missing from the network made the ordering fail with networkx's NetworkXError. Here it's a ValueError
  saying which reaches.

Numerical differences
---------------------
- The nearest cells between reaches are found in metres, where legacy used cells.
- Of cells equally far from a reach upstream, the first given becomes the downstream end, where legacy took the
  first reached.

What legacy did with the result, which isn't here
-------------------------------------------------
Legacy dropped a reach that got no bank elevation from the network (no observations, and none filled in) from the
bathymetry and the rating curves. Here its bank elevations are NaN, and what to do with them is up to the caller.
Legacy ran this pass whenever it wrote bathymetry, with or without Bathy_Use_Banks. Without it, the width filter
still changed the banks the channel was carved between, and the smoothed elevation went into legacy's bed smoothing,
which arc.bathymetry.bed_smoothing does from the stream cell's elevation instead.
"""
from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Sequence

import networkx as nx
import numpy as np
from numba import njit

from arc.bathymetry.banks import Banks, bank_control_elevation, banks_for_width, single_cell_banks
from arc.xsection.stream_path import along_stream_stations, downstream_order
from arc.xsection.xsection import XSection

MIN_GRADE = 1e-4  # every reach and every cross section falls at least this much per metre downstream
MAXIMUM_WIDTH_INCREASE = 10  # how many spacings wider than the median width a rebuilt channel may be
OUTLIER_PERCENTILES = (2, 97)  # observations outside these percentiles of their reach's are left out
ORIENTING_CELLS = 10  # how many cross sections at each end decide which end of an unconnected reach is upstream


class ReachWidths(NamedTuple):
    """The percentiles of a reach's valid channel widths, which filter_reach_widths works from."""
    q25: float
    median: float
    q75: float


class ReachSections(NamedTuple):
    """A reach's cross sections: each one's stream cell, the section, and its banks from find_banks."""
    rows: np.ndarray
    cols: np.ndarray
    sections: Sequence[XSection]
    banks: Sequence[Banks]


class SmoothedReach(NamedTuple):
    """A reach after smoothing. The arrays are in the order the cross sections were given, except order and
    stations, which are in order from upstream to downstream."""
    banks: list[Banks]  # after the width filter
    bank_elevations: np.ndarray  # the smoothed bank elevations, NaN if the network gave the reach none
    observations: np.ndarray  # each cross section's observation, NaN for none or an outlier
    anchors: np.ndarray  # whether each one's observation became an anchor of the surface
    order: np.ndarray  # the cross sections from upstream to downstream
    stations: np.ndarray  # how far each is along the stream from the upstream one, in metres
    widths: ReachWidths | None  # None if no cross section had valid banks


# --- One reach ----------------------------------------------------------------------------------------------------


def filter_reach_widths(sections: Sequence[XSection], banks: Sequence[Banks]) -> tuple[list[Banks], ReachWidths | None]:
    """A reach's banks with its outlying widths replaced, and the percentiles of its valid widths (legacy
    _apply_reach_top_width_filter and _apply_reach_median_top_width_to_missing_bank).

    A valid channel narrower than the 25th percentile or wider than the 75th is rebuilt at the median width
    (banks_for_width), widened a spacing at a time up to ten spacings until the banks are valid and no wider than the
    75th percentile. If none are, it becomes a single-cell channel, or stays as it was where it can't be one. A cross
    section without valid banks gets banks for the median width, if they're valid. None, and the banks unchanged, if
    no cross section has valid banks.
    """
    if len(sections) != len(banks):
        raise ValueError(f"There are {len(sections)} cross sections but {len(banks)} banks.")
    banks = list(banks)
    widths = np.array([b.top_width for b in banks if b.valid and math.isfinite(b.top_width) and b.top_width > 0.0])
    if widths.size == 0:
        return banks, None
    q25, median, q75 = (float(p) for p in np.percentile(widths, [25, 50, 75]))
    for k, (xs, b) in enumerate(zip(sections, banks)):
        if b.valid and (b.top_width < q25 or b.top_width > q75):
            banks[k] = _rebuilt_at_median(xs, b, median, q75)
        elif not b.valid:
            rebuilt = banks_for_width(xs, median)
            if rebuilt.valid:
                banks[k] = rebuilt
    return banks, ReachWidths(q25, median, q75)


def _rebuilt_at_median(xs: XSection, banks: Banks, median: float, q75: float) -> Banks:
    widest = q75 + np.finfo(np.float64).eps * max(abs(q75), 1.0)
    for increase in range(MAXIMUM_WIDTH_INCREASE + 1):
        rebuilt = banks_for_width(xs, median + increase * float(xs.ordinate_distance))
        if rebuilt.valid and rebuilt.top_width <= widest:
            return rebuilt
    single_cell = single_cell_banks(xs)
    return single_cell if single_cell.valid else banks


def reach_bank_observations(sections: Sequence[XSection], banks: Sequence[Banks]) -> tuple[np.ndarray, float, float]:
    """Each of a reach's cross sections' observation of its bank elevation, and the bounds outside which they were
    left out (legacy _compute_raw_bank_elevation_from_result, _exclude_thalweg_equal_bank_elevations and the
    percentile filter).

    An observation is the lower of a cross section's valid banks above its stream cell (bank_control_elevation), or
    NaN. With four or more, those below the 2nd percentile or above the 97th become NaN too, and those are the
    bounds; otherwise the bounds are infinite.
    """
    observations = np.array([bank_control_elevation(b, float(xs.elevations[xs.elevations.size // 2])) if b.valid
                             else math.nan for xs, b in zip(sections, banks)], dtype=np.float64)
    finite = np.isfinite(observations)
    if np.count_nonzero(finite) < 4:
        return observations, -math.inf, math.inf
    lower, upper = (float(p) for p in np.percentile(observations[finite], OUTLIER_PERCENTILES))
    observations[finite & ((observations < lower) | (observations > upper))] = math.nan
    return observations, lower, upper


def order_reach(network: nx.DiGraph, reach: int, rows, cols, dx: float, dy: float,
                cells: Mapping[int, tuple[np.ndarray, np.ndarray]], observations) -> tuple[np.ndarray, np.ndarray]:
    """A reach's cross sections in order from upstream to downstream, and their stations in metres from the upstream
    one (legacy _order_reach_stream_cells_from_network).

    The downstream end is nearest the first reach downstream that has cross sections, or, failing that, farthest
    along the stream from the one nearest any reach upstream (see downstream_order). cells holds each reach's cross
    sections' stream cells. A reach with neither is ordered from one end of its cells to the other, then turned round
    to put the higher end upstream: an unconnected reach's highest observation before its lowest, or else whichever
    end's first ten cross sections have the higher mean observation.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    downstream = next((cells[s] for s in network.successors(reach) if s in cells and len(cells[s][0]) > 0), None)
    upstream = [cells[p] for p in network.predecessors(reach) if p in cells]
    upstream_cells = (np.concatenate([c[0] for c in upstream]), np.concatenate([c[1] for c in upstream])) \
        if upstream else None
    ordered = downstream_order(rows, cols, dx, dy, downstream_cells=downstream, upstream_cells=upstream_cells)
    if ordered is not None:
        return ordered

    stations = along_stream_stations(rows, cols, dx, dy)
    order = np.argsort(stations, kind="stable")
    stations = stations[order]
    if _higher_end_is_downstream(np.asarray(observations, dtype=np.float64)[order],
                                 network.in_degree(reach) == 0 and network.out_degree(reach) == 0):
        order, stations = order[::-1].copy(), stations[-1] - stations[::-1]
    return order, stations


def _higher_end_is_downstream(ordered_observations: np.ndarray, unconnected: bool) -> bool:
    finite = np.flatnonzero(np.isfinite(ordered_observations))
    if unconnected and finite.size >= 2:
        highest = finite[np.argmax(ordered_observations[finite])]
        lowest = finite[np.argmin(ordered_observations[finite])]
        if highest != lowest:
            return bool(highest > lowest)
    count = min(ORIENTING_CELLS, ordered_observations.size)
    first, last = ordered_observations[:count], ordered_observations[-count:]
    first, last = first[np.isfinite(first)], last[np.isfinite(last)]
    return bool(first.size > 0 and last.size > 0 and first.mean() < last.mean())


# --- The network --------------------------------------------------------------------------------------------------


def reach_network(reach_ids, downstream_ids, lengths) -> nx.DiGraph:
    """The stream network (legacy _build_reach_network_graph, from its table's columns rather than the file): an
    edge from each reach to the one downstream of it, and each reach's length in metres as its "length".

    A reach's first row counts. A downstream ID that is missing (None, NaN or ''), the reach itself, or not a reach in
    the table makes it an outlet. A length that isn't positive and finite counts as 1 m.
    """
    records: dict[int, tuple[int | None, float]] = {}
    for reach, downstream, length in zip(reach_ids, downstream_ids, lengths):
        reach = _reach_id(reach)
        if reach is None or reach in records:
            continue
        downstream = _reach_id(downstream)
        length = float(length) if length is not None else math.nan
        records[reach] = (None if downstream == reach else downstream,
                          length if math.isfinite(length) and length > 0.0 else 1.0)
    if not records:
        raise ValueError("The stream network has no reaches.")
    network = nx.DiGraph()
    for reach, (_, length) in records.items():
        network.add_node(reach, length=length)
    for reach, (downstream, _) in records.items():
        if downstream is not None and downstream in records:
            network.add_edge(reach, downstream)
    return network


def _reach_id(value) -> int | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) else None


def _length(network: nx.DiGraph, reach: int) -> float:
    length = float(network.nodes[reach].get("length", 1.0))
    return length if math.isfinite(length) and length > 0.0 else 1.0


def network_outlet_elevations(network: nx.DiGraph, minima: Mapping[int, float], maxima: Mapping[int, float]
                              ) -> tuple[dict[int, float], dict[int, float]]:
    """Each reach's bank elevation at its outlet, and the grade its bank elevation falls at along it (legacy
    _estimate_network_smoothed_reach_min_bank_elevations, up to its cross-section pass).

    minima and maxima are each reach's lowest and highest observations (reaches without any left out). Starting from
    its lowest observation, each reach's outlet elevation is:
    - Where it has none, filled in along every path from a headwater down: interpolated between the reaches with one,
      and extended beyond them at the grade between the nearest two (or at the minimum grade, 0.0001, with just
      one). A reach on several paths takes the lowest.
    - Lowered where needed so that each reach falls at least the minimum grade along its length from the lowest of
      the reaches flowing into it.
    Each reach's grade is then:
    - A reach with reaches flowing into it: its fall from the lowest of those over its length. An outlet uses its own
      lowest observation, lowered if needed as above, or without one the median of the grades flowing into it.
    - A headwater: from its highest observation to its lowest over its length, or without them the median grade of
      the reaches downstream of it.
    - A reach with none flowing in or out: from its highest observation to its lowest.
    Every grade is at least the minimum grade.
    """
    if len(minima) == 0:
        return {}, {}
    if network.number_of_nodes() == 0:
        raise ValueError("The stream network is empty, so its bank elevations can't be smoothed.")

    candidates: dict[int, list[float]] = {}
    for reach, elevation in minima.items():
        if reach in network and math.isfinite(elevation):
            candidates.setdefault(reach, []).append(float(elevation))
    headwaters = [reach for reach in network if network.in_degree(reach) == 0] or list(network)
    unconnected = {reach for reach in network if network.in_degree(reach) == 0 and network.out_degree(reach) == 0}

    for headwater in headwaters:
        path, visited, reach = [headwater], set(), headwater
        while network.out_degree(reach) > 0 and reach not in visited:
            visited.add(reach)
            reach = next(iter(network.successors(reach)))
            path.append(reach)
        for reach, elevation in zip(path, _fill_path(network, path, minima)):
            if math.isfinite(elevation):
                candidates.setdefault(reach, []).append(float(elevation))
    outlets = {reach: min(values) for reach, values in candidates.items()}

    def lowest_inflow(reach):
        inflows = [outlets[p] for p in network.predecessors(reach) if p in outlets]
        return min(inflows) if inflows else None

    for _ in range(max(network.number_of_nodes(), 1)):
        changed = False
        for reach in network:
            inflow = lowest_inflow(reach)
            if reach in outlets and inflow is not None:
                highest = inflow - MIN_GRADE * _length(network, reach)
                if outlets[reach] > highest:
                    outlets[reach] = highest
                    changed = True
        if not changed:
            break

    grades: dict[int, float] = {}
    for reach, outlet in outlets.items():
        inflow = lowest_inflow(reach)
        if inflow is not None:
            grades[reach] = max((inflow - outlet) / _length(network, reach), MIN_GRADE)

    for reach in network:
        if network.out_degree(reach) != 0 or reach in unconnected:
            continue
        inflow = lowest_inflow(reach)
        length = _length(network, reach)
        if inflow is None:
            grades[reach] = MIN_GRADE
        elif math.isfinite(minima.get(reach, math.nan)):
            outlets[reach] = min(float(minima[reach]), inflow - MIN_GRADE * length)
            grades[reach] = max((inflow - outlets[reach]) / length, MIN_GRADE)
        else:
            inflow_grades = [grades[p] for p in network.predecessors(reach) if p in grades]
            grades[reach] = max(float(np.nanmedian(inflow_grades)), MIN_GRADE) if inflow_grades else MIN_GRADE
            outlets[reach] = inflow - grades[reach] * length

    for reach in unconnected:
        lowest, highest = minima.get(reach, math.nan), maxima.get(reach, math.nan)
        if math.isfinite(lowest):
            fall = highest - lowest if math.isfinite(highest) else 0.0
            grades[reach] = max(fall / _length(network, reach), MIN_GRADE)
            outlets[reach] = float(lowest)

    for headwater in headwaters:
        if headwater in unconnected:
            continue
        lowest, highest = minima.get(headwater, math.nan), maxima.get(headwater, math.nan)
        if math.isfinite(lowest) and math.isfinite(highest):
            grades[headwater] = max((highest - lowest) / _length(network, headwater), MIN_GRADE)
        else:
            downstream_grades = [grades[s] for s in network.successors(headwater) if s in grades]
            grades[headwater] = max(float(np.nanmedian(downstream_grades)), MIN_GRADE) if downstream_grades \
                else MIN_GRADE

    missing = sorted(reach for reach in minima if reach not in outlets)
    if missing:
        raise ValueError("Network smoothing gave no bank elevation for reach_id "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
    return outlets, grades


def _fill_path(network: nx.DiGraph, path: list[int], minima: Mapping[int, float]) -> np.ndarray:
    """Outlet elevations along a path from a headwater down, from the reaches on it with an observation. NaN for all
    of them if none has one."""
    stations = np.cumsum([_length(network, reach) for reach in path])
    observed = np.array([float(minima.get(reach, math.nan)) for reach in path])
    known = np.flatnonzero(np.isfinite(observed))
    if known.size == 0:
        return np.full(len(path), math.nan)
    if known.size == 1:
        return observed[known[0]] - MIN_GRADE * (stations - stations[known[0]])
    at, elevations = stations[known], observed[known]
    filled = np.interp(stations, at, elevations)
    first = max((elevations[0] - elevations[1]) / (at[1] - at[0]), MIN_GRADE) if at[1] > at[0] else MIN_GRADE
    last = max((elevations[-2] - elevations[-1]) / (at[-1] - at[-2]), MIN_GRADE) if at[-1] > at[-2] else MIN_GRADE
    before, after = stations < at[0], stations > at[-1]
    filled[before] = elevations[0] + first * (at[0] - stations[before])
    filled[after] = elevations[-1] - last * (stations[after] - at[-1])
    return filled


# --- Along a reach ------------------------------------------------------------------------------------------------


def reach_baseline(network: nx.DiGraph, reach: int, stations, outlets: Mapping[int, float],
                   grades: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """The bank elevation at each of a reach's ordered cross sections before observations refit it, and how far along
    the reach each is as a share of it (legacy _interpolate_reach_bank_elevation_surface).

    It falls in a straight line from its outlet elevation plus its grade times its length, at the first cross
    section, to its outlet elevation at the last. A reach with one cross section is all outlet. Cross sections at the
    same station are spread evenly in order.
    """
    stations = np.asarray(stations, dtype=np.float64)
    if stations.size == 0:
        return np.empty(0), np.empty(0)
    if reach not in outlets:
        raise ValueError(f"Network smoothing gave no bank elevation for reach_id {reach}.")
    outlet = float(outlets[reach])
    grade = float(grades.get(reach, 0.0))
    upstream = outlet + (grade if math.isfinite(grade) and grade >= 0.0 else 0.0) * _length(network, reach)
    if stations.size == 1:
        fractions = np.ones(1)
    else:
        span = float(stations[-1] - stations[0])
        if math.isfinite(span) and not np.isclose(span, 0.0):
            fractions = np.clip((stations - stations[0]) / span, 0.0, 1.0)
        else:
            fractions = np.linspace(0.0, 1.0, stations.size)
    return np.minimum.accumulate(upstream + fractions * (outlet - upstream)), fractions


@njit(cache=True, error_model="numpy")
def _anchor(observations, baseline, fractions, length, outlet, min_grade, ceiling, thalwegs, lower, upper):
    n = observations.size
    surface = np.empty(n)
    grades = np.full(n, min_grade)
    anchors = np.zeros(n, np.bool_)
    if n == 0:
        return surface, grades, anchors
    stations = np.empty(n)
    furthest = -np.inf
    for i in range(n):
        furthest = max(furthest, min(max(fractions[i], 0.0), 1.0) * length)
        stations[i] = furthest
    usable = np.empty(n, np.bool_)
    for i in range(n):
        usable[i] = (math.isfinite(observations[i]) and lower <= observations[i] <= upper
                     and (not math.isfinite(thalwegs[i]) or observations[i] > thalwegs[i]))

    anchor_station = stations[0]
    anchor = baseline[0]
    if usable[0] and observations[0] < anchor:
        anchor = observations[0]
        anchors[0] = True
    if math.isfinite(ceiling) and anchor > ceiling:
        anchor = ceiling
        anchors[0] = not abs(anchor - baseline[0]) <= 1e-8 + 1e-5 * abs(baseline[0])  # legacy's np.isclose
    anchor = max(anchor, outlet + min_grade * max(length - anchor_station, 0.0))
    surface[0] = anchor
    first = 0
    grade = _grade_to_outlet(anchor_station, anchor, length, outlet, min_grade)
    grades[0] = grade

    for i in range(1, n):
        station = stations[i]
        predicted = anchor - grade * max(station - anchor_station, 0.0)
        highest = surface[i - 1] - min_grade * max(station - stations[i - 1], 0.0)
        # An anchor can't be so low that the outlet can't be reached at the minimum grade
        z = max(observations[i], outlet + min_grade * max(length - station, 0.0)) if usable[i] else np.nan
        if usable[i] and z < predicted:
            z = min(z, highest)
            if station - anchor_station > 0.0:
                # Refit the stretch from the last anchor through this one
                grade = max((anchor - z) / (station - anchor_station), min_grade)
                for k in range(first, i + 1):
                    surface[k] = anchor - grade * max(stations[k] - anchor_station, 0.0)
                for k in range(first, i):
                    grades[k] = grade
            else:
                surface[i] = z
            first = i
            anchor_station = station
            anchor = surface[i]
            anchors[i] = True
            grade = _grade_to_outlet(anchor_station, anchor, length, outlet, min_grade)
        else:
            surface[i] = min(predicted, highest)
        grades[i] = grade
    return surface, grades, anchors


@njit(cache=True, error_model="numpy")
def _grade_to_outlet(station, elevation, length, outlet, min_grade):
    remaining = length - station
    if remaining <= 0.0:
        return min_grade
    return max((elevation - outlet) / remaining, min_grade)


def reach_bank_surface(observations, baseline, fractions, length: float, outlet: float, *,
                       min_grade: float = MIN_GRADE, ceiling: float | None = None, thalwegs=None,
                       lower: float = -math.inf, upper: float = math.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bank elevation at each of a reach's cross sections, in order from upstream to downstream, refitted
    through its low observations, with each one's grade downstream and whether it became an anchor (legacy
    _anchor_interpolated_bank_surface_to_cell_observations).

    It starts at the baseline's first elevation, or the first observation if that's lower, but no higher than the
    ceiling (the lowest outlet flowing into the reach), and falls in a straight line to the outlet. Walking
    downstream, an observation below that line becomes an anchor: the stretch from the last anchor is refitted
    through it, and from it the line starts again towards the outlet. An observation counts if it's between lower and
    upper and above its stream cell (thalwegs), and each anchor is kept high enough to still reach the outlet at the
    minimum grade. Every cross section is at least the minimum grade below the one before. Stations are the fractions
    times the reach's length.
    """
    observations = np.asarray(observations, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    thalwegs = np.full(observations.shape, math.nan) if thalwegs is None else np.asarray(thalwegs, dtype=np.float64)
    if not observations.shape == baseline.shape == fractions.shape == thalwegs.shape:
        raise ValueError("The observations, baseline, fractions and thalwegs must be the same shape.")
    if lower > upper:
        raise ValueError(f"lower ({lower}) is above upper ({upper}).")
    length = float(length) if math.isfinite(length) and length > 0.0 else 1.0
    return _anchor(observations, baseline, fractions, length, float(outlet), max(float(min_grade), 0.0),
                   math.nan if ceiling is None else float(ceiling), thalwegs, float(lower), float(upper))


# --- Everything ---------------------------------------------------------------------------------------------------


def smooth_bank_elevations(network: nx.DiGraph, reaches: Mapping[int, ReachSections], dx: float, dy: float
                           ) -> dict[int, SmoothedReach]:
    """Smooth the bank elevations of every reach's cross sections along the network (see the notes above; legacy
    _smooth_reach_bank_elevations).

    network runs from each reach to the one downstream of it, with each reach's length in metres (reach_network).
    Every reach with cross sections must be in it, and reaches without any can be too, as the network's links.
    """
    missing = sorted(reach for reach in reaches if reach not in network)
    if missing:
        raise ValueError("These reaches have cross sections but aren't in the stream network: "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
    cells = {reach: (np.asarray(r.rows, dtype=np.int64), np.asarray(r.cols, dtype=np.int64))
             for reach, r in reaches.items() if len(r.sections) > 0}

    prepared = {}
    for reach, sections in reaches.items():
        if len(sections.sections) == 0:
            continue
        banks, widths = filter_reach_widths(sections.sections, sections.banks)
        observations, lower, upper = reach_bank_observations(sections.sections, banks)
        order, stations = order_reach(network, reach, sections.rows, sections.cols, dx, dy, cells, observations)
        thalwegs = np.array([float(xs.elevations[xs.elevations.size // 2]) for xs in sections.sections])
        prepared[reach] = (banks, widths, observations, lower, upper, order, stations, thalwegs)

    finite = {reach: p[2][np.isfinite(p[2])] for reach, p in prepared.items()}
    minima = {reach: float(values.min()) for reach, values in finite.items() if values.size > 0}
    maxima = {reach: float(values.max()) for reach, values in finite.items() if values.size > 0}
    if prepared and not minima:
        raise ValueError("No reach has a bank elevation above its stream cells, so the network can't be smoothed.")
    outlets, grades = network_outlet_elevations(network, minima, maxima)

    smoothed = {}
    for reach, (banks, widths, observations, lower, upper, order, stations, thalwegs) in prepared.items():
        bank_elevations = np.full(observations.size, math.nan)
        anchors = np.zeros(observations.size, dtype=bool)
        if reach in outlets:
            baseline, fractions = reach_baseline(network, reach, stations, outlets, grades)
            inflows = [outlets[p] for p in network.predecessors(reach) if p in outlets]
            surface, _, ordered_anchors = reach_bank_surface(
                observations[order], baseline, fractions, _length(network, reach), outlets[reach],
                ceiling=min(inflows) if inflows else None, thalwegs=thalwegs[order], lower=lower, upper=upper)
            bank_elevations[order] = surface
            anchors[order] = ordered_anchors
        smoothed[reach] = SmoothedReach(banks, bank_elevations, observations, anchors, order, stations, widths)
    return smoothed
