"""Filling in and smoothing the channel depths along the reaches of a stream network.

These are legacy ARC's reach and network steps between working out each cross section's bathymetry depth
(bathymetry_depth) and carving it (carve_channel): _smooth_reach_bathymetry_depths and
_smooth_reach_excavated_bed_elevations. smooth_channel_depths does both, taking each reach's cross sections from
upstream to downstream in the order the bank smoothing put them in (smooth_bank_elevations):

1. A depth that isn't above 0 and at most 25 m is filled in with the one before it along the stream. The first in a
   reach takes the shallowest of the last depths of the reaches flowing into it, or failing that 0.5 m
   (fill_reach_depths).
2. The bed, each cross section's reference level less its depth, is smoothed with a running median of five cross
   sections, two either side and fewer at a reach's ends. With the bed cap on it then rises or falls no more than
   1 cm per metre along the stream, starting within that of the lowest bed flowing into the reach
   (smooth_reach_bed). The depth to carve is the reference level's height above the smoothed bed, or 0 where the
   bed is above it.

The reference level is the smoothed bank elevation with Bathy_Use_Banks, and otherwise the stream cell's elevation,
as for bathymetry_depth and carve_channel.

The bed cap
-----------
It keeps the bed from following the banks down a reach falling faster than 1%, so the channel fills in: on a reach
of 10 m cells falling 2%, a 1 m channel is 0.2 m deep after 100 m and gone after 120 m. At a confluence it keeps a
reach's first bed near the lowest bed flowing in, so where the banks drop 3 m, the reach below has no channel for
its first 210 m. max_bed_grade=None turns it off, along the reaches and across the confluences, leaving the running
median.

The running median has fewer cross sections to one side at a reach's ends, so on a sloping bed it comes out low at
the upstream end and high at the downstream end: on that 2% reach, the first cross section is 0.2 m deeper and the
last 0.2 m shallower. Legacy's did the same.

Errors in the legacy code, not repeated here
--------------------------------------------
- A NaN depth wasn't filled in, and the next depth to fill in after one got 0.5 m, not the depth before the NaN.
- A reach with no reach next to it that had cross sections was put in the order its cells came in, row by row, with
  stations 0, 1, 2 and so on rather than metres. So its median ran across the raster rather than along the stream,
  and the cap let its bed fall only 1 cm per cross section: on a reach falling 0.5%, a 1 m channel was 0.05 m deep
  250 m along and gone by 490 m. Here the order and the stations are the bank smoothing's, along the stream in
  metres.
- Without Bathy_Use_Banks the bed was the smoothed bank elevation less the depth, but the channel was carved that
  depth below the stream cell, so the bed smoothed wasn't the bed carved. Where the stream cells rise and fall more
  than the smoothed banks, as they usually do, the carved bed kept all of that. Here the bed is the stream cell's
  elevation less the depth.
- A reach's first bed was held to the lowest bed flowing into it by 1 cm per metre of its own station, which is
  always 0, so by legacy's shortest step of 0.1 m: to within 1 mm. Here it's per metre from the cross section that
  bed came from.
- A missing bed (a NaN bank elevation or stream cell) made the running median NaN within two cross sections of it,
  and the cap then made every bed downstream of that in the reach NaN, which carves nothing: one missing bank
  elevation took the channel out of 49 of a reach's 50 cross sections. Here a missing bed is left out, and its cross
  section keeps its depth.
- A loop in the stream network raised networkx's NetworkXUnfeasible. Here it's a ValueError.

What isn't here
---------------
- Legacy then ran the cap back upstream, which after the pass downstream has nothing left to change.
"""
from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Sequence

import networkx as nx
import numpy as np
from numba import njit

from arc.bathymetry.channel import MAX_DEPTH
from arc.bathymetry.smoothing import ReachSections, SmoothedReach

DEFAULT_DEPTH = 0.5  # the depth filled in with none before it to take
MEDIAN_WINDOW = 5  # how many cross sections the running median of the bed takes in
MAX_BED_GRADE = 0.01  # the bed cap: how much the bed may rise or fall per metre along the stream
SHORTEST_STEP = 0.1  # metres: the cap allows at least this much distance between cross sections


class ChannelDepths(NamedTuple):
    """A reach's depths after filling in and smoothing, in the order its cross sections were given."""
    depths: np.ndarray  # the depths to carve
    beds: np.ndarray  # the smoothed bed elevations, NaN where the reference level is missing
    filled: np.ndarray  # the depths after filling in, before smoothing


def _upstream_to_downstream(network: nx.DiGraph) -> list:
    try:
        return list(nx.topological_sort(network))
    except nx.NetworkXUnfeasible:
        raise ValueError("The stream network has a loop, so its reaches can't be put in order downstream.") from None


def fill_reach_depths(network: nx.DiGraph, depths: Mapping[int, Sequence[float]],
                      orders: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Each reach's depths, in the order given, with those that aren't above 0 and at most 25 m (NaN included)
    filled in (legacy _smooth_reach_bathymetry_depths).

    orders holds each reach's cross sections from upstream to downstream, as indices into its depths. Going
    downstream, a depth is filled in with the one before it, the first in a reach with the shallowest last depth of the
    reaches flowing into it (those with cross sections), or with neither, 0.5 m.
    """
    _check_reaches(network, depths)
    filled: dict[int, np.ndarray] = {}
    last_depths: dict[int, float] = {}
    for reach in _upstream_to_downstream(network):
        if reach not in depths:
            continue
        values = np.array(depths[reach], dtype=np.float64)
        order = np.asarray(orders[reach], dtype=np.int64)
        if order.size != values.size:
            raise ValueError(f"Reach {reach} has {values.size} depths but {order.size} cross sections in order.")
        inflows = [last_depths[p] for p in network.predecessors(reach) if p in last_depths]
        previous = min(inflows) if inflows else DEFAULT_DEPTH
        for i in order:
            if not 0.0 < values[i] <= MAX_DEPTH:
                values[i] = previous
            previous = values[i]
        if order.size > 0:
            last_depths[reach] = previous
        filled[reach] = values
    return filled


@njit(cache=True, error_model="numpy")
def _smooth_bed(beds, stations, window, max_grade, inflow_bed, inflow_distance):
    n = beds.size
    pad = window // 2
    smoothed = np.full(n, np.nan)
    values = np.empty(2 * pad + 1)
    for i in range(n):
        if math.isnan(beds[i]):
            continue
        count = 0
        for k in range(max(i - pad, 0), min(i + pad + 1, n)):
            if not math.isnan(beds[k]):
                values[count] = beds[k]
                count += 1
        smoothed[i] = np.median(values[:count])
    if not max_grade >= 0.0:
        return smoothed

    last = -1  # the last cross section with a bed
    for i in range(n):
        if math.isnan(smoothed[i]):
            continue
        if last >= 0:
            limit = max_grade * max(stations[i] - stations[last], SHORTEST_STEP)
            smoothed[i] = min(max(smoothed[i], smoothed[last] - limit), smoothed[last] + limit)
        elif not math.isnan(inflow_bed):
            limit = max_grade * max(inflow_distance + (stations[i] - stations[0]), SHORTEST_STEP)
            smoothed[i] = min(max(smoothed[i], inflow_bed - limit), inflow_bed + limit)
        last = i
    return smoothed


def smooth_reach_bed(beds, stations, *, window: int = MEDIAN_WINDOW, max_bed_grade: float | None = MAX_BED_GRADE,
                     inflow_bed: float | None = None, inflow_distance: float = 0.0) -> np.ndarray:
    """A reach's bed elevations, in order from upstream to downstream, smoothed (legacy
    _smooth_reach_excavated_bed_elevations, for one reach).

    Each bed becomes the median of those within window // 2 cross sections of it. A cross section without a bed
    (NaN) is left out, and stays NaN. With max_bed_grade, the beds then change from each to the next by no more than
    that times the distance between them along the stream (stations, in metres), and at least SHORTEST_STEP apart.
    The first is held that way to inflow_bed, the lowest bed flowing into the reach, inflow_distance upstream of the
    first cross section.
    """
    beds = np.asarray(beds, dtype=np.float64)
    stations = np.asarray(stations, dtype=np.float64)
    if beds.shape != stations.shape or beds.ndim != 1:
        raise ValueError("The beds and stations must be 1-d and the same length.")
    if int(window) < 1:
        raise ValueError(f"The median window must be at least 1 cross section, not {window!r}.")
    grade = math.nan if max_bed_grade is None else float(max_bed_grade)
    if grade < 0.0:
        raise ValueError(f"max_bed_grade can't be negative ({max_bed_grade!r}).")
    return _smooth_bed(beds, stations, int(window), grade, math.nan if inflow_bed is None else float(inflow_bed),
                       float(inflow_distance))


def smooth_channel_depths(network: nx.DiGraph, reaches: Mapping[int, ReachSections],
                          smoothed: Mapping[int, SmoothedReach], depths: Mapping[int, Sequence[float]], dx: float,
                          dy: float, *, use_banks: bool, max_bed_grade: float | None = MAX_BED_GRADE,
                          window: int = MEDIAN_WINDOW) -> dict[int, ChannelDepths]:
    """Fill in and smooth every reach's depths along the network (see the notes above).

    reaches and smoothed are what smooth_bank_elevations took and gave, and depths holds each reach's depths from
    bathymetry_depth, in the same order. use_banks is Bathy_Use_Banks: whether the reference level is the smoothed
    bank elevation or the stream cell's. max_bed_grade=None turns the bed cap off.
    """
    missing = sorted(reach for reach in depths if reach not in reaches or reach not in smoothed)
    if missing:
        raise ValueError("These reaches have depths but no cross sections or bank smoothing: "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
    for reach, values in depths.items():
        if len(values) != len(reaches[reach].sections) or len(values) != smoothed[reach].order.size:
            raise ValueError(f"Reach {reach} has {len(values)} depths but {len(reaches[reach].sections)} cross "
                             f"sections.")
    filled = fill_reach_depths(network, depths, {reach: smoothed[reach].order for reach in depths})

    results: dict[int, ChannelDepths] = {}
    ends: dict[int, tuple[float, int, int]] = {}  # each reach's last bed and its stream cell
    for reach in _upstream_to_downstream(network):
        if reach not in depths:
            continue
        sections, order, stations = reaches[reach], smoothed[reach].order, smoothed[reach].stations
        if use_banks:
            references = np.asarray(smoothed[reach].bank_elevations, dtype=np.float64)
        else:
            references = np.array([float(xs.elevations[xs.elevations.size // 2]) for xs in sections.sections])
        rows = np.asarray(sections.rows, dtype=np.int64)[order]
        cols = np.asarray(sections.cols, dtype=np.int64)[order]

        inflow_bed, inflow_distance = None, 0.0
        inflows = [ends[p] for p in network.predecessors(reach) if p in ends]
        if inflows and order.size > 0:
            bed, row, col = min(inflows, key=lambda end: end[0])
            inflow_bed = bed
            inflow_distance = math.hypot((cols[0] - col) * dx, (rows[0] - row) * dy)
        beds = smooth_reach_bed(references[order] - filled[reach][order], stations, window=window,
                                max_bed_grade=max_bed_grade, inflow_bed=inflow_bed, inflow_distance=inflow_distance)
        with_bed = np.flatnonzero(~np.isnan(beds))
        if with_bed.size > 0:
            last = with_bed[-1]
            ends[reach] = (float(beds[last]), int(rows[last]), int(cols[last]))

        reach_beds = np.full(order.size, math.nan)
        reach_beds[order] = beds
        carved = filled[reach].copy()
        known = ~np.isnan(reach_beds) & ~np.isnan(references)
        carved[known] = np.maximum(references[known] - reach_beds[known], 0.0)
        results[reach] = ChannelDepths(carved, reach_beds, filled[reach])
    return results


def _check_reaches(network: nx.DiGraph, reaches) -> None:
    missing = sorted(reach for reach in reaches if reach not in network)
    if missing:
        raise ValueError("These reaches have cross sections but aren't in the stream network: "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
