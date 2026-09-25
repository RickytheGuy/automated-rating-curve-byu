"""Stream slopes from the DEM at a reach's stream cells.

The slope between two stream cells is the difference in their elevations over the distance between them along the
stream (see arc.xsection.stream_path), and never over less than the straight line between them. Legacy ARC used the
straight line, which is shorter wherever the stream bends, so slopes on winding streams come out lower here. On a
straight reach along a row, a column or a diagonal the two are the same. At other angles a path of whole cells is
longer than the straight line, up to 8.2% at 22.5 degrees to the rows, so those slopes come out up to 7.6% lower.

The slope methods are legacy ARC's Stream_Slope_Method options:
- local_average: each cell's slope is the mean of its slopes to the reach's other cells within Gen_Slope_Dist cells
  of it along each axis, from 0.0001 to 0.5 (local_average_slopes).
- reach_average: every cell takes its reach's median slope, from the slopes between all pairs of the reach's cells
  that are within Gen_Slope_Dist cells of each other along each axis. The median is of the slopes between two
  percentiles of them, 25 and 75 by default (reach_median_slope).
- local_average_corrected: each cell's local average, limited to its reach's percentiles (corrected_local_slopes).
- end_points: the drop between the ends of the reach's stream line over its length (end_point_slope).
A reach whose two percentiles are equal, as one too short or too flat to measure, takes its neighbours' slope
(fill_unresolved_reach_slopes).

Errors in the legacy code, not repeated here
--------------------------------------------
- The local average looked at rows and columns from Gen_Slope_Dist before the cell to one less than Gen_Slope_Dist
  after it, so one edge of its square was missing: going east, the cells 10 columns downstream never counted, but
  the cells 10 columns upstream did. Here the square reaches the same distance on every side.
- A reach with exactly two slopes between its cells has neither between its 25th and 75th percentiles, so legacy
  gave it the default slope, 0.0002, while its percentiles held its real slopes: slopes of 0.05 and 0.1 made a reach
  slope of 0.0002 between percentiles of 0.0625 and 0.0875. Here it's the median of the two.
- Filling an unresolved reach from upstream took the upstream neighbour with the most reaches above it, even when
  that neighbour had no slope (it wasn't on the raster). The reach was then filled from downstream only, or not at
  all. Here the upstream neighbour is the one with the most reaches above it among those with a slope. Ties go to the
  lower reach ID, where legacy took an arbitrary one.
- The end-point slope:
  - An end moved along the line to the raster only when the other end was on it. A line with both ends off the
    raster got no slope, even where it crossed the raster, which failed later with a KeyError. Here each end moves
    along the line to the first point with data, and a line without any has a slope of NaN.
  - With both ends on the raster, an end was used even where the DEM had no data. Here an end without data is
    treated like one off the raster.
  - Of a line in several parts it measured the longest part's ends, but divided by the length of all of them. Here
    it divides by the longest part's share of the length.
  - A point less than a cell beyond the raster's left or top edge was read from the edge cell, since int() rounds
    towards zero. Here it's off the raster.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
from numba import njit

from arc.xsection.stream_path import _columns_between, _row_index

MIN_SLOPE = 1e-4
MAX_SLOPE = 0.5
UNRESOLVED_SLOPE = 0.0002  # legacy's reach slope, and its percentiles, when a reach has no slopes to take them from
NO_DATA = -9999.0  # an elevation at or below this is missing, as in legacy ARC


@njit(cache=True, error_model="numpy")
def _between(rows, cols, stations, dx, dy, i, k):
    """How far apart two cells are along the stream, and never less than the straight line between them."""
    along = abs(stations[k] - stations[i])
    across, down = (cols[k] - cols[i]) * dx, (rows[k] - rows[i]) * dy
    straight_squared = across * across + down * down
    return along if along * along >= straight_squared else math.sqrt(straight_squared)


@njit(cache=True, error_model="numpy")
def _local_average_slopes(z, rows, cols, stations, dx, dy, distance):
    n = rows.size
    order, sorted_cols, row_start, first_row = _row_index(rows, cols)
    last_row = first_row + row_start.size - 2
    slopes = np.empty(n)
    for i in range(n):
        total = 0.0
        count = 0
        for row in range(max(rows[i] - distance, first_row), min(rows[i] + distance, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - distance, cols[i] + distance)
            for p in range(lo, hi):
                k = order[p]
                if k == i:
                    continue
                length = _between(rows, cols, stations, dx, dy, i, k)
                if length > 0.0:
                    total += abs(z[i] - z[k]) / length
                    count += 1
        slope = total / count if count > 0 else 0.0
        if slope <= MIN_SLOPE:
            slope = MIN_SLOPE
        if slope >= MAX_SLOPE:
            slope = MAX_SLOPE
        slopes[i] = slope
    return slopes


@njit(cache=True, error_model="numpy")
def _pair_slopes(z, rows, cols, stations, dx, dy, distance):
    """The slope between each pair of cells within `distance` cells of each other along each axis, rounded to 1e-8,
    for the pairs whose slope rounds to more than 0. Each pair counts once."""
    n = rows.size
    order, sorted_cols, row_start, first_row = _row_index(rows, cols)
    last_row = first_row + row_start.size - 2
    slopes = np.empty(n * (2 * distance + 1) + 16)  # room for every pair along a path of cells
    size = 0
    for p in range(n):
        i = order[p]
        for row in range(rows[i], min(rows[i] + distance, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - distance, cols[i] + distance)
            if row == rows[i]:
                lo = p + 1  # in the cell's own row only the cells after it, so that each pair counts once
            for q in range(lo, hi):
                k = order[q]
                length = _between(rows, cols, stations, dx, dy, i, k)
                if length == 0.0:
                    continue
                slope = np.rint(abs(z[i] - z[k]) / length * 1e8) / 1e8  # np.round(..., 8), without its pow
                if slope > 0.0:
                    if size == slopes.size:
                        slopes = np.concatenate((slopes, np.empty(slopes.size)))
                    slopes[size] = slope
                    size += 1
    return slopes[:size]


@njit(cache=True, error_model="numpy")
def _round_significant(x, digits):
    """Legacy ARC's round_sig."""
    if x == 0.0:
        return 0.0
    if not np.isfinite(x):
        return x
    factor = 10.0 ** (digits - 1 - int(math.floor(math.log10(abs(x)))))
    return math.floor(x * factor + 0.5) / factor


@njit(cache=True, error_model="numpy")
def _reach_median_slope(z, rows, cols, stations, dx, dy, distance, low, high):
    if rows.size < 2:
        return UNRESOLVED_SLOPE, UNRESOLVED_SLOPE, UNRESOLVED_SLOPE
    slopes = _pair_slopes(z, rows, cols, stations, dx, dy, distance)
    if slopes.size == 0:
        return UNRESOLVED_SLOPE, UNRESOLVED_SLOPE, UNRESOLVED_SLOPE
    # Legacy rounded the slopes to 8 significant figures for the percentiles. That changes nothing below 1, where the
    # slopes already have 8 decimals, so only those above need it
    rounded = slopes
    if slopes.max() >= 1.0:
        rounded = slopes.copy()
        for k in range(slopes.size):
            if slopes[k] >= 1.0:
                rounded[k] = _round_significant(slopes[k], 8)
    percentiles = np.percentile(rounded, np.array([low, high]))
    lower, upper = np.round(percentiles[0], 8), np.round(percentiles[1], 8)
    kept = np.empty(slopes.size)
    count = 0
    for slope in slopes:
        if lower <= slope <= upper:
            kept[count] = slope
            count += 1
    if count == 0:  # two slopes, neither between their percentiles
        return np.median(slopes), lower, upper
    return np.median(kept[:count]), lower, upper


def _reach_arrays(z, rows, cols, stations):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    z = np.asarray(z)
    stations = np.asarray(stations, dtype=np.float64)
    if not rows.ndim == 1 or not rows.shape == cols.shape == z.shape == stations.shape:
        raise ValueError(f"z, rows, cols and stations must be 1-D and the same length, not {z.shape}, {rows.shape}, "
                         f"{cols.shape} and {stations.shape}.")
    return z, rows, cols, stations


def local_average_slopes(z, rows, cols, stations, dx: float, dy: float, distance: int) -> np.ndarray:
    """Each of a reach's cells' slope, the mean of its slopes to the reach's other cells within `distance` cells of
    it along each axis, from 0.0001 to 0.5 (legacy get_local_average_stream_slope_information).

    z is the DEM at the cells, rows and cols are the cells, which should be all of the reach's cells, and stations are
    their distances along the stream from one end (along_stream_stations).
    """
    z, rows, cols, stations = _reach_arrays(z, rows, cols, stations)
    if rows.size == 0:
        return np.empty(0)
    return _local_average_slopes(z, rows, cols, stations, float(dx), float(dy), int(distance))


def reach_median_slope(z, rows, cols, stations, dx: float, dy: float, distance: int, low_percentile: float = 25,
                       high_percentile: float = 75) -> tuple[float, float, float]:
    """A reach's slope, and the percentiles its slope is taken between (legacy
    get_reach_median_stream_slope_information).

    The slopes are between each pair of the reach's cells within `distance` cells of each other along each axis,
    leaving out those that round to 0 at 1e-8. The reach's slope is the median of those between the two percentiles,
    or of all of them if none is. A reach with fewer than two cells or no slopes has 0.0002 for all three. The
    arguments are as for local_average_slopes.
    """
    z, rows, cols, stations = _reach_arrays(z, rows, cols, stations)
    if rows.size == 0:
        return UNRESOLVED_SLOPE, UNRESOLVED_SLOPE, UNRESOLVED_SLOPE
    slope, lower, upper = _reach_median_slope(z, rows, cols, stations, float(dx), float(dy), int(distance),
                                              float(low_percentile), float(high_percentile))
    return float(slope), float(lower), float(upper)


def corrected_local_slopes(local_slopes, lower: float, upper: float) -> np.ndarray:
    """Local average slopes limited to their reach's percentiles (legacy local_average_corrected)."""
    local_slopes = np.asarray(local_slopes, dtype=np.float64)
    return np.where(local_slopes < lower, lower, np.where(local_slopes > upper, upper, local_slopes))


def fill_unresolved_reach_slopes(network: nx.DiGraph, slopes: dict, lower: dict, upper: dict) -> set:
    """Give the reaches whose two percentiles are equal their neighbours' slope, in place, and return the reaches
    still left without one.

    A reach takes the mean of the slopes of its upstream neighbour with the most reaches above it and its downstream
    neighbour, or the one of those it has, among neighbours with a slope of their own. Its percentiles become that
    slope too. This repeats while any reach gets filled, each time from the reaches filled before. network runs from
    each reach to the one downstream of it, and reaches not in it aren't filled.
    """
    unresolved = {reach for reach in slopes if lower[reach] == upper[reach]}
    while True:
        filled = {}
        for reach in sorted(unresolved):
            if reach not in network:
                continue
            upstream = [p for p in network.predecessors(reach) if p in slopes and p not in unresolved]
            downstream = [s for s in network.successors(reach) if s in slopes and s not in unresolved]
            neighbour_slopes = []
            if upstream:
                main_stem = max(upstream, key=lambda p: (len(nx.ancestors(network, p)), -p))
                neighbour_slopes.append(slopes[main_stem])
            if downstream:
                neighbour_slopes.append(slopes[downstream[0]])
            if neighbour_slopes:
                filled[reach] = sum(neighbour_slopes) / len(neighbour_slopes)
        if not filled:
            return unresolved
        for reach, slope in filled.items():
            slopes[reach] = lower[reach] = upper[reach] = slope
        unresolved -= filled.keys()


def end_point_slope(line, length: float, dem: np.ndarray, geotransform) -> float:
    """The drop between the ends of a reach's line over its length in metres, at least 0.0001 (legacy
    line_slope_from_dem, for the end_points method).

    line is a shapely LineString or MultiLineString in the DEM's coordinates, of which only the longest part counts.
    geotransform is the DEM's GDAL geotransform, for a north-up raster. Each end is the DEM at the cell holding it,
    and an end off the raster or without data (at or below -9999, or NaN) moves along the line, 2% of its length at
    a time, to the first point with data, which shortens the length between them. NaN if no point has any.
    """
    from shapely.geometry import LineString, MultiLineString

    if line is None or line.is_empty or not length > 0.0:
        return math.nan
    if isinstance(line, MultiLineString):
        whole = line.length
        line = max(line.geoms, key=lambda part: part.length)
        length *= line.length / whole
    elif not isinstance(line, LineString):
        line = LineString(line)
    if len(line.coords) < 2:
        return math.nan

    z_start, start = _first_elevation_along(line, dem, geotransform)
    z_end, end = _first_elevation_along(LineString(line.coords[::-1]), dem, geotransform)
    if math.isnan(z_start) or math.isnan(z_end):
        return math.nan
    along = (1.0 - end - start) * length
    if not along > 0.0:
        return math.nan
    return max(round(abs(z_end - z_start) / along, 8), MIN_SLOPE)


def _first_elevation_along(line, dem, geotransform, step=0.02) -> tuple[float, float]:
    """The DEM at the first point with data along the line from its start, and how far along it is as a share of
    the line's length."""
    for i in range(int(round(1 / step)) + 1):
        fraction = i * step
        point = line.interpolate(fraction, normalized=True)
        z = _elevation_at(point.x, point.y, dem, geotransform)
        if not math.isnan(z):
            return z, fraction
    return math.nan, math.nan


def _elevation_at(x, y, dem, geotransform) -> float:
    col = math.floor((x - geotransform[0]) / geotransform[1])
    row = math.floor((y - geotransform[3]) / geotransform[5])
    if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
        return math.nan
    z = float(dem[row, col])
    return z if math.isfinite(z) and z > NO_DATA else math.nan
