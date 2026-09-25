"""Distances along a reach's stream cells.

A reach's stream cells join their eight neighbours: dx apart along a row, dy along a column, and hypot(dx, dy) across
a diagonal. The distance along the stream between two cells is the length of the shortest path between them through
the reach's cells. Where the cells don't all join up, as where a reach leaves the raster and comes back or its
rasterisation skips a cell, the path jumps each gap in a straight line, across the narrowest part of the gap.

A station is a cell's distance along the stream from one end of the reach.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def _row_index(rows, cols):
    """The cells in order of row and then column, their columns in that order, and where each row's cells start in
    it: row r's are at positions row_start[r - first_row] to row_start[r - first_row + 1]."""
    first_row, first_col = rows.min(), cols.min()
    order = np.argsort((rows - first_row) * (cols.max() - first_col + 1) + (cols - first_col))
    row_start = np.zeros(rows.max() - first_row + 2, np.int64)
    for r in rows:
        row_start[r - first_row + 1] += 1
    for k in range(1, row_start.size):
        row_start[k] += row_start[k - 1]
    return order, cols[order], row_start, first_row


@njit(cache=True, error_model="numpy")
def _columns_between(sorted_cols, first, last, low, high):
    """The positions, from first to last, of one row's cells whose columns are from low to high."""
    if last - first <= 16:  # a stream crosses most rows in a cell or two
        lo = first
        while lo < last and sorted_cols[lo] < low:
            lo += 1
        hi = lo
        while hi < last and sorted_cols[hi] <= high:
            hi += 1
        return lo, hi
    row = sorted_cols[first:last]
    return first + np.searchsorted(row, low), first + np.searchsorted(row, high, side="right")


@njit(cache=True, error_model="numpy")
def _adjacency(rows, cols, dx, dy):
    """Each cell's neighbours among the cells, and how far away each is, as offsets into flat arrays: cell i's
    neighbours are neighbours[start[i]:start[i + 1]]. A cell at the same place as another is its neighbour at 0 m."""
    n = rows.size
    order, sorted_cols, row_start, first_row = _row_index(rows, cols)
    last_row = first_row + row_start.size - 2
    start = np.zeros(n + 1, np.int64)
    for i in range(n):
        count = -1  # not itself
        for row in range(max(rows[i] - 1, first_row), min(rows[i] + 1, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - 1, cols[i] + 1)
            count += hi - lo
        start[i + 1] = start[i] + count
    neighbours = np.empty(start[n], np.int64)
    lengths = np.empty(start[n])
    for i in range(n):
        k = start[i]
        for row in range(max(rows[i] - 1, first_row), min(rows[i] + 1, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - 1, cols[i] + 1)
            for p in range(lo, hi):
                j = order[p]
                if j != i:
                    neighbours[k] = j
                    lengths[k] = math.hypot((cols[j] - cols[i]) * dx, (rows[j] - rows[i]) * dy)
                    k += 1
    return start, neighbours, lengths


@njit(cache=True, error_model="numpy")
def _push(heap_distance, heap_cell, size, distance, cell):
    """Add a cell to a binary min-heap on distance, returning the new size."""
    i = size
    while i > 0:
        parent = (i - 1) // 2
        if not distance < heap_distance[parent]:
            break
        heap_distance[i] = heap_distance[parent]
        heap_cell[i] = heap_cell[parent]
        i = parent
    heap_distance[i] = distance
    heap_cell[i] = cell
    return size + 1


@njit(cache=True, error_model="numpy")
def _pop(heap_distance, heap_cell, size):
    """Remove the nearest cell from the heap, returning its distance, the cell and the new size."""
    distance, cell = heap_distance[0], heap_cell[0]
    size -= 1
    last_distance, last_cell = heap_distance[size], heap_cell[size]
    i = 0
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and heap_distance[child + 1] < heap_distance[child]:
            child += 1
        if not heap_distance[child] < last_distance:
            break
        heap_distance[i] = heap_distance[child]
        heap_cell[i] = heap_cell[child]
        i = child
    heap_distance[i] = last_distance
    heap_cell[i] = last_cell
    return distance, cell, size


@njit(cache=True, error_model="numpy")
def _distances_from(rows, cols, dx, dy, source, start, neighbours, lengths):
    """Dijkstra's shortest paths from one cell through the adjacency, jumping gaps (see the module notes)."""
    n = rows.size
    distance = np.full(n, np.inf)
    settled = np.zeros(n, np.bool_)
    heap_distance = np.empty(neighbours.size + n + 1)
    heap_cell = np.empty(neighbours.size + n + 1, np.int64)
    distance[source] = 0.0
    size = _push(heap_distance, heap_cell, 0, 0.0, source)
    remaining = n
    while remaining > 0:
        if size == 0:
            # The cells left don't join the ones reached. Jump the narrowest gap to one of them in a straight line.
            gap, near, far = np.inf, -1, -1
            for u in range(n):
                if settled[u]:
                    for v in range(n):
                        if not settled[v]:
                            length = math.hypot((cols[v] - cols[u]) * dx, (rows[v] - rows[u]) * dy)
                            if length < gap:
                                gap, near, far = length, u, v
            distance[far] = distance[near] + gap
            size = _push(heap_distance, heap_cell, size, distance[far], far)
        d, u, size = _pop(heap_distance, heap_cell, size)
        if settled[u]:
            continue
        settled[u] = True
        remaining -= 1
        for k in range(start[u], start[u + 1]):
            v = neighbours[k]
            if d + lengths[k] < distance[v]:
                distance[v] = d + lengths[k]
                size = _push(heap_distance, heap_cell, size, distance[v], v)
    return distance


@njit(cache=True, error_model="numpy")
def _along_stream_distances(rows, cols, dx, dy, source):
    start, neighbours, lengths = _adjacency(rows, cols, dx, dy)
    return _distances_from(rows, cols, dx, dy, source, start, neighbours, lengths)


@njit(cache=True, error_model="numpy")
def _along_stream_stations(rows, cols, dx, dy):
    """Stations from one end of the reach: the cell farthest along the stream from the first cell."""
    start, neighbours, lengths = _adjacency(rows, cols, dx, dy)
    end = np.argmax(_distances_from(rows, cols, dx, dy, 0, start, neighbours, lengths))
    return _distances_from(rows, cols, dx, dy, end, start, neighbours, lengths)


@njit(cache=True, error_model="numpy")
def _nearest_cell(rows, cols, dx, dy, other_rows, other_cols):
    """The cell nearest in a straight line to any of the other cells (the first of any that tie)."""
    best, nearest = np.inf, -1
    for i in range(rows.size):
        for j in range(other_rows.size):
            length = math.hypot((cols[i] - other_cols[j]) * dx, (rows[i] - other_rows[j]) * dy)
            if length < best:
                best, nearest = length, i
    return nearest


def _cell_arrays(rows, cols) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.shape != cols.shape or rows.ndim != 1:
        raise ValueError(f"rows and cols must be 1-D and the same length, not {rows.shape} and {cols.shape}.")
    return rows, cols


def along_stream_distances(rows, cols, dx: float, dy: float, source: int) -> np.ndarray:
    """Each cell's distance along the stream from cell `source`, in metres."""
    rows, cols = _cell_arrays(rows, cols)
    return _along_stream_distances(rows, cols, float(dx), float(dy), int(source))


def along_stream_stations(rows, cols, dx: float, dy: float) -> np.ndarray:
    """Each cell's distance along the stream from one end of the reach, in metres. Which end isn't defined."""
    rows, cols = _cell_arrays(rows, cols)
    if rows.size == 0:
        return np.empty(0)
    return _along_stream_stations(rows, cols, float(dx), float(dy))


def downstream_order(rows, cols, dx: float, dy: float, *, downstream_cells=None, upstream_cells=None
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """The reach's cells in order from upstream to downstream, and their stations, which start at 0 upstream.

    The downstream end is the cell nearest the reach downstream (downstream_cells, a (rows, cols) pair). Without
    that, the upstream end is the cell nearest the reach upstream (upstream_cells), and the downstream end is the
    cell farthest along the stream from it. None if neither is given, or if the reach has no cells. Cells equally far
    along stay in the order given. A reach of one cell is in order already.
    """
    rows, cols = _cell_arrays(rows, cols)
    dx, dy = float(dx), float(dy)
    if rows.size <= 1:
        return np.arange(rows.size), np.zeros(rows.size)
    downstream = _neighbour_cell(rows, cols, dx, dy, downstream_cells)
    if downstream < 0:
        upstream = _neighbour_cell(rows, cols, dx, dy, upstream_cells)
        if upstream < 0:
            return None
        downstream = int(np.argmax(_along_stream_distances(rows, cols, dx, dy, upstream)))
    to_downstream = _along_stream_distances(rows, cols, dx, dy, downstream)
    order = np.argsort(-to_downstream, kind="stable")
    return order, to_downstream[order[0]] - to_downstream[order]


def _neighbour_cell(rows, cols, dx, dy, cells) -> int:
    if cells is None:
        return -1
    other_rows, other_cols = _cell_arrays(*cells)
    if other_rows.size == 0:
        return -1
    return int(_nearest_cell(rows, cols, dx, dy, other_rows, other_cols))


def stream_cells_by_reach(streams: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The rows and columns of each reach's cells in a stream raster, whose cells above 0 hold their reach's ID."""
    rows, cols = np.nonzero(streams > 0)
    ids = streams[rows, cols]
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], ids.size]
    return {int(ids[s]): (rows[order[s:e]].astype(np.int64), cols[order[s:e]].astype(np.int64))
            for s, e in zip(starts, ends)}
