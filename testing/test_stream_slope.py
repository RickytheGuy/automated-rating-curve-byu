from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString

from arc import Automated_Rating_Curve_Generator as legacy
from arc.xsection import slope as slope_module
from arc.xsection.slope import (MIN_SLOPE, UNRESOLVED_SLOPE, corrected_local_slopes, end_point_slope,
                                fill_unresolved_reach_slopes, local_average_slopes, reach_median_slope)
from arc.xsection.stream_path import (along_stream_distances, along_stream_stations, downstream_order,
                                      stream_cells_by_reach)

CELL = 10.0
REACH = 7


def reach_raster(cells, shape=(60, 60)):
    """A stream raster holding REACH at the given (row, col) cells, and those cells' rows and columns."""
    streams = np.zeros(shape, dtype=np.int64)
    rows = np.array([r for r, _ in cells], dtype=np.int64)
    cols = np.array([c for _, c in cells], dtype=np.int64)
    streams[rows, cols] = REACH
    return streams, rows, cols


def hairpin():
    """A stream that runs east along row 10, turns down through column 16, and comes back west along row 13, so the
    two arms are 30 m apart but up to 238 m apart along the stream."""
    return ([(10, c) for c in range(5, 16)] + [(11, 16), (12, 16)] + [(13, c) for c in range(15, 4, -1)])


def winding_reach(rng, n_cells, size, slope=0.001, noise=0.01):
    """A meandering stream rasterised a third of a cell at a time, with a DEM that falls `slope` per metre along
    its cells plus noise, and rougher ground around it."""
    dem = rng.normal(300.0, 1.0, (size, size)).astype(np.float32)
    streams = np.zeros((size, size), dtype=np.int64)
    x = y = 0.0
    cells = []
    while len(cells) < n_cells:
        heading = 0.9 * math.sin(len(cells) / 12.0) + 0.7
        x += math.cos(heading) / 3
        y += math.sin(heading) / 3
        cell = (20 + int(round(y)), 20 + int(round(x)))
        if not cells or cell != cells[-1]:
            cells.append(cell)
    along = 0.0
    for k, (r, c) in enumerate(cells):
        if k:
            along += CELL * math.hypot(r - cells[k - 1][0], c - cells[k - 1][1])
        streams[r, c] = REACH
        dem[r, c] = np.float32(250.0 - slope * along + rng.normal(0.0, noise))
    return dem, streams


# --- Distances along the stream -----------------------------------------------------------------------------------


def test_distances_follow_the_stream_one_cell_at_a_time() -> None:
    """With 10 m wide, 20 m tall cells: a step along the row, one across a diagonal, and one down the column."""
    rows, cols = [0, 0, 1, 2], [0, 1, 2, 2]

    distances = along_stream_distances(rows, cols, 10.0, 20.0, 0)

    np.testing.assert_allclose(distances, [0.0, 10.0, 10.0 + math.hypot(10.0, 20.0), 30.0 + math.hypot(10.0, 20.0)],
                               rtol=1e-15)


def test_a_path_takes_the_diagonal_across_a_corner() -> None:
    """Round a corner of whole cells, the path cuts the corner cell's diagonal (legacy's test of the same)."""
    distances = along_stream_distances([0, 0, 1, 2], [0, 1, 1, 1], 1.0, 1.0, 3)

    np.testing.assert_allclose(distances, [1.0 + math.sqrt(2.0), 2.0, 1.0, 0.0], rtol=1e-15)


def test_a_gap_in_the_cells_is_jumped_in_a_straight_line() -> None:
    rows, cols = [0] * 8, [0, 1, 2, 3, 4, 8, 9, 10]

    np.testing.assert_allclose(along_stream_distances(rows, cols, CELL, CELL, 0), [0, 10, 20, 30, 40, 80, 90, 100])


def test_stations_run_from_one_end_of_the_reach_to_the_other() -> None:
    cells = hairpin()
    order = np.random.default_rng(0).permutation(len(cells))
    rows = np.array([cells[k][0] for k in order])
    cols = np.array([cells[k][1] for k in order])

    stations = along_stream_stations(rows, cols, CELL, CELL)

    total = 100.0 + math.hypot(10, 10) + 10.0 + math.hypot(10, 10) + 100.0  # along each arm, and round the bend
    ends = sorted(stations[[list(order).index(0), list(order).index(len(cells) - 1)]])
    assert ends == pytest.approx([0.0, total], abs=1e-9)
    assert np.sort(stations)[-1] == pytest.approx(total, abs=1e-9)


def test_a_reach_is_ordered_from_the_cell_by_the_reach_downstream() -> None:
    """Legacy's test: a reach that bends on its way to the reach below it, starting at (0, 0)."""
    order, stations = downstream_order([0, 0, 1, 2], [0, 1, 1, 1], 1.0, 1.0, downstream_cells=([3], [1]))

    assert order.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(stations, [0.0, math.sqrt(2.0) - 1.0, math.sqrt(2.0), 1.0 + math.sqrt(2.0)])


def test_a_reach_without_one_downstream_is_ordered_away_from_the_one_upstream() -> None:
    """The cell nearest the reach upstream is the upstream end, and the cell farthest from it downstream."""
    cells = hairpin()
    rows = np.array([r for r, _ in cells])[::-1]
    cols = np.array([c for _, c in cells])[::-1]

    order, stations = downstream_order(rows, cols, CELL, CELL, upstream_cells=([10], [4]))

    assert (rows[order[0]], cols[order[0]]) == (10, 5)
    assert (rows[order[-1]], cols[order[-1]]) == (13, 5)
    assert np.all(np.diff(stations) > 0.0)


def test_a_reach_needs_a_reach_next_to_it_to_know_its_downstream_end() -> None:
    assert downstream_order([0, 0], [0, 1], CELL, CELL) is None
    assert downstream_order([0, 0], [0, 1], CELL, CELL, downstream_cells=([], [])) is None
    order, stations = downstream_order([5], [5], CELL, CELL)
    assert order.tolist() == [0] and stations.tolist() == [0.0]


def test_ordering_matches_legacy_on_winding_reaches() -> None:
    """Legacy's order and stations for reaches running into the reach below them, their cells shuffled."""
    rng = np.random.default_rng(11)
    compared = 0
    while compared < 150:
        heading = rng.uniform(0.0, 2.0 * math.pi)
        x = y = 0.0
        cells = [(0, 0)]
        for _ in range(int(rng.integers(20, 1500))):
            heading += rng.normal(0.0, 0.15)
            x += math.cos(heading) / 3
            y += math.sin(heading) / 3
            cell = (int(round(y)), int(round(x)))
            if cell == cells[-1]:
                continue
            if cell in cells:
                break
            cells.append(cell)
        beyond = (cells[-1][0] + int(round(math.sin(heading) * 1.4)), cells[-1][1] + int(round(math.cos(heading) * 1.4)))
        if len(cells) < 2 or beyond in cells or beyond == cells[-1]:
            continue
        shuffle = rng.permutation(len(cells))
        rows = np.array([cells[k][0] for k in shuffle]) + 200
        cols = np.array([cells[k][1] for k in shuffle]) + 200
        entries = [{"row": int(r), "col": int(c)} for r, c in zip(rows, cols)]
        graph = nx.DiGraph()
        graph.add_edge(1, 2)
        expected_order, expected_stations = legacy._order_reach_stream_cells_from_network(
            graph, 1, entries, {1: entries, 2: [{"row": beyond[0] + 200, "col": beyond[1] + 200}]},
            np.arange(len(cells)), np.zeros(len(cells)), CELL, CELL)

        order, stations = downstream_order(rows, cols, CELL, CELL,
                                           downstream_cells=([beyond[0] + 200], [beyond[1] + 200]))

        assert order.tolist() == expected_order.tolist()
        np.testing.assert_allclose(stations, expected_stations, rtol=0, atol=1e-9)
        compared += 1


def test_cells_are_grouped_by_reach() -> None:
    streams = np.zeros((5, 6), dtype=np.int64)
    streams[1, 1:4] = 3
    streams[3, 2:5] = 1

    cells = stream_cells_by_reach(streams)

    assert sorted(cells) == [1, 3]
    assert cells[3][0].tolist() == [1, 1, 1] and cells[3][1].tolist() == [1, 2, 3]
    assert cells[1][0].tolist() == [3, 3, 3] and cells[1][1].tolist() == [2, 3, 4]


# --- Slopes -------------------------------------------------------------------------------------------------------


def hairpin_slopes(distance: int):
    """The hairpin with a bed falling exactly 1 mm per metre along the stream, and both slopes of each kind."""
    streams, rows, cols = reach_raster(hairpin())
    stations = along_stream_stations(rows, cols, CELL, CELL)
    dem = np.full(streams.shape, 500.0)
    dem[rows, cols] = 100.0 - 0.001 * (stations - stations.min())
    z = dem[rows, cols]
    local = local_average_slopes(z, rows, cols, stations, CELL, CELL, distance)
    median = reach_median_slope(z, rows, cols, stations, CELL, CELL, distance)
    legacy_local = [legacy.get_local_average_stream_slope_information(r, c, dem, streams, CELL, CELL, distance)
                    for r, c in zip(rows, cols)]
    legacy_median = legacy.get_reach_median_stream_slope_information(dem, rows, cols, CELL, CELL, distance, 25, 75)
    return local, median, np.array(legacy_local), legacy_median


def test_slopes_are_the_drop_over_the_distance_along_the_stream() -> None:
    """Across the hairpin's two arms the straight line is 30 m but the stream up to 238 m, so legacy's slopes
    between them were up to eight times the bed's: its local averages reached 3.8 times, and its reach's 75th
    percentile 3.2 times. Along the stream every slope is the bed's."""
    local, (median, lower, upper), legacy_local, legacy_median = hairpin_slopes(distance=5)

    np.testing.assert_allclose(local, 0.001, rtol=1e-9)
    assert (median, lower, upper) == pytest.approx((0.001, 0.001, 0.001), rel=1e-9)
    assert legacy_local.max() > 0.0035
    assert legacy_median[2] > 0.003


def test_slopes_match_legacy_on_straight_reaches() -> None:
    """Along a straight row, column or diagonal the stream is the straight line, so the reach median and its
    percentiles are legacy's. So are the local averages wherever legacy's square was complete (see below)."""
    rng = np.random.default_rng(5)
    distance = 10
    for step in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
        cells = [(130 + step[0] * k if step[0] < 0 else 50 + step[0] * k, 50 + step[1] * k) for k in range(80)]
        streams, rows, cols = reach_raster(cells, shape=(260, 260))
        dem = rng.normal(200.0, 1.0, streams.shape).astype(np.float32)
        dem[rows, cols] = (150.0 - 0.02 * math.hypot(*step) * np.arange(80) + rng.normal(0.0, 0.02, 80)).astype(np.float32)
        z = dem[rows, cols]
        stations = along_stream_stations(rows, cols, CELL, CELL)

        assert reach_median_slope(z, rows, cols, stations, CELL, CELL, distance) == \
            legacy.get_reach_median_stream_slope_information(dem, rows, cols, CELL, CELL, distance, 25, 75)
        local = local_average_slopes(z, rows, cols, stations, CELL, CELL, distance)
        if step == (-1, 1):
            continue  # every cell has another at the edge of legacy's square that it missed
        compared = 0
        for k in range(rows.size):
            box = streams[rows[k] - distance:rows[k] + distance + 1, cols[k] - distance:cols[k] + distance + 1]
            if not (box[-1, :] == REACH).any() and not (box[:, -1] == REACH).any():
                assert local[k] == legacy.get_local_average_stream_slope_information(
                    rows[k], cols[k], dem, streams, CELL, CELL, distance)
                compared += 1
        assert compared >= 10


def test_the_local_average_reaches_as_far_downstream_as_upstream() -> None:
    """On a straight row, the average at column 15 of the slopes to columns 10 to 20. Legacy's square stopped at
    column 19, so it was the average to columns 10 to 19."""
    cells = [(10, c) for c in range(31)]
    streams, rows, cols = reach_raster(cells)
    dem = np.full(streams.shape, 500.0)
    dem[rows, cols] = 100.0 - 0.001 * (cols - 15.0) ** 2
    stations = along_stream_stations(rows, cols, CELL, CELL)

    local = local_average_slopes(dem[rows, cols], rows, cols, stations, CELL, CELL, 5)

    def mean_slope(columns):
        return np.mean([abs(dem[10, 15] - dem[10, c]) / (CELL * abs(c - 15)) for c in columns if c != 15])

    assert local[15] == pytest.approx(mean_slope(range(10, 21)), rel=1e-12)
    assert legacy.get_local_average_stream_slope_information(10, 15, dem, streams, CELL, CELL, 5) == \
        pytest.approx(mean_slope(range(10, 20)), rel=1e-12)


@pytest.mark.parametrize(("drop", "expected"), [(0.0, MIN_SLOPE), (1e-4, MIN_SLOPE), (100.0, 0.5)])
def test_local_averages_are_kept_between_0p0001_and_0p5(drop: float, expected: float) -> None:
    rows, cols = np.array([0, 0]), np.array([0, 1])
    z = np.array([100.0, 100.0 - drop])

    assert local_average_slopes(z, rows, cols, along_stream_stations(rows, cols, CELL, CELL), CELL, CELL, 5) \
        == pytest.approx([expected, expected])


def test_the_reach_median_is_of_the_slopes_between_its_percentiles() -> None:
    """Six cells along a row make 15 pairs. The flat pair is left out, and the median is of the slopes between the
    25th and 75th percentiles."""
    rows, cols = np.zeros(6, dtype=np.int64), np.arange(6)
    z = np.array([10.0, 9.9, 9.9, 9.6, 9.5, 9.0])
    stations = along_stream_stations(rows, cols, CELL, CELL)

    slopes = np.round([abs(z[i] - z[k]) / (CELL * (k - i)) for i in range(6) for k in range(i + 1, 6)], 8)
    slopes = slopes[slopes > 0.0]
    lower, upper = np.round(np.percentile(slopes, [25, 75]), 8)
    kept = slopes[(slopes >= lower) & (slopes <= upper)]

    assert slopes.size == 14 and 0 < kept.size < slopes.size
    assert reach_median_slope(z, rows, cols, stations, CELL, CELL, 10) == pytest.approx(
        (np.median(kept), lower, upper), rel=1e-12)


def test_a_reach_with_two_slopes_takes_their_median() -> None:
    """Three cells with one flat pair leave slopes of 0.05 and 0.1, neither between their percentiles (0.0625 and
    0.0875). Legacy gave the reach 0.0002."""
    dem = np.zeros((3, 5))
    dem[1, 1:4] = [10.0, 10.0, 9.0]
    rows, cols = np.array([1, 1, 1]), np.array([1, 2, 3])

    result = reach_median_slope(dem[rows, cols], rows, cols, along_stream_stations(rows, cols, CELL, CELL), CELL,
                                CELL, 10)

    assert result == pytest.approx((0.075, 0.0625, 0.0875), rel=1e-12)
    assert legacy.get_reach_median_stream_slope_information(dem, rows, cols, CELL, CELL, 10, 25, 75) == \
        pytest.approx((UNRESOLVED_SLOPE, 0.0625, 0.0875))


@pytest.mark.parametrize("z", [[100.0], [100.0, 100.0, 100.0]])
def test_a_reach_without_slopes_is_unresolved(z) -> None:
    rows, cols = np.zeros(len(z), dtype=np.int64), np.arange(len(z))
    stations = along_stream_stations(rows, cols, CELL, CELL)

    assert reach_median_slope(np.array(z), rows, cols, stations, CELL, CELL, 10) == \
        (UNRESOLVED_SLOPE, UNRESOLVED_SLOPE, UNRESOLVED_SLOPE)


def test_corrected_slopes_are_kept_between_the_reach_percentiles() -> None:
    np.testing.assert_allclose(corrected_local_slopes([0.001, 0.003, 0.009], 0.002, 0.008), [0.002, 0.003, 0.008])


def test_a_slope_between_two_cells_is_never_over_less_than_the_straight_line() -> None:
    """Round a corner of whole cells the path's stations for (0, 0) and (0, 1) are only 0.41 cells apart, since the
    path cuts the corner, but the cells are a whole cell apart. Both of (0, 1)'s slopes are then 1 cm over 10 m."""
    rows, cols = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1])
    z = np.array([10.0, 9.99, 9.98, 9.97])
    stations = along_stream_distances(rows, cols, CELL, CELL, 3)

    slopes = local_average_slopes(z, rows, cols, stations, CELL, CELL, 1)

    assert stations[0] - stations[1] == pytest.approx(CELL * (math.sqrt(2.0) - 1.0))
    assert slopes[1] == pytest.approx(0.001, rel=1e-9)


def reach_with_a_long_row(rng):
    """A reach running 40 cells along a row and then winding away, with random elevations."""
    cells = [(20, c) for c in range(10, 50)]
    x, y, heading = 49.0, 20.0, 0.0
    while len(cells) < 220:
        heading = 0.9 * math.sin(len(cells) / 9.0) + 0.6
        x += math.cos(heading) / 3
        y += math.sin(heading) / 3
        cell = (int(round(y)), int(round(x)))
        if cell != cells[-1] and cell not in cells:
            cells.append(cell)
    order = rng.permutation(len(cells))
    rows = np.array([cells[k][0] for k in order], dtype=np.int64)
    cols = np.array([cells[k][1] for k in order], dtype=np.int64)
    z = 100.0 + rng.normal(0.0, 0.5, rows.size)
    return rows, cols, z, along_stream_stations(rows, cols, CELL, CELL)


def brute_force_length(rows, cols, stations, a, b):
    return max(abs(stations[a] - stations[b]), CELL * math.hypot(rows[a] - rows[b], cols[a] - cols[b]))


def test_each_pair_of_cells_within_the_distance_gives_one_slope() -> None:
    rows, cols, z, stations = reach_with_a_long_row(np.random.default_rng(8))
    distance = 6
    expected = []
    for a in range(rows.size):
        for b in range(a + 1, rows.size):
            if abs(rows[a] - rows[b]) <= distance and abs(cols[a] - cols[b]) <= distance:
                slope = np.round(abs(z[a] - z[b]) / brute_force_length(rows, cols, stations, a, b), 8)
                if slope > 0.0:
                    expected.append(slope)

    got = slope_module._pair_slopes(z, rows, cols, stations, CELL, CELL, distance)

    np.testing.assert_allclose(np.sort(got), np.sort(expected), rtol=1e-15)


def test_each_local_average_is_over_the_cells_within_the_distance() -> None:
    rows, cols, z, stations = reach_with_a_long_row(np.random.default_rng(9))
    distance = 6
    expected = []
    for a in range(rows.size):
        slopes = [abs(z[a] - z[b]) / brute_force_length(rows, cols, stations, a, b) for b in range(rows.size)
                  if b != a and abs(rows[a] - rows[b]) <= distance and abs(cols[a] - cols[b]) <= distance]
        expected.append(min(max(np.mean(slopes), MIN_SLOPE), 0.5))

    np.testing.assert_allclose(local_average_slopes(z, rows, cols, stations, CELL, CELL, distance), expected,
                               rtol=1e-12)


def test_slopes_at_a_percentile_count_as_between_them() -> None:
    """On a DEM rounded to centimetres slopes tie. Here two tie at the 75th percentile, and counting them makes the
    median 0.0015, as legacy's does, where leaving them out would make it 0.00133."""
    dem = np.zeros((3, 6))
    dem[1, 1:5] = [99.98, 99.97, 99.95, 99.94]
    rows, cols = np.ones(4, dtype=np.int64), np.arange(1, 5)
    stations = along_stream_stations(rows, cols, CELL, CELL)

    result = reach_median_slope(dem[rows, cols], rows, cols, stations, CELL, CELL, 10)

    assert result == legacy.get_reach_median_stream_slope_information(dem, rows, cols, CELL, CELL, 10, 25, 75)
    assert result[0] == pytest.approx(0.0015)


# --- Filling unresolved reaches from their neighbours -------------------------------------------------------------


def neighbours() -> nx.DiGraph:
    """Reach 3 has reaches 1 (with reach 5 above it) and 2 flowing into it, and flows into reach 4."""
    network = nx.DiGraph()
    network.add_edges_from([(5, 1), (1, 3), (2, 3), (3, 4)])
    return network


def test_an_unresolved_reach_takes_the_mean_of_its_main_stem_neighbours() -> None:
    slopes = {5: 0.004, 1: 0.003, 2: 0.009, 3: UNRESOLVED_SLOPE, 4: 0.001}
    lower = {5: 0.002, 1: 0.002, 2: 0.005, 3: UNRESOLVED_SLOPE, 4: 0.0005}
    upper = {5: 0.006, 1: 0.004, 2: 0.01, 3: UNRESOLVED_SLOPE, 4: 0.002}

    assert fill_unresolved_reach_slopes(neighbours(), slopes, lower, upper) == set()
    assert slopes[3] == lower[3] == upper[3] == pytest.approx(0.002)


def test_unresolved_reaches_fill_from_those_filled_before() -> None:
    """Reach 4 has only reach 3 next to it, so it fills once reach 3 has."""
    slopes = {5: 0.004, 1: 0.003, 2: 0.009, 3: UNRESOLVED_SLOPE, 4: UNRESOLVED_SLOPE}
    lower = {5: 0.002, 1: 0.002, 2: 0.005, 3: UNRESOLVED_SLOPE, 4: UNRESOLVED_SLOPE}
    upper = {**lower, 5: 0.006, 1: 0.004, 2: 0.01}

    assert fill_unresolved_reach_slopes(neighbours(), slopes, lower, upper) == set()
    assert slopes[3] == pytest.approx(0.003)
    assert slopes[4] == pytest.approx(0.003)


def test_the_main_stem_is_chosen_among_neighbours_with_a_slope() -> None:
    """Reach 1 has the most reaches above it but isn't on the raster, so reach 2 is the upstream neighbour. Legacy
    took reach 1, found no slope, and used reach 4 alone."""
    slopes = {2: 0.009, 3: UNRESOLVED_SLOPE, 4: 0.001}
    lower = {2: 0.005, 3: UNRESOLVED_SLOPE, 4: 0.0005}
    upper = {2: 0.01, 3: UNRESOLVED_SLOPE, 4: 0.002}

    fill_unresolved_reach_slopes(neighbours(), slopes, lower, upper)

    assert slopes[3] == pytest.approx(0.005)


def test_reaches_without_neighbours_stay_unresolved() -> None:
    network = nx.DiGraph()
    network.add_node(8)
    slopes, lower, upper = {8: UNRESOLVED_SLOPE, 9: 0.001}, {8: 0.1, 9: 0.001}, {8: 0.1, 9: 0.001}

    assert fill_unresolved_reach_slopes(network, slopes, lower, upper) == {8, 9}


# --- End points ---------------------------------------------------------------------------------------------------


GEOTRANSFORM = (0.0, 10.0, 0.0, 100.0, 0.0, -10.0)  # 10 m cells, the top left corner at (0, 100)


def end_point_dem() -> np.ndarray:
    """A 10 x 10 DEM falling 1 m per cell to the south east."""
    rows, cols = np.mgrid[0:10, 0:10]
    return 120.0 - rows - cols


def test_the_end_point_slope_is_the_drop_between_the_ends_over_the_length() -> None:
    line = LineString([(5.0, 95.0), (95.0, 5.0)])  # from cell (0, 0) to cell (9, 9)

    assert end_point_slope(line, 150.0, end_point_dem(), GEOTRANSFORM) == pytest.approx(18.0 / 150.0)
    slope_percent = legacy.line_slope_from_dem(line, end_point_dem(), GEOTRANSFORM, 150.0)[0]
    assert end_point_slope(line, 150.0, end_point_dem(), GEOTRANSFORM) == round(slope_percent / 100, 8)


def test_an_end_off_the_raster_moves_along_the_line_to_it() -> None:
    """A 140 m line along row 4 that ends 45 m east of the raster. In steps of 2% (2.8 m) from that end, the first
    point on the raster is 34% along, in column 9."""
    line = LineString([(5.0, 55.0), (145.0, 55.0)])
    dem = end_point_dem()

    slope = end_point_slope(line, 140.0, dem, GEOTRANSFORM)

    assert slope == pytest.approx((dem[4, 0] - dem[4, 9]) / (0.66 * 140.0))
    assert slope == round(legacy.line_slope_from_dem(line, dem, GEOTRANSFORM, 140.0)[0] / 100, 8)


def test_a_line_with_both_ends_off_the_raster_moves_both() -> None:
    """A 1 km line along row 4 from 40 m west of the raster. In steps of 2% (20 m) the first point on the raster is
    at its west edge, 4% along, and the last is 80 m east of it, 12% from the far end. Legacy only moved an end when
    the other was on the raster, and gave this line no slope."""
    line = LineString([(-40.0, 55.0), (960.0, 55.0)])
    dem = end_point_dem()

    assert end_point_slope(line, 1000.0, dem, GEOTRANSFORM) == pytest.approx((dem[4, 0] - dem[4, 8]) / 80.0)
    assert math.isnan(legacy.line_slope_from_dem(line, dem, GEOTRANSFORM, 1000.0)[0])


def test_an_end_without_data_moves_along_the_line_too() -> None:
    """Legacy used an end on the raster whatever the DEM held there."""
    line = LineString([(5.0, 55.0), (95.0, 55.0)])  # along row 4, from column 0 to column 9
    dem = end_point_dem()
    dem[4, 0] = -9999.0

    slope = end_point_slope(line, 90.0, dem, GEOTRANSFORM)

    assert slope == pytest.approx(8.0 / (0.94 * 90.0))  # the first point with data is 6% along, in column 1


def test_a_point_just_beyond_the_left_edge_is_off_the_raster() -> None:
    """3 m west of the raster is off it, so the line's first point on it is 4% along. Legacy's int() read the
    start from column 0."""
    dem = end_point_dem()
    line = LineString([(-3.0, 55.0), (97.0, 55.0)])

    assert end_point_slope(line, 100.0, dem, GEOTRANSFORM) == pytest.approx(9.0 / 96.0)
    assert legacy.line_slope_from_dem(line, dem, GEOTRANSFORM, 100.0)[0] / 100 == pytest.approx(9.0 / 100.0)


def test_a_line_with_no_data_at_either_end_has_no_slope() -> None:
    line = LineString([(200.0, 200.0), (300.0, 300.0)])

    assert math.isnan(end_point_slope(line, 141.0, end_point_dem(), GEOTRANSFORM))


def test_a_line_in_parts_uses_the_longest_part_and_its_share_of_the_length() -> None:
    """The long part runs down column 5 from row 0 to row 8 (80 m of the 100 m of line)."""
    line = MultiLineString([[(55.0, 95.0), (55.0, 15.0)], [(0.5, 0.5), (0.5, 20.5)]])

    assert end_point_slope(line, 200.0, end_point_dem(), GEOTRANSFORM) == pytest.approx(8.0 / 160.0)


def test_the_end_point_slope_is_at_least_0p0001() -> None:
    line = LineString([(5.0, 95.0), (95.0, 95.0)])
    dem = np.full((10, 10), 50.0)

    assert end_point_slope(line, 90.0, dem, GEOTRANSFORM) == MIN_SLOPE


# --- Speed --------------------------------------------------------------------------------------------------------


def _best_seconds(function, repeats: int = 5) -> float:
    function()
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def test_reach_slopes_are_faster_than_legacy_s() -> None:
    """A 2000-cell winding reach: the new slopes, the along-stream stations included, against legacy's local
    average at every cell and its reach median."""
    dem, streams = winding_reach(np.random.default_rng(5), 2000, size=1800)
    rows, cols = stream_cells_by_reach(streams)[REACH]
    z = dem[rows, cols]

    def new_local():
        local_average_slopes(z, rows, cols, along_stream_stations(rows, cols, CELL, CELL), CELL, CELL, 10)

    def new_median():
        reach_median_slope(z, rows, cols, along_stream_stations(rows, cols, CELL, CELL), CELL, CELL, 10)

    def old_local():
        for r, c in zip(rows, cols):
            legacy.get_local_average_stream_slope_information(r, c, dem, streams, CELL, CELL, 10)

    assert _best_seconds(new_local) < _best_seconds(old_local)
    assert _best_seconds(new_median) < _best_seconds(
        lambda: legacy.get_reach_median_stream_slope_information(dem, rows, cols, CELL, CELL, 10, 25, 75))
