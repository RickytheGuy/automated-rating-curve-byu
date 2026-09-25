from __future__ import annotations

import math
import time

import numpy as np
import pytest

from arc.Automated_Rating_Curve_Generator import get_stream_direction_information
from arc.cross_section import CrossSection
from arc.hydraulics import top_widths
from arc.xsection.orientation import (TEST_DEPTH, angle_offsets, downhill_stream_direction, narrowest_cross_section,
                                      narrowest_direction, stream_direction)
from arc.xsection.sampling import sample_cross_section, sample_elevations

CELL = 10.0
REACH = 7


def stream_raster(cells, shape=(61, 61)):
    streams = np.zeros(shape, dtype=np.int64)
    for r, c in cells:
        streams[r, c] = REACH
    return streams


def rasterised_line(angle, offset=0.0, size=101, half_length=40):
    """A straight stream through the middle cell, rasterised by rounding points every 0.02 cells along it."""
    centre = size // 2
    cells = {(centre, centre)}
    for t in np.linspace(-half_length, half_length, 4001):
        cells.add((int(math.floor(centre + t * math.sin(angle) + offset * math.cos(angle) + 0.5)),
                   int(math.floor(centre + t * math.cos(angle) - offset * math.sin(angle) + 0.5))))
    return stream_raster(cells, (size, size)), centre


def axis_difference(a: float, b: float) -> float:
    """How far apart two lines' directions are, in radians, whichever way along them each points."""
    d = (a - b) % math.pi
    return min(d, math.pi - d)


# --- The stream's direction ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("step", "expected"), [((0, 1), 0.0), ((1, 0), math.pi / 2), ((1, 1), math.pi / 4),
                                               ((-1, 1), 3 * math.pi / 4)])
def test_a_straight_stream_runs_along_its_cells(step, expected) -> None:
    streams = stream_raster([(30 + step[0] * k, 30 + step[1] * k) for k in range(-10, 11)])

    assert stream_direction(streams, 30, 30, 5, CELL, CELL) == pytest.approx(expected, abs=1e-12)


def test_the_direction_is_in_metres_where_cells_are_not_square() -> None:
    """With 10 m wide, 20 m tall cells, a stream one row down for two columns across runs at 45 degrees on the
    ground. Legacy fitted it in cells, at 26.6 degrees."""
    streams = stream_raster([(30 + k, 30 + 2 * k) for k in range(-5, 6)])

    assert stream_direction(streams, 30, 30, 10, 10.0, 20.0) == pytest.approx(math.pi / 4, abs=1e-12)
    assert get_stream_direction_information(30, 30, streams, 11)[0] == pytest.approx(math.atan(0.5), abs=1e-12)


def test_rows_and_columns_are_treated_alike() -> None:
    """Transposing the raster mirrors a stream's direction about 45 degrees. Legacy's fit of rows against columns
    didn't: on straight streams at random angles it moved by up to 5 degrees."""
    rng = np.random.default_rng(0)
    new_error = legacy_error = 0.0
    for _ in range(400):
        streams, centre = rasterised_line(rng.uniform(0.0, math.pi), rng.uniform(-0.5, 0.5))
        transposed = streams.T.copy()

        new_error = max(new_error, axis_difference(stream_direction(transposed, centre, centre, 10, CELL, CELL),
                                                   math.pi / 2 - stream_direction(streams, centre, centre, 10, CELL,
                                                                                  CELL)))
        legacy_error = max(legacy_error, axis_difference(
            get_stream_direction_information(centre, centre, transposed, 10)[0],
            math.pi / 2 - get_stream_direction_information(centre, centre, streams, 10)[0]))

    assert new_error < 1e-12
    assert legacy_error > math.radians(3.0)


def test_straight_streams_are_measured_as_well_near_the_columns_as_near_the_rows() -> None:
    rng = np.random.default_rng(1)
    errors = {"rows": [], "columns": []}
    for _ in range(400):
        angle = rng.uniform(0.0, math.pi)
        streams, centre = rasterised_line(angle, rng.uniform(-0.5, 0.5))
        near = "rows" if min(angle, math.pi - angle) < math.pi / 4 else "columns"
        errors[near].append(axis_difference(stream_direction(streams, centre, centre, 10, CELL, CELL), angle))

    rows, columns = np.degrees(np.mean(errors["rows"])), np.degrees(np.mean(errors["columns"]))
    assert rows < 0.5 and columns < 0.5
    assert abs(rows - columns) < 0.1


def test_a_stream_with_one_step_sideways() -> None:
    """Ten cells down a column, stepping one column over half way, and the same cells along a row. Legacy's came
    out 78.7 and 8.3 degrees from the rows, which don't mirror each other."""
    along_a_column = stream_raster([(r, 30 if r <= 30 else 31) for r in range(25, 35)])
    along_a_row = along_a_column.T.copy()

    down = stream_direction(along_a_column, 30, 30, 5, CELL, CELL)
    across = stream_direction(along_a_row, 30, 30, 5, CELL, CELL)

    assert down == pytest.approx(math.pi / 2 - across, abs=1e-12)
    assert math.degrees(across) == pytest.approx(8.34, abs=0.01)
    assert math.degrees(get_stream_direction_information(30, 30, along_a_column, 5)[0]) == pytest.approx(78.69,
                                                                                                         abs=0.01)
    assert math.degrees(get_stream_direction_information(30, 30, along_a_row, 5)[0]) == pytest.approx(8.28, abs=0.01)


def test_the_square_reaches_as_far_each_way() -> None:
    """A cell of the reach 5 rows and 5 columns on counts, as does one 5 back. Legacy missed the one 5 on."""
    ahead = stream_raster([(30, 30), (35, 35)])
    behind = stream_raster([(30, 30), (25, 25)])

    assert stream_direction(ahead, 30, 30, 5, CELL, CELL) == pytest.approx(math.pi / 4)
    assert stream_direction(behind, 30, 30, 5, CELL, CELL) == pytest.approx(math.pi / 4)
    assert get_stream_direction_information(30, 30, ahead, 5) == (0.0, 0.0)
    assert get_stream_direction_information(30, 30, behind, 5)[0] == pytest.approx(math.pi / 4)


def test_a_lone_cell_s_cross_section_runs_along_its_row() -> None:
    """As legacy's did, though legacy called its stream direction 0."""
    streams = stream_raster([(30, 30), (30, 40)])

    assert stream_direction(streams, 30, 30, 5, CELL, CELL) == pytest.approx(math.pi / 2)
    assert get_stream_direction_information(30, 30, streams, 5) == (0.0, 0.0)


@pytest.mark.parametrize(("fall", "expected"), [(0.01, 0.0), (-0.01, math.pi), (0.0, 0.0)])
def test_with_a_dem_the_direction_points_downhill(fall, expected) -> None:
    streams = stream_raster([(30, c) for c in range(20, 41)])
    dem = np.tile(100.0 - fall * CELL * np.arange(61.0), (61, 1))

    assert downhill_stream_direction(streams, dem, 30, 30, 5, CELL, CELL) == pytest.approx(expected, abs=1e-12)


def test_only_a_stream_cell_has_a_direction() -> None:
    streams = stream_raster([(30, c) for c in range(20, 41)])

    with pytest.raises(ValueError, match="isn't a stream cell"):
        stream_direction(streams, 31, 30, 5, CELL, CELL)
    with pytest.raises(ValueError, match="same shape"):
        downhill_stream_direction(streams, np.zeros((5, 5)), 30, 30, 5, CELL, CELL)


def test_the_square_stops_at_the_raster_s_edge() -> None:
    """Nothing wraps round to the far side of the raster."""
    streams = stream_raster([(0, c) for c in range(5, 16)] + [(19, 10), (0, 19)], shape=(20, 20))

    assert stream_direction(streams, 0, 10, 5, CELL, CELL) == pytest.approx(0.0, abs=1e-12)


# --- The angle search ---------------------------------------------------------------------------------------------


def legacy_offsets(degree_manipulation: float, degree_interval: float) -> np.ndarray:
    class Holder:
        pass

    holder = Holder()
    CrossSection.set_angles_to_test(holder, {"d_degree_manipulation": degree_manipulation,
                                             "d_degree_interval": degree_interval})
    return np.asarray(holder.l_angles_to_test)


@pytest.mark.parametrize(("manipulation", "interval"), [(6.0, 1.0), (1.1, 1.0), (0.0, 1.0), (10.0, 0.0), (7.0, 2.0),
                                                       (180.0, 90.0)])
def test_the_offsets_are_legacy_s(manipulation, interval) -> None:
    np.testing.assert_array_equal(angle_offsets(manipulation, interval), legacy_offsets(manipulation, interval))


def test_the_search_samples_as_sample_cross_section_does() -> None:
    """The elevations the search compares are sample_cross_section's, with NaNs, in float32 and off the raster."""
    rng = np.random.default_rng(3)
    for trial in range(400):
        shape = tuple(rng.integers(3, 40, 2))
        dem = rng.normal(100.0, 5.0, shape)
        if trial % 3 == 0:
            dem[rng.random(shape) < 0.05] = np.nan
        if trial % 2 == 0:
            dem = dem.astype(np.float32)
        dx, dy = float(rng.choice([10.0, 7.0, 23.0])), float(rng.choice([10.0, 13.0, 31.0]))
        row, col = int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1]))
        direction = float(rng.uniform(-7.0, 7.0)) if trial % 5 else int(rng.integers(0, 8)) * math.pi / 4
        length = float(rng.uniform(20.0, 600.0))

        expected = sample_cross_section(dem, np.ones_like(dem), row, col, direction, length, dx, dy)
        elevations, spacing = sample_elevations(dem, row, col, direction, length, dx, dy)

        assert spacing == expected.ordinate_distance
        assert elevations.dtype == expected.elevations.dtype
        np.testing.assert_array_equal(elevations, expected.elevations)


def v_valley(angle: float, size: int = 201) -> np.ndarray:
    """A straight channel at an angle through cell (100, 100), its sides rising 0.2 m per metre from it."""
    rows, cols = np.mgrid[0:size, 0:size]
    across = np.abs(-(cols - 100) * CELL * math.sin(angle) + (rows - 100) * CELL * math.cos(angle))
    return 100.0 + 0.2 * across


@pytest.mark.parametrize(("start", "expected"), [(10, 30), (20, 30), (30, 30), (40, 30), (50, 30), (0, 20)])
def test_the_search_turns_the_cross_section_across_the_channel(start, expected) -> None:
    """A channel at 30 degrees, searched within 20 degrees either way of the start, 5 degrees at a time."""
    dem = v_valley(math.radians(30.0))

    direction = narrowest_direction(dem, 100, 100, math.radians(start), 1000.0, CELL, CELL, angle_offsets(40, 5))

    assert math.degrees(direction) == pytest.approx(expected, abs=1e-9)


def test_the_search_matches_legacy_along_rows_and_columns() -> None:
    """A channel down column 100, from a stream direction wrongly flowing east. Turned a quarter turn, the cross
    section runs along a row and crosses the channel, which legacy found too, at the same width."""
    dem = v_valley(math.pi / 2)
    params = {"d_x_section_distance": 1000.0, "dx": CELL, "dy": CELL, "d_degree_manipulation": 180.0,
              "d_degree_interval": 90.0, "i_boundary_number": 0, "nrows": 201, "ncols": 201,
              "b_FindBanksBasedOnLandCover": False, "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": 0.2,
              "b_bathy_use_banks": False, "s_output_bathymetry_path": ""}
    old = CrossSection(CELL, CELL, dem, np.zeros(dem.shape, dtype=np.uint8), None, params)
    old.associate_with_precomputed_index_arrays(*CrossSection.create_cross_section_ordinates(params))
    old.set_cross_section(100, 100, 15, math.pi / 2)  # legacy's cross section for a stream flowing east
    old.test_angles_and_reset_cross_section(100, 100)

    direction, xs = narrowest_cross_section(dem, np.ones_like(dem), 100, 100, 0.0, 1000.0, CELL, CELL,
                                            angle_offsets(180, 90))
    left, right = top_widths(xs.elevations, xs.ordinate_distance, xs.elevations[xs.elevations.size // 2] + TEST_DEPTH)

    assert axis_difference(direction - math.pi / 2, old.d_xs_direction) == pytest.approx(0.0, abs=1e-12)
    assert left + right == pytest.approx(old.calculate_top_width_of_wse(old.get_thalweg() + TEST_DEPTH), abs=1e-3)
    assert left + right == pytest.approx(2.0 * 0.5 / 0.2, rel=1e-12)


def test_equally_narrow_directions_keep_the_first() -> None:
    """In a round hollow a cross section along a row is as narrow as one along a column."""
    rows, cols = np.mgrid[0:201, 0:201]
    dem = 100.0 + 0.2 * CELL * np.hypot(rows - 100, cols - 100)

    assert narrowest_direction(dem, 100, 100, 0.0, 1000.0, CELL, CELL, [0.0, math.pi / 2]) == 0.0
    assert narrowest_direction(dem, 100, 100, 0.0, 1000.0, CELL, CELL, [math.pi / 2, 0.0]) == math.pi / 2


def uncapped_choice(dem, row, col, start, length, offsets) -> int:
    """Which offset legacy's rule would pick: the smallest top width, however far each cross section reaches."""
    widths = []
    for offset in offsets:
        elevations, spacing = sample_elevations(dem, row, col, start + offset, length, CELL, CELL)
        left, right = top_widths(elevations, spacing, elevations[elevations.size // 2] + TEST_DEPTH)
        widths.append(left + right)
    return int(np.argmin(widths))


def test_water_reaching_every_cross_section_s_ends_doesnt_choose_the_direction() -> None:
    """On a flat floodplain the water 0.5 m up reaches both ends of every cross section, whose lengths differ: 50 m
    a side along rows and columns, 42 m on the diagonals. Legacy's rule picks a diagonal, the shortest."""
    dem = np.full((201, 201), 100.0)
    dem[100, 100] = 99.8
    offsets = angle_offsets(90, 45)

    assert narrowest_direction(dem, 100, 100, 0.0, 100.0, CELL, CELL, offsets) == 0.0
    assert uncapped_choice(dem, 100, 100, 0.0, 100.0, offsets) == 1


def channel_by_a_floodplain(shape, row, col, angle=math.radians(30.0)) -> np.ndarray:
    """A channel at an angle through (row, col). On its north-east side the ground rises 1 cm per metre, so the water
    0.5 m up reaches 50 m out, and on its south-west side a floodplain rises 1 mm per metre, so the water reaches 500
    m out. Each side is a plane, which linear interpolation between cells reproduces exactly, so a cross section
    turned from square to the channel is wider by exactly 1 / cos of the turn."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    across = -(cols - col) * CELL * math.sin(angle) + (rows - row) * CELL * math.cos(angle)  # > 0 to the south-west
    return np.where(across > 0.0, 100.0 + 0.001 * across, 100.0 - 0.01 * across)


def test_a_side_reaching_the_raster_s_edge_leaves_the_other_side_to_choose() -> None:
    """The channel three cells from the raster's west edge, so the floodplain floods to the edge, which each
    direction's cross section reaches at a different distance. The north-east side still shows which way the
    channel runs. Legacy's rule picked the direction that reached the edge soonest."""
    dem = channel_by_a_floodplain((101, 60), 50, 3)
    offsets = angle_offsets(40, 5)
    start = math.radians(20.0)

    assert math.degrees(narrowest_direction(dem, 50, 3, start, 600.0, CELL, CELL, offsets)) == pytest.approx(30.0)
    assert math.degrees(start + offsets[uncapped_choice(dem, 50, 3, start, 600.0, offsets)]) == pytest.approx(40.0)


def test_water_reaching_missing_ground_counts_as_reaching_the_end() -> None:
    """As at the raster's edge, but with no data (NaN) 30 m out on the floodplain."""
    dem = channel_by_a_floodplain((101, 101), 50, 50)
    rows, cols = np.mgrid[0:101, 0:101]
    dem[-(cols - 50) * CELL * 0.5 + (rows - 50) * CELL * math.sqrt(0.75) > 30.0] = np.nan

    direction = narrowest_direction(dem, 50, 50, math.radians(20.0), 600.0, CELL, CELL, angle_offsets(40, 5))

    assert math.degrees(direction) == pytest.approx(30.0)


def test_offsets_half_a_turn_apart_are_the_same_cross_section() -> None:
    dem = v_valley(math.radians(30.0))
    start = math.radians(30.0)

    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [0.0, math.pi]) == start
    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [0.2, 0.2 + math.pi, 0.2 - math.pi]) \
        == pytest.approx(start + 0.2)


def test_the_width_is_compared_at_the_test_depth() -> None:
    """A channel at 30 degrees whose sides rise 1 cm per metre for 100 m and then level off. Half a metre up, the
    water reaches 50 m out square to the channel, so the channel's direction is narrowest. 1.5 m up it floods past
    the sides to the ends of every cross section, and the start is kept."""
    angle = math.radians(30.0)
    rows, cols = np.mgrid[0:201, 0:201]
    across = np.abs(-(cols - 100) * CELL * math.sin(angle) + (rows - 100) * CELL * math.cos(angle))
    dem = 100.0 + 0.01 * np.minimum(across, 100.0)
    start, offsets = math.radians(20.0), angle_offsets(20, 10)

    assert math.degrees(narrowest_direction(dem, 100, 100, start, 600.0, CELL, CELL, offsets)) == pytest.approx(30.0)
    assert narrowest_direction(dem, 100, 100, start, 600.0, CELL, CELL, offsets, test_depth=1.5) == start


def test_an_offset_beyond_a_quarter_turn_is_taken_half_a_turn_round() -> None:
    dem = v_valley(math.radians(30.0))
    start = math.radians(30.0)

    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [math.pi + 0.1]) == pytest.approx(start + 0.1)
    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [-0.1 - math.pi]) == pytest.approx(start - 0.1)


def test_the_narrowest_cross_section_is_sampled_in_its_direction() -> None:
    dem = v_valley(math.radians(30.0))
    manning_n = np.full(dem.shape, 0.035)

    direction, xs = narrowest_cross_section(dem, manning_n, 100, 100, math.radians(12.0), 1000.0, CELL, CELL,
                                            angle_offsets(40, 3))
    expected = sample_cross_section(dem, manning_n, 100, 100, direction, 1000.0, CELL, CELL)

    np.testing.assert_array_equal(xs.elevations, expected.elevations)
    np.testing.assert_array_equal(xs.mannings_n, expected.mannings_n)
    assert xs.ordinate_distance == expected.ordinate_distance
    assert math.degrees(direction) == pytest.approx(30.0)


def test_no_offsets_to_try_keeps_the_direction() -> None:
    dem = v_valley(math.radians(30.0))

    assert narrowest_direction(dem, 100, 100, 0.3, 1000.0, CELL, CELL, angle_offsets(1.1, 1.0)) == 0.3
    assert narrowest_direction(dem, 100, 100, 0.3, 1000.0, CELL, CELL, []) == 0.3


# --- Speed --------------------------------------------------------------------------------------------------------


def _seconds_per_call(function, calls: int) -> float:
    function()
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def test_the_search_is_faster_than_legacy_s() -> None:
    """Seven directions over a 5 km cross section of 10 m cells, sampling the chosen one, against legacy's search
    and resampling."""
    rng = np.random.default_rng(0)
    dem = (100.0 + rng.normal(0.0, 0.3, (1200, 1200))).astype(np.float32)
    manning_n = np.full(dem.shape, 0.035)
    params = {"d_x_section_distance": 5000.0, "dx": CELL, "dy": CELL, "d_degree_manipulation": 6.0,
              "d_degree_interval": 1.0, "i_boundary_number": 0, "nrows": 1200, "ncols": 1200,
              "b_FindBanksBasedOnLandCover": False, "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": 0.2,
              "b_bathy_use_banks": False, "s_output_bathymetry_path": ""}
    old = CrossSection(CELL, CELL, dem, np.ones(dem.shape, dtype=np.uint8), None, params)
    old.associate_with_precomputed_index_arrays(*CrossSection.create_cross_section_ordinates(params))
    old.set_cross_section(600, 600, 7, 7 * math.pi / 30)
    offsets = angle_offsets(6.0, 1.0)

    new = _seconds_per_call(lambda: narrowest_cross_section(dem, manning_n, 600, 600, 0.7, 5000.0, CELL, CELL,
                                                            offsets), calls=200)
    legacy = _seconds_per_call(lambda: old.test_angles_and_reset_cross_section(600, 600), calls=200)

    assert new < legacy
