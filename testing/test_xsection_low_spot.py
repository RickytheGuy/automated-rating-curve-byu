from __future__ import annotations

import math
import time

import numpy as np
import pytest

from arc.cross_section import CrossSection
from arc.xsection.low_spot import low_spot_cell
from arc.xsection.sampling import sample_cross_section

# 10 m cells, and a 200 m cross section: 10 ordinates either side. A stream flowing south (pi / 2) has its cross
# section along the row, and one flowing east (0) along the column.
CELL = 10.0
LENGTH = 200.0
SOUTH, NORTH, EAST, WEST = math.pi / 2, 3 * math.pi / 2, 0.0, math.pi


def valley(channel_col, size=41) -> np.ndarray:
    """A V-shaped valley down a column, its sides rising 0.2 m per metre."""
    return np.tile(100.0 + 0.2 * CELL * np.abs(np.arange(size) - channel_col), (size, 1))


def legacy_cell(dem, row, col, along_row, low_spot_range, length=LENGTH):
    params = {"d_x_section_distance": length, "dx": CELL, "dy": CELL, "d_degree_manipulation": 0.0,
              "d_degree_interval": 0.0, "i_boundary_number": 0, "nrows": dem.shape[0], "ncols": dem.shape[1],
              "b_FindBanksBasedOnLandCover": False, "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": 0.2,
              "b_bathy_use_banks": False, "s_output_bathymetry_path": ""}
    old = CrossSection(CELL, CELL, dem, np.zeros(dem.shape, dtype=np.uint8), None, params)
    old.associate_with_precomputed_index_arrays(*CrossSection.create_cross_section_ordinates(params))
    old.set_cross_section(row, col, 0, 0.0) if along_row else old.set_cross_section(row, col, 15, math.pi / 2)
    old.adjust_cross_section_to_lowest_point(low_spot_range)
    return tuple(int(v) for v in old.get_row_col())


def test_the_cross_section_moves_to_the_lowest_ground_nearby() -> None:
    dem = valley(23)

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 23)
    assert low_spot_cell(dem, 20, 20, NORTH, LENGTH, CELL, CELL, 5) == (20, 23)


def test_a_range_of_n_looks_n_ordinates_out() -> None:
    """With the channel 5 cells away, a range of 5 reaches it and 4 gets as near as it can. Legacy needed a range of
    6 to reach it, and with 1 didn't move at all."""
    dem = valley(25)

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 25)
    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 4) == (20, 24)
    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 1) == (20, 21)
    assert legacy_cell(dem, 20, 20, True, 5) == (20, 24)
    assert legacy_cell(dem, 20, 20, True, 6) == (20, 25)
    assert legacy_cell(dem, 20, 20, True, 1) == (20, 20)


def test_the_stream_cell_stays_where_nothing_nearby_is_lower() -> None:
    dem = valley(20)
    dem[20, 23] = 100.0  # as low, but not lower

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 20)
    assert low_spot_cell(valley(23), 20, 20, SOUTH, LENGTH, CELL, CELL, 0) == (20, 20)


@pytest.mark.parametrize(("direction", "lower", "expected"), [
    pytest.param(SOUTH, [(20, 17), (20, 23)], (20, 23), id="along-a-row-flowing-south"),
    pytest.param(NORTH, [(20, 17), (20, 23)], (20, 23), id="along-a-row-flowing-north"),
    pytest.param(EAST, [(17, 20), (23, 20)], (23, 20), id="along-a-column-flowing-east"),
    pytest.param(WEST, [(17, 20), (23, 20)], (23, 20), id="along-a-column-flowing-west"),
])
def test_of_two_equally_low_ordinates_as_near_legacy_s_first_side_wins(direction, lower, expected) -> None:
    """Legacy looked first towards the higher rows, or along a row the higher columns, whichever way the stream
    flows."""
    dem = np.full((41, 41), 100.0)
    for cell in lower:
        dem[cell] = 99.0

    assert low_spot_cell(dem, 20, 20, direction, LENGTH, CELL, CELL, 5) == expected


def test_of_equally_low_ordinates_the_nearest_wins() -> None:
    dem = np.full((41, 41), 100.0)
    dem[20, 24] = dem[20, 18] = dem[20, 15] = 99.0

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 18)


def test_the_low_spot_matches_legacy_s_along_rows_and_columns() -> None:
    """Random ground, whole metres for plenty of ties. Legacy's range is one more, for the ordinate it didn't reach."""
    rng = np.random.default_rng(5)
    for trial in range(300):
        dem = rng.integers(95, 105, (41, 41)).astype(np.float64) if trial % 2 else rng.normal(100.0, 2.0, (41, 41))
        row, col = (int(v) for v in rng.integers(12, 29, 2))
        along_row = trial % 4 < 2
        low_spot_range = int(rng.integers(1, 11))
        expected = legacy_cell(dem, row, col, along_row, low_spot_range + 1)
        for direction in ((SOUTH, NORTH) if along_row else (EAST, WEST)):
            assert low_spot_cell(dem, row, col, direction, LENGTH, CELL, CELL, low_spot_range) == expected


def test_nan_and_ordinates_off_the_raster_are_never_the_low_spot() -> None:
    """A DEM in feet, 12000 ft up, two cells from its east edge: the ordinates off the raster sample as 9999."""
    dem = np.full((41, 23), 12000.0)
    dem[20, 18] = np.nan

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 10) == (20, 20)
    assert sample_cross_section(dem, dem, 20, 20, SOUTH, LENGTH, CELL, CELL).elevations[-1] == 9999.0


def test_ground_at_or_below_sea_level_can_be_the_low_spot() -> None:
    """Legacy only moved to ground above 0 m."""
    dem = valley(23) - 101.0

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 23)


def test_an_oblique_cross_section_moves_to_the_cell_nearest_the_low_ordinate() -> None:
    """A stream flowing south-south-east: its cross section steps a column at a time and 0.4 of a row, so the lowest
    ordinate, three out, is 1.2 rows up, nearest the row above."""
    stream_direction = math.pi / 2 - math.atan(0.4)
    xs = sample_cross_section(np.zeros((41, 41)), np.zeros((41, 41)), 20, 20, stream_direction, LENGTH, CELL, CELL)
    dem = np.full((41, 41), 100.0)
    dem[19, 23] = dem[18, 23] = 90.0

    assert xs.ordinate_distance == pytest.approx(CELL * math.hypot(1.0, 0.4))
    assert low_spot_cell(dem, 20, 20, stream_direction, LENGTH, CELL, CELL, 5) == (19, 23)


def test_the_cells_of_a_cross_section_on_cells_that_aren_t_square() -> None:
    """Cells 10 m wide and 20 m tall: a stream flowing east has its cross section down the column, a row per
    ordinate, 20 m apart."""
    dem = np.tile(100.0 + np.abs(np.arange(41) - 23.0)[:, None], (1, 41))

    assert low_spot_cell(dem, 20, 20, EAST, 400.0, 10.0, 20.0, 5) == (23, 20)
    assert low_spot_cell(dem, 20, 20, WEST, 400.0, 10.0, 20.0, 5) == (23, 20)


def _best_seconds(function, calls: int = 200, repeats: int = 5) -> float:
    function()
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def test_moving_and_sampling_again_is_faster_than_legacy_s() -> None:
    """A 5 km cross section of 10 m cells, whose stream cell is 3 cells from the channel."""
    dem = np.tile(100.0 + 0.02 * CELL * np.abs(np.arange(1200) - 603), (1200, 1)).astype(np.float32)
    manning_n = np.full(dem.shape, 0.035)
    params = {"d_x_section_distance": 5000.0, "dx": CELL, "dy": CELL, "d_degree_manipulation": 0.0,
              "d_degree_interval": 0.0, "i_boundary_number": 0, "nrows": 1200, "ncols": 1200,
              "b_FindBanksBasedOnLandCover": False, "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": 0.2,
              "b_bathy_use_banks": False, "s_output_bathymetry_path": ""}
    old = CrossSection(CELL, CELL, dem, np.ones(dem.shape, dtype=np.uint8), None, params)
    old.associate_with_precomputed_index_arrays(*CrossSection.create_cross_section_ordinates(params))

    def legacy():
        old.set_cross_section(600, 600, 0, 0.0)
        old.adjust_cross_section_to_lowest_point(11)

    def new():
        row, col = low_spot_cell(dem, 600, 600, SOUTH, 5000.0, CELL, CELL, 10)
        return sample_cross_section(dem, manning_n, row, col, SOUTH, 5000.0, CELL, CELL)

    legacy()
    assert old.get_row_col() == (600, 603) and new().elevations[250] == 100.0
    assert _best_seconds(new) < _best_seconds(legacy)
