from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from arc.cross_section import CrossSection
from arc.xsection.sampling import _compute_dem_coordinates, sample_cross_section

# Most tests use 10 m cells and a 100 m cross section centred on cell (50, 50) of a 101 x 101 raster.
# Ordinates step one cell along whichever axis the section crosses fastest: 10 m along a row or column,
# so 5 fit on each side of the stream cell, and one cell diagonal (14.14 m) at 45 degrees. Rows increase
# downward, so a stream_direction of 0 flows east and pi / 2 flows south.
CELL = 10.0
LENGTH = 100.0
ROW = COL = 50
STEPS = np.arange(-5, 6)  # ordinate positions from the left end to the right end


@pytest.mark.parametrize(
    ("stream_direction", "right_bank_step"),
    [
        pytest.param(0.0, (0, -1), id="flowing-east-right-bank-north"),
        pytest.param(np.pi / 2, (1, 0), id="flowing-south-right-bank-east"),
        pytest.param(np.pi, (0, 1), id="flowing-west-right-bank-south"),
        pytest.param(3 * np.pi / 2, (-1, 0), id="flowing-north-right-bank-west"),
    ],
)
def test_cross_section_runs_perpendicular_to_the_stream(stream_direction: float, right_bank_step: tuple[int, int]) -> None:
    """Looking upstream, the ordinates run from the left bank to the right bank, one cell apart."""
    cols, rows, spacing = _compute_dem_coordinates(ROW, COL, stream_direction, LENGTH, CELL, CELL)

    col_step, row_step = right_bank_step
    assert spacing == pytest.approx(CELL)
    np.testing.assert_allclose(cols, COL + STEPS * col_step)
    np.testing.assert_allclose(rows, ROW + STEPS * row_step)


def test_diagonal_cross_section_steps_through_cell_centres() -> None:
    """A stream flowing south-east gets a south-west to north-east section, one cell diagonal per ordinate."""
    cols, rows, spacing = _compute_dem_coordinates(ROW, COL, np.pi / 4, LENGTH, CELL, CELL)

    assert spacing == pytest.approx(CELL * np.sqrt(2))
    np.testing.assert_allclose(cols, [47, 48, 49, 50, 51, 52, 53])
    np.testing.assert_allclose(rows, [53, 52, 51, 50, 49, 48, 47])


def test_cross_section_is_perpendicular_in_metres_when_cells_are_not_square() -> None:
    """With 10 m wide, 20 m tall cells, a 30 degree stream still gets a section at right angles on the ground.

    The section crosses a 10 m column every 20 m and a 20 m row every 23 m, so it steps one column per ordinate.
    """
    stream_direction = np.radians(30.0)
    cols, rows, spacing = _compute_dem_coordinates(ROW, COL, stream_direction, LENGTH, 10.0, 20.0)

    east_m = (cols - COL) * 10.0
    south_m = (rows - ROW) * 20.0
    along_stream_m = east_m * np.cos(stream_direction) + south_m * np.sin(stream_direction)

    assert spacing == pytest.approx(20.0)
    np.testing.assert_allclose(cols, COL + np.arange(-2, 3))
    np.testing.assert_allclose(along_stream_m, 0.0, atol=1e-9)
    np.testing.assert_allclose(np.hypot(east_m, south_m), spacing * np.array([2, 1, 0, 1, 2]))


@pytest.mark.parametrize("length", [30.0, 50.0, 100.0, 150.0])
def test_stream_cell_is_the_middle_ordinate_for_any_length(length: float) -> None:
    cols, rows, spacing = _compute_dem_coordinates(ROW, COL, np.radians(20.0), length, CELL, CELL)
    middle = cols.size // 2

    assert cols.size % 2 == 1
    assert (cols[middle], rows[middle]) == (COL, ROW)
    np.testing.assert_allclose(np.hypot(np.diff(cols) * CELL, np.diff(rows) * CELL), spacing)
    assert (cols.size - 1) * spacing <= length


def test_samples_a_v_shaped_valley() -> None:
    """A south-flowing channel on column 50, with valley sides rising 1 m per 10 m cell."""
    cells_from_channel = np.abs(np.arange(101) - COL)
    dem = np.tile(100.0 + cells_from_channel, (101, 1))
    manning_n = np.tile(np.where(cells_from_channel == 0, 0.035, 0.1), (101, 1))

    xs = sample_cross_section(dem, manning_n, ROW, COL, np.pi / 2, LENGTH, CELL, CELL)

    np.testing.assert_allclose(xs.elevations, [105, 104, 103, 102, 101, 100, 101, 102, 103, 104, 105])
    np.testing.assert_allclose(xs.mannings_n, [0.1] * 5 + [0.035] + [0.1] * 5)
    assert xs.ordinate_distance == pytest.approx(CELL)


@pytest.mark.parametrize(
    ("stream_direction", "expected_elevations"),
    [
        pytest.param(np.pi / 2, [110, 108, 106, 104, 102, 100, 101, 102, 103, 104, 105], id="flowing-south-left-bank-is-west"),
        pytest.param(3 * np.pi / 2, [105, 104, 103, 102, 101, 100, 102, 104, 106, 108, 110], id="flowing-north-left-bank-is-east"),
    ],
)
def test_left_bank_looking_upstream_comes_first(stream_direction: float, expected_elevations: list[float]) -> None:
    """The west side of the valley rises 2 m per cell and the east side 1 m per cell."""
    cells_from_channel = np.arange(101) - COL
    dem = np.tile(100.0 + np.where(cells_from_channel < 0, -2.0 * cells_from_channel, cells_from_channel), (101, 1))

    xs = sample_cross_section(dem, np.full_like(dem, 0.035), ROW, COL, stream_direction, LENGTH, CELL, CELL)

    np.testing.assert_allclose(xs.elevations, expected_elevations)


def test_ordinates_beyond_the_raster_are_walls() -> None:
    """An 11 x 11 raster reaches 5 cells either side of its centre, so a 200 m section runs off both edges.

    Off-raster ordinates get an elevation of 9999, a wall that water cannot spread past. The cells on the
    raster's edge keep their real values.
    """
    dem = np.full((11, 11), 100.0)

    xs = sample_cross_section(dem, np.full_like(dem, 0.035), 5, 5, np.pi / 2, 200.0, CELL, CELL)

    off_raster = np.abs(np.arange(-10, 11)) > 5
    np.testing.assert_allclose(xs.elevations, np.where(off_raster, 9999.0, 100.0))
    np.testing.assert_allclose(xs.mannings_n[~off_raster], 0.035)


def scipy_cross_section(dem, manning_n, row, col, stream_direction, length, dx, dy):
    """What sample_cross_section gave when it interpolated with scipy."""
    cols, rows, spacing = _compute_dem_coordinates(row, col, stream_direction, length, dx, dy)
    return (map_coordinates(dem, [rows, cols], order=1, mode="constant", cval=9999),
            map_coordinates(manning_n, [rows, cols], order=1, mode="constant", cval=9999), spacing)


def test_values_are_map_coordinates_s_bit_for_bit() -> None:
    """Random rasters, directions and cells, with NaNs, float32, views and centres off the raster."""
    rng = np.random.default_rng(11)
    cells = [(10.0, 10.0), (7.0, 13.0), (23.0, 31.0), (0.0002777, 0.0001944)]
    for trial in range(2000):
        shape = tuple(int(s) for s in rng.integers(1, 50, 2))
        dem = rng.normal(100.0, 5.0, shape)
        manning_n = rng.uniform(0.01, 0.2, shape)
        if trial % 3 == 0:
            dem[rng.random(shape) < 0.05] = np.nan
        if trial % 2 == 0:
            dem = dem.astype(np.float32)
        if trial % 4 == 1:
            manning_n = manning_n.astype(np.float32)
        if trial % 7 == 0:
            dem = dem[::-1]
        dx, dy = cells[trial % len(cells)]
        row, col = int(rng.integers(-3, shape[0] + 3)), int(rng.integers(-3, shape[1] + 3))
        stream_direction = float(rng.uniform(-7.0, 7.0)) if trial % 5 else int(rng.integers(-8, 9)) * np.pi / 4
        length = float(rng.uniform(0.0, 800.0)) * max(dx, dy) / CELL

        elevations, mannings_n, spacing = scipy_cross_section(dem, manning_n, row, col, stream_direction, length, dx, dy)
        xs = sample_cross_section(dem, manning_n, row, col, stream_direction, length, dx, dy)

        assert xs.ordinate_distance == spacing
        assert xs.elevations.dtype == elevations.dtype and xs.mannings_n.dtype == mannings_n.dtype
        np.testing.assert_array_equal(xs.elevations, elevations)
        np.testing.assert_array_equal(xs.mannings_n, mannings_n)


def test_a_nan_reaches_the_ordinates_that_interpolate_with_it() -> None:
    """Like map_coordinates, an ordinate is interpolated from its cell and the next one along each axis, even with no
    weight on the next, and on the last row or column from it and the one before. So a NaN in column 9 of 11 reaches
    the ordinates in columns 8, 9 and 10."""
    dem = np.full((11, 11), 100.0)
    dem[5, 9] = np.nan

    xs = sample_cross_section(dem, np.ones_like(dem), 5, 5, np.pi / 2, 100.0, CELL, CELL)

    np.testing.assert_array_equal(xs.elevations, [100.0] * 8 + [np.nan] * 3)


def test_rasters_that_aren_t_float_are_sampled_as_float64() -> None:
    """Where map_coordinates would have rounded an integer raster's values."""
    cells = np.arange(121).reshape(11, 11)

    xs = sample_cross_section(cells, cells, 5, 5, np.radians(80.0), 100.0, CELL, CELL)

    expected, _, _ = scipy_cross_section(cells.astype(np.float64), cells.astype(np.float64), 5, 5, np.radians(80.0),
                                         100.0, CELL, CELL)
    assert xs.elevations.dtype == xs.mannings_n.dtype == np.float64
    np.testing.assert_array_equal(xs.elevations, expected)


def _best_microseconds(function, calls: int = 500, repeats: int = 5) -> float:
    function()
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        best = min(best, (time.perf_counter() - start) / calls)
    return best * 1e6


def test_sampling_is_faster_than_scipy_and_legacy() -> None:
    """A 5 km cross section of 10 m cells, at the angle legacy snaps 42 degrees to."""
    rng = np.random.default_rng(0)
    dem = (100.0 + rng.normal(0.0, 0.3, (1200, 1200))).astype(np.float32)
    manning_n = np.full(dem.shape, 0.035)
    params = {"d_x_section_distance": 5000.0, "dx": CELL, "dy": CELL, "d_degree_manipulation": 0.0,
              "d_degree_interval": 0.0, "i_boundary_number": 0, "nrows": 1200, "ncols": 1200,
              "b_FindBanksBasedOnLandCover": False, "i_lc_water_value": 80, "d_bathymetry_trapzoid_height": 0.2,
              "b_bathy_use_banks": False, "s_output_bathymetry_path": ""}
    old = CrossSection(CELL, CELL, dem, np.ones(dem.shape, dtype=np.uint8), None, params)
    old.associate_with_precomputed_index_arrays(*CrossSection.create_cross_section_ordinates(params))
    direction = 7 * np.pi / 30 + np.pi / 2

    new = _best_microseconds(lambda: sample_cross_section(dem, manning_n, 600, 600, direction, 5000.0, CELL, CELL))
    scipy = _best_microseconds(lambda: scipy_cross_section(dem, manning_n, 600, 600, direction, 5000.0, CELL, CELL))
    legacy = _best_microseconds(lambda: old.set_cross_section(600, 600, 7, 7 * np.pi / 30))

    assert new < scipy / 2
    assert new < legacy
