from __future__ import annotations

import numpy as np
import pytest

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
