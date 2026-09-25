from __future__ import annotations

import math
import time

import numpy as np
import pytest
from numba import njit

from arc import hydraulics
from arc.cross_section import (_calculate_stream_geometry_and_topwidth, _compound_section_conveyance,
                               calculate_discharge_from_wse)
from arc.hydraulics import (CompoundGeometry, ConveyanceTable, HydraulicGeometry, compound_geometry, discharge,
                            hydraulic_geometry, wse_for_discharge)
from arc.xsection.sampling import sample_cross_section
from arc.xsection.xsection import XSection

BED = 100.0
N = 0.035
SLOPE = 0.001
DRY = HydraulicGeometry(0.0, 0.0, 0.0, 0.0, 0.0)


def make_section(elevations, mannings_n=N, spacing=1.0) -> XSection:
    elevations = np.asarray(elevations, dtype=np.float64)
    mannings_n = np.broadcast_to(np.asarray(mannings_n, dtype=np.float64), elevations.shape).copy()
    return XSection(elevations, mannings_n, spacing)


def trapezoid(bottom_width: float, side_slope: float, ordinates_per_side: int = 60, spacing: float = 1.0) -> XSection:
    """A flat bed at 100 m, with banks rising 1 m for every side_slope metres across. A bottom width of 0 is a V."""
    x = spacing * np.arange(-ordinates_per_side, ordinates_per_side + 1)
    return make_section(BED + np.maximum(np.abs(x) - bottom_width / 2, 0.0) / side_slope, spacing=spacing)


def ridge_and_hollow() -> XSection:
    """A V channel whose right bank tops out in a ridge 3 m up, with a hollow behind the ridge lower than the bed."""
    left = BED + np.arange(10.0, 0.0, -1.0)
    right = [100.0, 101.0, 102.0, 103.0, 99.0, 99.0, 99.0, 104.0, 105.0, 106.0, 107.0]
    return make_section(np.concatenate([left, right]))


def flat_floodplain() -> XSection:
    """A 2 m deep V channel between perfectly flat floodplains 50 m wide, then valley walls."""
    distance = np.abs(np.arange(-60.0, 61.0))
    elevations = np.where(distance <= 2, BED + distance, BED + 2.0)
    return make_section(np.where(distance > 52, BED + 2.0 + (distance - 52), elevations))


def walled_bed() -> XSection:
    """A 20 m wide flat bed with 9999 m walls on both sides, like a section that runs off the raster."""
    elevations = np.full(25, BED)
    mannings_n = np.full(25, N)
    elevations[[0, 1, -2, -1]] = 9999.0
    mannings_n[[0, 1, -2, -1]] = 9999.0  # sampling gives off-raster ordinates a roughness of 9999 too
    return make_section(elevations, mannings_n)


def rough_valley() -> XSection:
    """A valley whose banks steepen and flatten but never dip, with a different roughness at every ordinate."""
    rng = np.random.default_rng(4)
    right = BED + np.concatenate([[0.0], np.cumsum(rng.uniform(0.01, 0.8, 80))])
    left = BED + np.concatenate([[0.0], np.cumsum(rng.uniform(0.01, 0.8, 80))])
    return make_section(np.concatenate([left[:0:-1], right]), rng.uniform(0.02, 0.12, 161), spacing=2.0)


def typical_section() -> XSection:
    """5 km at 10 m spacing (501 ordinates): a 20 m wide, 2 m deep channel in a valley with 2% side slopes."""
    distance = np.abs(10.0 * np.arange(-250, 251))
    return make_section(BED + np.minimum(distance, 10.0) * 0.2 + np.maximum(distance - 10.0, 0.0) * 0.02, spacing=10.0)


SHAPES = {
    "v-channel": lambda: trapezoid(0.0, 1.5),
    "trapezoid": lambda: trapezoid(10.0, 2.0),
    "ridge-and-hollow": ridge_and_hollow,
    "flat-floodplain": flat_floodplain,
    "walled-bed": walled_bed,
    "rough-valley": rough_valley,
}


def with_banks(xs: XSection, left: float, right: float) -> XSection:
    xs.left_bank_distance = left
    xs.right_bank_distance = right
    return xs


def channel_and_floodplains() -> XSection:
    """A 20 m wide bed, with banks rising 2 m over 2 m to flat floodplains 18 m wide on each side. The channel's
    banks are 12 m either side of the stream cell."""
    distance = np.abs(np.arange(-30.0, 31.0))
    return with_banks(make_section(BED + np.clip(distance - 10.0, 0.0, 2.0)), 12.0, 12.0)


def refine(xs: XSection, factor: int) -> XSection:
    """The same ground with `factor` times as many ordinates. Each new ordinate takes the Manning's n of the
    original ordinate nearer the stream cell, so every segment keeps its roughness."""
    size = xs.elevations.size
    position = np.arange((size - 1) * factor + 1) / factor
    nearer = np.where(position >= size // 2, np.floor(position), np.ceil(position)).astype(int)
    return XSection(np.interp(position, np.arange(size), xs.elevations), xs.mannings_n[nearer],
                    xs.ordinate_distance / factor)


def manning_conveyance(area: float, perimeter: float, mannings_n: float = N) -> float:
    return area * (area / perimeter) ** (2 / 3) / mannings_n


# Pairs of bank distances: on ordinates, between them, at the stream cell, and beyond the ends of the section
BANKS = [(3.0, 5.0), (2.5, 7.25), (0.0, 4.0), (0.0, 0.0), (6.0, 1e6)]


# --- Geometry ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side_slope", [1.0, 2.5])
@pytest.mark.parametrize("depth", [0.25, 1.0, 2.37])
def test_v_channel_matches_the_triangle_formulas(side_slope: float, depth: float) -> None:
    geometry = hydraulic_geometry(trapezoid(0.0, side_slope), depth=depth)

    area = side_slope * depth**2
    perimeter = 2 * depth * math.hypot(1, side_slope)
    assert geometry.area == pytest.approx(area, rel=1e-12)
    assert geometry.top_width == pytest.approx(2 * side_slope * depth, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(perimeter, rel=1e-12)
    assert geometry.hydraulic_radius == pytest.approx(area / perimeter, rel=1e-12)


@pytest.mark.parametrize("depth", [0.5, 1.3, 3.0])
def test_trapezoid_matches_the_trapezoid_formulas(depth: float) -> None:
    """A 10 m wide bed, with banks rising 1 m for every 2 m across."""
    geometry = hydraulic_geometry(trapezoid(10.0, 2.0), depth=depth)

    area = (10 + 2 * depth) * depth
    perimeter = 10 + 2 * depth * math.hypot(1, 2)
    assert geometry.area == pytest.approx(area, rel=1e-12)
    assert geometry.top_width == pytest.approx(10 + 4 * depth, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(perimeter, rel=1e-12)
    assert geometry.hydraulic_radius == pytest.approx(area / perimeter, rel=1e-12)


def test_banks_with_different_slopes() -> None:
    """The left bank rises 1 m for every 1 m across, and the right bank 1 m for every 3 m."""
    x = np.arange(-30.0, 31.0)
    geometry = hydraulic_geometry(make_section(BED + np.where(x < 0, -x, x / 3)), depth=2.0)

    assert geometry.area == pytest.approx((1 + 3) * 2.0**2 / 2, rel=1e-12)
    assert geometry.top_width == pytest.approx((1 + 3) * 2.0, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(2.0 * (math.hypot(1, 1) + math.hypot(1, 3)), rel=1e-12)


def test_depth_is_measured_from_the_stream_cell() -> None:
    xs = trapezoid(10.0, 2.0)

    assert hydraulic_geometry(xs, depth=1.5) == hydraulic_geometry(xs, wse=101.5)
    assert discharge(xs, SLOPE, depth=1.5) == discharge(xs, SLOPE, wse=101.5)


@pytest.mark.parametrize("depth", [0.0, -0.5])
def test_dry_when_the_water_surface_is_not_above_the_stream_cell(depth: float) -> None:
    xs = trapezoid(10.0, 2.0)

    assert hydraulic_geometry(xs, depth=depth) == DRY
    assert discharge(xs, SLOPE, depth=depth) == 0.0


def test_water_spreads_out_from_the_stream_cell() -> None:
    """The stream cell sits on a bump 0.5 m above the ground either side of it, which only floods once the
    water reaches the stream cell."""
    distance = np.abs(np.arange(-10.0, 11.0))
    elevations = BED + 0.5 * distance
    elevations[10] = 101.0
    xs = make_section(elevations)

    assert hydraulic_geometry(xs, wse=100.8) == DRY
    # At 101.5 m, each side has two full segments (depths 0.5 to 1.0, then 1.0 to 0.5) and one that the
    # water just reaches the end of (0.5 to 0)
    geometry = hydraulic_geometry(xs, wse=101.5)
    assert geometry.area == pytest.approx(2 * (0.75 + 0.75 + 0.25), rel=1e-12)
    assert geometry.top_width == pytest.approx(6.0, rel=1e-12)
    # A depth is measured from the stream cell, not from the lowest ground
    assert hydraulic_geometry(xs, depth=0.5) == geometry


def test_low_ground_behind_a_ridge_stays_dry_until_the_ridge_is_overtopped() -> None:
    xs = ridge_and_hollow()

    # Below the ridge the section is just the V channel, 2.5 m deep
    assert hydraulic_geometry(xs, wse=102.5).area == pytest.approx(2.5**2, rel=1e-12)
    # Above it, the hollow (bed 99 m) fills too. Left: 3.5**2 / 2. Right: 6 in the channel up to the ridge,
    # then 2.5, 4.5 and 4.5 over the hollow, and 2.025 where the water meets the far side (0.9 m across)
    assert hydraulic_geometry(xs, wse=103.5).area == pytest.approx(3.5**2 / 2 + 6 + 2.5 + 4.5 + 4.5 + 2.025, rel=1e-12)


def test_the_ends_of_the_cross_section_act_as_walls() -> None:
    """Water above all of a 20 m wide flat section is held in by vertical walls at its ends."""
    geometry = hydraulic_geometry(make_section(np.full(21, BED)), depth=2.0)

    assert geometry.area == pytest.approx(40.0, rel=1e-12)
    assert geometry.top_width == pytest.approx(20.0, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(20.0 + 2 * 2.0, rel=1e-12)


def test_off_raster_walls_hold_the_water_in() -> None:
    """A 200 m section sampled from an 11 x 11 raster of 10 m cells runs 50 m off each edge, into 9999 m walls."""
    dem = np.full((11, 11), BED)
    xs = sample_cross_section(dem, np.full_like(dem, N), 5, 5, np.pi / 2, 200.0, 10.0, 10.0)

    geometry = hydraulic_geometry(xs, depth=1.0)

    assert geometry.top_width == pytest.approx(100.0, abs=0.01)
    assert geometry.area == pytest.approx(100.0, abs=0.01)
    assert geometry.wetted_perimeter == pytest.approx(100.0 + 2 * 1.0, abs=0.01)
    assert geometry.mannings_n == pytest.approx(N, rel=1e-12)  # the walls' 9999 roughness doesn't count


@pytest.mark.parametrize("shape", SHAPES)
def test_top_widths_are_each_side_s_share_of_the_top_width(shape: str) -> None:
    """The distances from the stream cell to the water's edges, which add up to the top width."""
    xs = SHAPES[shape]()
    ground = xs.elevations[xs.elevations < 9000]
    center = xs.elevations.size // 2

    for wse in np.concatenate([np.linspace(ground.min() - 1, ground.max() + 3, 200), ground]):
        left, right = hydraulics.top_widths(xs.elevations, xs.ordinate_distance, wse)
        if not wse > xs.elevations[center]:
            assert (left, right) == (0.0, 0.0)
            continue
        assert left == pytest.approx(hydraulics._side_geometry(
            xs.elevations, xs.mannings_n, center, -1, xs.ordinate_distance, wse)[2], rel=1e-12)
        assert right == pytest.approx(hydraulics._side_geometry(
            xs.elevations, xs.mannings_n, center, 1, xs.ordinate_distance, wse)[2], rel=1e-12)
        assert left + right == pytest.approx(hydraulic_geometry(xs, wse=wse).top_width, rel=1e-12)


def test_mirroring_the_section_changes_nothing() -> None:
    xs = rough_valley()
    mirrored = XSection(xs.elevations[::-1].copy(), xs.mannings_n[::-1].copy(), xs.ordinate_distance)

    for wse in np.linspace(100.0, 130.0, 61):
        assert hydraulic_geometry(mirrored, wse=wse) == pytest.approx(hydraulic_geometry(xs, wse=wse), rel=1e-12)


def test_float32_sections_give_the_same_answers() -> None:
    xs = trapezoid(10.0, 2.0)
    xs32 = XSection(xs.elevations.astype(np.float32), xs.mannings_n.astype(np.float32), xs.ordinate_distance)

    assert hydraulic_geometry(xs32, depth=1.3) == pytest.approx(hydraulic_geometry(xs, depth=1.3), rel=1e-6)
    assert discharge(xs32, SLOPE, depth=1.3) == pytest.approx(discharge(xs, SLOPE, depth=1.3), rel=1e-6)


def test_a_single_ordinate_section_never_carries_water() -> None:
    xs = make_section([BED])

    assert hydraulic_geometry(xs, depth=1.0).area == 0.0
    assert math.isnan(wse_for_discharge(xs, 1.0, SLOPE))


def test_needs_exactly_one_of_wse_and_depth() -> None:
    xs = trapezoid(10.0, 2.0)

    with pytest.raises(TypeError, match="exactly one of wse or depth"):
        hydraulic_geometry(xs)
    with pytest.raises(TypeError, match="exactly one of wse or depth"):
        discharge(xs, SLOPE, wse=101.0, depth=1.0)


@pytest.mark.parametrize("slope", [0.0, -0.001, math.nan])
def test_slope_must_be_positive(slope: float) -> None:
    with pytest.raises(ValueError, match="slope must be positive"):
        discharge(trapezoid(10.0, 2.0), slope, depth=1.0)


# --- Roughness and discharge -------------------------------------------------------------------------------


def test_a_uniform_bed_s_roughness_is_the_composite_roughness() -> None:
    xs = trapezoid(10.0, 2.0)
    xs.mannings_n[:] = 0.042

    assert hydraulic_geometry(xs, depth=1.7).mannings_n == pytest.approx(0.042, rel=1e-12)


def test_composite_roughness_weights_each_segment_by_its_wetted_length() -> None:
    """A V with 45 degree banks, with n = 0.02 up to and including the stream cell and 0.05 to its right.

    Each segment takes the n of its end nearer the stream cell. So at 2 m deep, the two left segments and
    the first right segment are 0.02, the second right segment is 0.05, and all four are sqrt(2) m long.
    """
    xs = trapezoid(0.0, 1.0, ordinates_per_side=10)
    xs.mannings_n[:] = np.where(np.arange(21) <= 10, 0.02, 0.05)

    expected = ((3 * 0.02**1.5 + 0.05**1.5) / 4) ** (2 / 3)
    assert hydraulic_geometry(xs, depth=2.0).mannings_n == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("depth", [0.5, 1.3, 3.0])
def test_discharge_matches_mannings_equation(depth: float) -> None:
    area = (10 + 2 * depth) * depth
    perimeter = 10 + 2 * depth * math.hypot(1, 2)
    expected = area * (area / perimeter) ** (2 / 3) * math.sqrt(SLOPE) / N

    assert discharge(trapezoid(10.0, 2.0), SLOPE, depth=depth) == pytest.approx(expected, rel=1e-12)


def test_discharge_scales_with_the_square_root_of_slope() -> None:
    xs = trapezoid(10.0, 2.0)

    assert discharge(xs, 4 * SLOPE, depth=1.0) == pytest.approx(2 * discharge(xs, SLOPE, depth=1.0), rel=1e-12)


def test_discharge_rises_with_the_water_level_between_straight_banks() -> None:
    """Including once the water is above the banks and held in by the walls at the ends (127.5 m).

    With uneven banks and roughness this needn't hold: in rough_valley, discharge falls slightly as the water
    spreads over flatter or rougher ground, as it does onto flat_floodplain.
    """
    flows = [discharge(trapezoid(10.0, 2.0), SLOPE, wse=wse) for wse in np.linspace(BED, 160.0, 2001)]

    assert np.all(np.diff(flows) > 0)


def test_matches_the_legacy_calculation_for_a_uniform_bed() -> None:
    """ARC's current calculation, with its depth-varying roughness switched off, on a bumpy valley that the
    water never fills to either end (where the legacy code drops the first segment)."""
    rng = np.random.default_rng(7)
    x = 2.0 * np.arange(-60, 61)
    elevations = BED + 0.004 * x**2 + rng.normal(0.0, 0.15, x.size)
    elevations[60] = BED
    xs = make_section(elevations, spacing=2.0)
    right, left = xs.elevations[60:].copy(), xs.elevations[60::-1].copy()
    n_right, n_left = xs.mannings_n[60:].copy(), xs.mannings_n[60::-1].copy()

    for wse in [100.3, 101.0, 102.5, 104.0]:
        a1, p1, _, t1 = _calculate_stream_geometry_and_topwidth(right, wse, 2.0, n_right, 6.0, 1.0, 1.0)
        a2, p2, _, t2 = _calculate_stream_geometry_and_topwidth(left, wse, 2.0, n_left, 6.0, 1.0, 1.0)
        legacy_q = calculate_discharge_from_wse(wse, math.sqrt(SLOPE), right, right.size, n_right,
                                                left, left.size, n_left, 2.0, 6.0, 1.0, 1.0)

        geometry = hydraulic_geometry(xs, wse=wse)
        assert geometry.area == pytest.approx(a1 + a2, rel=1e-12)
        assert geometry.wetted_perimeter == pytest.approx(p1 + p2, rel=1e-12)
        assert geometry.top_width == pytest.approx(t1 + t2, rel=1e-12)
        assert discharge(xs, SLOPE, wse=wse) == pytest.approx(legacy_q, rel=1e-12)


# --- Conveyance table and water surface elevation for a discharge ------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_table_matches_the_direct_calculation(shape: str) -> None:
    xs = SHAPES[shape]()
    table = ConveyanceTable(xs)
    ground = xs.elevations[xs.elevations < 9000]

    # Include every ordinate's elevation, where the wet ordinates change
    for wse in np.concatenate([np.linspace(ground.min() - 1, ground.max() + 3, 400), ground]):
        assert table.geometry(wse=wse) == pytest.approx(hydraulic_geometry(xs, wse=wse), rel=1e-9, abs=1e-9)
        assert table.discharge(SLOPE, wse=wse) == pytest.approx(discharge(xs, SLOPE, wse=wse), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("shape", ["v-channel", "trapezoid", "flat-floodplain", "walled-bed", "rough-valley"])
def test_wse_for_discharge_carries_that_discharge(shape: str) -> None:
    xs = SHAPES[shape]()
    table = ConveyanceTable(xs)

    for q in np.geomspace(0.01, 5000.0, 60):
        wse = wse_for_discharge(xs, q, SLOPE)
        assert table.wse_for_discharge(q, SLOPE) == wse
        assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-8)


def test_wse_for_discharge_agrees_with_bisection() -> None:
    """On a section whose discharge always rises with the water level, so that bisection has one answer."""
    xs = trapezoid(10.0, 2.0)

    for q in [0.5, 5.0, 50.0, 500.0]:
        bisected = _bisect_wse(xs.elevations, xs.mannings_n, xs.ordinate_distance, q / math.sqrt(SLOPE), 60)
        assert wse_for_discharge(xs, q, SLOPE) == pytest.approx(bisected, abs=1e-9)


def test_lowest_wse_is_returned_when_discharge_falls_as_water_spreads() -> None:
    """Water spilling from the channel onto a flat floodplain carries less than the full channel did, so
    discharge falls and then rises again, and two water levels carry the same discharge."""
    xs = flat_floodplain()
    bankfull = discharge(xs, SLOPE, wse=102.0)
    q = 0.9 * bankfull
    assert discharge(xs, SLOPE, wse=102.01) < q

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse < 102.0
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)
    assert max(discharge(xs, SLOPE, wse=level) for level in np.linspace(BED, wse, 2000, endpoint=False)) < q


def test_wse_sits_on_the_ridge_when_overtopping_it_jumps_past_the_discharge() -> None:
    xs = ridge_and_hollow()
    below = discharge(xs, SLOPE, wse=103.0)  # the hollow is still dry
    above = discharge(xs, SLOPE, wse=103.0 + 1e-9)  # the hollow has filled
    assert above > 2 * below

    assert wse_for_discharge(xs, (below + above) / 2, SLOPE) == 103.0


def test_discharge_above_the_highest_ground_is_held_between_the_walls() -> None:
    xs = trapezoid(10.0, 2.0, ordinates_per_side=20)  # the banks top out at 107.5 m
    q = 10 * discharge(xs, SLOPE, wse=107.5)

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse > 107.5
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)


def test_no_discharge_is_no_depth() -> None:
    xs = trapezoid(10.0, 2.0)

    assert wse_for_discharge(xs, 0.0, SLOPE) == BED
    assert ConveyanceTable(xs).wse_for_discharge(0.0, SLOPE) == BED


def test_one_table_serves_any_slope() -> None:
    table = ConveyanceTable(trapezoid(10.0, 2.0))
    wse = table.wse_for_discharge(50.0, SLOPE)

    # Quadrupling the slope doubles the discharge at the same water level
    assert table.wse_for_discharge(100.0, 4 * SLOPE) == pytest.approx(wse, abs=1e-9)
    assert table.discharge(4 * SLOPE, wse=wse) == pytest.approx(100.0, rel=1e-9)


def test_a_table_limited_in_depth_covers_that_depth() -> None:
    xs = rough_valley()
    table = ConveyanceTable(xs, max_depth=5.0)
    top = float(table.table[-1, 1])  # the end of its last interval
    assert top >= BED + 5.0

    assert table.discharge(SLOPE, depth=4.0) == pytest.approx(discharge(xs, SLOPE, depth=4.0), rel=1e-9)
    assert math.isnan(table.discharge(SLOPE, wse=top + 0.1))
    assert math.isnan(table.wse_for_discharge(discharge(xs, SLOPE, wse=top + 0.1), SLOPE))


# --- Cross sections divided at the banks -------------------------------------------------------------------


def test_the_banks_divide_the_water_into_the_channel_and_the_overbanks() -> None:
    """With the water 1 m over the floodplains, the channel holds its full 2 m deep trapezoid and 1 m of water
    across its 24 m top, and each overbank holds 1 m of water over 18 m of floodplain, against a 1 m wall."""
    parts = compound_geometry(channel_and_floodplains(), wse=103.0)

    channel_perimeter = 20 + 2 * math.hypot(2, 2)  # the lines at the banks add no wetted perimeter
    assert parts.channel == pytest.approx(HydraulicGeometry(68.0, channel_perimeter, 68.0 / channel_perimeter, 24.0, N),
                                          rel=1e-12)
    overbank = HydraulicGeometry(18.0, 18.0 + 1.0, 18.0 / 19.0, 18.0, N)
    assert parts.left_overbank == pytest.approx(overbank, rel=1e-12)
    assert parts.right_overbank == pytest.approx(overbank, rel=1e-12)


def test_discharge_with_banks_adds_up_the_subsections() -> None:
    xs = channel_and_floodplains()
    expected = (manning_conveyance(68.0, 20 + 2 * math.hypot(2, 2)) + 2 * manning_conveyance(18.0, 19.0)) * math.sqrt(SLOPE)

    assert discharge(xs, SLOPE, wse=103.0) == pytest.approx(expected, rel=1e-12)
    # Treating it all as one section, the floodplains' long wetted perimeter holds back the channel's water too
    whole = manning_conveyance(68.0 + 36.0, 20 + 2 * math.hypot(2, 2) + 38.0) * math.sqrt(SLOPE)
    assert expected > 1.1 * whole


def test_the_banks_change_the_discharge_but_not_the_geometry() -> None:
    xs = channel_and_floodplains()
    parts = compound_geometry(xs, wse=103.0)
    whole = hydraulic_geometry(xs, wse=103.0)

    assert sum(part.area for part in parts) == pytest.approx(whole.area, rel=1e-12)
    assert sum(part.wetted_perimeter for part in parts) == pytest.approx(whole.wetted_perimeter, rel=1e-12)
    assert sum(part.top_width for part in parts) == pytest.approx(whole.top_width, rel=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("banks", BANKS)
def test_the_subsections_add_up_to_the_whole_section(shape: str, banks: tuple[float, float]) -> None:
    xs = with_banks(SHAPES[shape](), *banks)
    ground = xs.elevations[xs.elevations < 9000]

    for wse in np.concatenate([np.linspace(ground.min(), ground.max() + 2, 80), ground]):
        parts = hydraulics.compound_wetted_geometry(xs.elevations, xs.mannings_n, xs.ordinate_distance, *banks, wse)
        whole = hydraulics.wetted_geometry(xs.elevations, xs.mannings_n, xs.ordinate_distance, wse)
        assert np.sum(parts, axis=0) == pytest.approx(whole, rel=1e-12, abs=1e-12)
        assert np.min(parts) >= 0.0


def test_without_banks_the_section_is_all_channel() -> None:
    xs = rough_valley()  # XSection's banks start at -1, not yet found
    assert xs.left_bank_distance < 0 and xs.right_bank_distance < 0

    for wse in [100.5, 104.0, 112.0]:
        parts = compound_geometry(xs, wse=wse)
        assert parts.channel == hydraulic_geometry(xs, wse=wse)
        assert parts.left_overbank == parts.right_overbank == DRY


@pytest.mark.parametrize("banks", [(-1.0, -1.0), (math.nan, math.nan), (-1.0, math.nan), (160.0, 1e9), (math.inf, 500.0)])
def test_banks_that_divide_nothing_leave_one_section(banks: tuple[float, float]) -> None:
    """Undefined banks (negative or NaN), and banks at or beyond the ends of the section (160 m here)."""
    xs = rough_valley()

    for wse in np.linspace(100.0, 160.0, 61):
        k = hydraulics.compound_conveyance(xs.elevations, xs.mannings_n, xs.ordinate_distance, *banks, wse)
        assert k == pytest.approx(hydraulics.conveyance(xs.elevations, xs.mannings_n, xs.ordinate_distance, wse),
                                  rel=1e-12)


def test_water_inside_the_banks_is_all_channel() -> None:
    """At 101.5 m the water's edges are 11.5 m out, inside the banks at 12 m."""
    xs = channel_and_floodplains()

    parts = compound_geometry(xs, wse=101.5)

    assert parts.left_overbank == parts.right_overbank == DRY
    assert parts.channel == pytest.approx(hydraulic_geometry(xs, wse=101.5), rel=1e-12)
    assert discharge(xs, SLOPE, wse=101.5) == pytest.approx(
        hydraulics.conveyance(xs.elevations, xs.mannings_n, 1.0, 101.5) * math.sqrt(SLOPE), rel=1e-12)


@pytest.mark.parametrize("banks", [(7.0, 13.0), (7.3, 12.9), (0.4, 0.0)])
def test_a_bank_between_ordinates_divides_its_segment(banks: tuple[float, float]) -> None:
    """The same ground and roughness sampled more finely, so that the banks can fall on or off ordinates, gives
    the same subsections. At 2 m spacing, banks 7 m and 13 m out are half way along a segment. At 1 m they are on
    ordinates."""
    xs = with_banks(rough_valley(), *banks)

    for factor in [2, 3]:
        fine = with_banks(refine(xs, factor), *banks)
        for wse in np.linspace(100.0, 135.0, 141):
            assert np.array(compound_geometry(fine, wse=wse)) == pytest.approx(
                np.array(compound_geometry(xs, wse=wse)), rel=1e-11, abs=1e-11)


def test_a_bank_at_the_stream_cell_puts_that_whole_side_in_its_overbank() -> None:
    xs = with_banks(trapezoid(10.0, 2.0), 0.0, 1e6)
    parts = compound_geometry(xs, wse=102.0)

    # The trapezoid is symmetric, so each side holds half the water
    half = hydraulic_geometry(trapezoid(10.0, 2.0), wse=102.0).area / 2
    assert parts.left_overbank.area == pytest.approx(half, rel=1e-12)
    assert parts.channel.area == pytest.approx(half, rel=1e-12)
    assert parts.right_overbank == DRY

    # With both banks there, the channel has no width, and the sides are separate subsections
    both = compound_geometry(with_banks(trapezoid(10.0, 2.0), 0.0, 0.0), wse=102.0)
    assert both.channel == DRY
    assert both.left_overbank == pytest.approx(both.right_overbank, rel=1e-12)


def test_the_walls_are_in_the_overbanks_or_in_the_channel_without_them() -> None:
    xs = channel_and_floodplains()
    wse = 104.0  # 2 m over the floodplains, against the walls at 30 m

    assert compound_geometry(xs, wse=wse).right_overbank.wetted_perimeter == pytest.approx(18.0 + 2.0, rel=1e-12)
    no_right_overbank = compound_geometry(with_banks(xs, 12.0, 30.0), wse=wse)  # the bank is on the wall
    assert no_right_overbank.right_overbank == DRY
    assert no_right_overbank.channel.wetted_perimeter == pytest.approx(
        10.0 + math.hypot(2, 2) + 18.0 + 2.0 + 10.0 + math.hypot(2, 2), rel=1e-12)


def test_off_raster_walls_bound_the_overbanks() -> None:
    """The 9999 m walls from sampling are in the overbanks, and their 9999 roughness still doesn't count."""
    dem = np.full((11, 11), BED)
    xs = with_banks(sample_cross_section(dem, np.full_like(dem, N), 5, 5, np.pi / 2, 200.0, 10.0, 10.0), 20.0, 20.0)

    parts = compound_geometry(xs, depth=1.0)

    assert parts.channel.top_width == pytest.approx(40.0, abs=0.01)
    assert parts.left_overbank.top_width == pytest.approx(30.0, abs=0.01)
    for part in parts:
        assert part.mannings_n == pytest.approx(N, rel=1e-12)


def test_mirroring_swaps_the_overbanks() -> None:
    xs = with_banks(rough_valley(), 7.3, 21.0)
    mirrored = with_banks(XSection(xs.elevations[::-1].copy(), xs.mannings_n[::-1].copy(), xs.ordinate_distance),
                          21.0, 7.3)

    for wse in np.linspace(100.0, 130.0, 61):
        parts = compound_geometry(xs, wse=wse)
        mirror = compound_geometry(mirrored, wse=wse)
        assert mirror.left_overbank == pytest.approx(parts.right_overbank, rel=1e-12, abs=1e-12)
        assert mirror.channel == pytest.approx(parts.channel, rel=1e-12, abs=1e-12)
        assert discharge(mirrored, SLOPE, wse=wse) == pytest.approx(discharge(xs, SLOPE, wse=wse), rel=1e-12)


def test_matches_the_legacy_compound_calculation_for_a_uniform_bed() -> None:
    """ARC's current divided-channel calculation, with banks on ordinates and its depth-varying roughness switched
    off, on a bumpy valley that the water never fills to either end."""
    rng = np.random.default_rng(7)
    x = 2.0 * np.arange(-60, 61)
    elevations = BED + 0.004 * x**2 + rng.normal(0.0, 0.15, x.size)
    elevations[60] = BED
    right, left = elevations[60:].copy(), elevations[60::-1].copy()
    n_right, n_left = np.full(61, N), np.full(61, N)

    for left_bank, right_bank in [(1, 1), (4, 9), (12, 3)]:
        xs = with_banks(make_section(elevations, spacing=2.0), 2.0 * left_bank, 2.0 * right_bank)
        for wse in [100.3, 101.0, 102.5, 104.0, 108.0]:
            area, perimeter, width, legacy_k = _compound_section_conveyance(
                left, n_left, left_bank, right, n_right, right_bank, wse, 2.0, 6.0, 1.0, 1.0)
            whole = hydraulic_geometry(xs, wse=wse)
            assert whole.area == pytest.approx(area, rel=1e-12)
            assert whole.wetted_perimeter == pytest.approx(perimeter, rel=1e-12)
            assert whole.top_width == pytest.approx(width, rel=1e-12)
            assert discharge(xs, SLOPE, wse=wse) == pytest.approx(legacy_k * math.sqrt(SLOPE), rel=1e-12)


def test_banks_at_the_channel_edge_remove_the_dip_onto_a_flat_floodplain() -> None:
    """As one section, discharge falls as the channel spills onto the floodplain. Divided at the channel's edges,
    the floodplain's conveyance starts from nothing and the channel's keeps rising."""
    whole = flat_floodplain()
    divided = with_banks(flat_floodplain(), 2.0, 2.0)
    levels = np.linspace(BED, 110.0, 2001)

    assert discharge(whole, SLOPE, wse=102.01) < discharge(whole, SLOPE, wse=102.0)
    flows = [discharge(divided, SLOPE, wse=wse) for wse in levels]
    assert np.all(np.diff(flows) > 0)


def test_discharge_is_continuous_as_the_water_edge_passes_a_bank() -> None:
    """The bank 7.3 m out is part way up a rising segment, whose ground there is at bank_ground."""
    xs = with_banks(rough_valley(), 30.0, 7.3)
    bank_ground = np.interp(7.3 / 2.0, np.arange(81), xs.elevations[80:])
    assert hydraulic_geometry(xs, wse=bank_ground).top_width > 7.3  # the water gets there across the channel

    below, above = discharge(xs, SLOPE, wse=bank_ground - 1e-7), discharge(xs, SLOPE, wse=bank_ground + 1e-7)
    assert above == pytest.approx(below, rel=1e-6)


def test_conveyance_without_powers_matches_the_formula() -> None:
    """Conveyance A**(5/3) / W**(2/3) is found as A**2 * (A * W**2)**(-1/3), with no power or division."""
    rng = np.random.default_rng(3)
    areas = np.exp(rng.uniform(math.log(1e-8), math.log(1e8), 20000))
    weighted = np.exp(rng.uniform(math.log(1e-6), math.log(1e4), 20000))

    found = np.array([hydraulics._conveyance(a, w) for a, w in zip(areas, weighted)])

    assert found == pytest.approx(areas ** (5 / 3) / weighted ** (2 / 3), rel=1e-14)
    assert hydraulics._conveyance(0.0, 1.0) == 0.0
    assert hydraulics._conveyance(1.0, 0.0) == math.inf  # a wetted perimeter with no roughness
    assert math.isnan(hydraulics._conveyance(math.nan, 1.0))
    assert hydraulics._conveyance(1e120, 1.0) == pytest.approx(1e200, rel=1e-14)  # beyond the fast method's range


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("banks", BANKS)
def test_compound_table_matches_the_direct_calculation(shape: str, banks: tuple[float, float]) -> None:
    xs = with_banks(SHAPES[shape](), *banks)
    table = ConveyanceTable(xs)
    assert table.compound
    ground = xs.elevations[xs.elevations < 9000]
    center = xs.elevations.size // 2
    # The ground under each bank, where the water's edge passes from the channel into an overbank
    bank_ground = [np.interp(banks[0] / xs.ordinate_distance, np.arange(center + 1), xs.elevations[center::-1]),
                   np.interp(banks[1] / xs.ordinate_distance, np.arange(center + 1), xs.elevations[center:])]

    # Compare area, wetted perimeter, top width and sum(P_i * n_i**1.5) rather than hydraulic radius and roughness,
    # which are ratios of two tiny numbers for a subsection that the water only just reaches
    for wse in np.concatenate([np.linspace(ground.min() - 1, ground.max() + 3, 300), ground, bank_ground]):
        assert np.array(hydraulics.compound_table_geometry(table.table, wse)) == pytest.approx(
            np.array(hydraulics.compound_wetted_geometry(xs.elevations, xs.mannings_n, xs.ordinate_distance, *banks,
                                                         wse)), rel=1e-9, abs=1e-9)
        assert table.discharge(SLOPE, wse=wse) == pytest.approx(discharge(xs, SLOPE, wse=wse), rel=1e-9, abs=1e-12)
    for depth in [0.7, 2.3, 6.1]:
        assert np.array(table.compound_geometry(depth=depth)) == pytest.approx(
            np.array(compound_geometry(xs, depth=depth)), rel=1e-9, abs=1e-9)
        assert table.geometry(depth=depth) == pytest.approx(hydraulic_geometry(xs, depth=depth), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("shape", ["v-channel", "trapezoid", "flat-floodplain", "walled-bed", "rough-valley"])
@pytest.mark.parametrize("banks", BANKS)
def test_compound_wse_for_discharge_carries_that_discharge(shape: str, banks: tuple[float, float]) -> None:
    xs = with_banks(SHAPES[shape](), *banks)
    table = ConveyanceTable(xs)

    for q in np.geomspace(0.01, 5000.0, 40):
        wse = wse_for_discharge(xs, q, SLOPE)
        assert table.wse_for_discharge(q, SLOPE) == wse
        assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-8)


def test_compound_wse_agrees_with_bisection() -> None:
    """On a section whose divided discharge always rises with the water level, so that bisection has one answer."""
    xs = with_banks(flat_floodplain(), 2.0, 2.0)

    for q in [0.5, 5.0, 50.0, 500.0]:
        bisected = _bisect_compound_wse(xs.elevations, xs.mannings_n, 1.0, 2.0, q / math.sqrt(SLOPE), 60)
        assert wse_for_discharge(xs, q, SLOPE) == pytest.approx(bisected, abs=1e-9)


def test_compound_wse_is_the_lowest_when_an_overbank_s_discharge_falls() -> None:
    """With banks inside the channel, the overbanks hold the channel's upper banks, and spill onto the flat
    floodplain at 102 m, where their discharge falls and so does the total."""
    xs = with_banks(flat_floodplain(), 1.0, 1.0)
    q = 0.97 * discharge(xs, SLOPE, wse=102.0)
    assert discharge(xs, SLOPE, wse=102.01) < q

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse < 102.0
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)
    assert max(discharge(xs, SLOPE, wse=level) for level in np.linspace(BED, wse, 2000, endpoint=False)) < q


def test_compound_wse_sits_on_the_ridge_when_the_overbank_behind_it_floods() -> None:
    xs = with_banks(ridge_and_hollow(), 3.0, 3.0)  # the right bank is on the ridge, with the hollow beyond
    below = discharge(xs, SLOPE, wse=103.0)
    above = discharge(xs, SLOPE, wse=103.0 + 1e-9)
    assert compound_geometry(xs, wse=103.0 + 1e-9).right_overbank.area > 10.0
    assert above > 1.5 * below

    assert wse_for_discharge(xs, (below + above) / 2, SLOPE) == 103.0
    assert ConveyanceTable(xs).wse_for_discharge((below + above) / 2, SLOPE) == 103.0


def test_compound_conveyance_is_convex_between_ordinate_elevations() -> None:
    """The exact solve relies on this (see the notes above the compound table columns): sampled across each
    interval of a table, conveyance never bends downward."""
    for shape, banks in [("rough-valley", (7.3, 12.9)), ("flat-floodplain", (1.0, 3.5)), ("trapezoid", (0.0, 6.0))]:
        xs = with_banks(SHAPES[shape](), *banks)
        table = hydraulics.build_compound_conveyance_table(xs.elevations, xs.mannings_n, xs.ordinate_distance,
                                                           *banks, 125.0)
        for row in table:
            end = min(row[1], row[0] + 10.0)  # the last interval goes on forever
            if not end - row[0] > 1e-6:
                continue
            levels = np.linspace(row[0], end, 23)[1:]
            k = np.array([hydraulics.compound_table_conveyance(table, level) for level in levels])
            assert np.all(k[:-2] - 2 * k[1:-1] + k[2:] >= -1e-9 * k[2:])


def test_compound_discharge_above_the_highest_ground_is_held_between_the_walls() -> None:
    xs = with_banks(trapezoid(10.0, 2.0, ordinates_per_side=20), 5.0, 5.0)  # the banks top out at 107.5 m
    q = 10 * discharge(xs, SLOPE, wse=107.5)

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse > 107.5
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)


def test_compound_edge_cases() -> None:
    single = with_banks(make_section([BED]), 0.0, 0.0)
    assert compound_geometry(single, depth=1.0).channel.area == 0.0
    assert math.isnan(wse_for_discharge(single, 1.0, SLOPE))

    xs = with_banks(trapezoid(10.0, 2.0), 3.0, 4.5)
    assert wse_for_discharge(xs, 0.0, SLOPE) == BED
    assert ConveyanceTable(xs).wse_for_discharge(0.0, SLOPE) == BED
    assert compound_geometry(xs, depth=0.0) == CompoundGeometry(DRY, DRY, DRY)

    xs32 = with_banks(XSection(xs.elevations.astype(np.float32), xs.mannings_n.astype(np.float32), 1.0), 3.0, 4.5)
    assert discharge(xs32, SLOPE, depth=1.3) == pytest.approx(discharge(xs, SLOPE, depth=1.3), rel=1e-6)
    assert wse_for_discharge(xs32, 20.0, SLOPE) == pytest.approx(wse_for_discharge(xs, 20.0, SLOPE), abs=1e-5)


def test_a_compound_table_limited_in_depth_covers_that_depth() -> None:
    xs = with_banks(rough_valley(), 7.3, 12.9)
    table = ConveyanceTable(xs, max_depth=5.0)
    top = float(table.table[-1, 1])
    assert top >= BED + 5.0

    assert table.discharge(SLOPE, depth=4.0) == pytest.approx(discharge(xs, SLOPE, depth=4.0), rel=1e-9)
    assert math.isnan(table.discharge(SLOPE, wse=top + 0.1))
    assert all(math.isnan(part.area) for part in table.compound_geometry(wse=top + 0.1))
    assert math.isnan(table.wse_for_discharge(discharge(xs, SLOPE, wse=top + 0.1), SLOPE))


def test_an_undivided_table_has_all_its_water_in_the_channel() -> None:
    xs = rough_valley()
    table = ConveyanceTable(xs)
    assert not table.compound

    parts = table.compound_geometry(wse=110.0)
    assert parts.channel == pytest.approx(hydraulic_geometry(xs, wse=110.0), rel=1e-9)
    assert parts.left_overbank == parts.right_overbank == DRY


# --- Speed (each compiled loop runs once first, so compilation isn't counted) ------------------------------

# These helpers are compiled but not cached: a cached function isn't recompiled when a function it calls
# in another module changes, so a cache would keep testing old versions of the hydraulics functions.


@njit
def _bisect_wse(elevations, mannings_n, spacing, target, evaluations):
    lo = elevations[elevations.size // 2]
    hi = lo + 25.0
    for _ in range(evaluations):
        mid = 0.5 * (lo + hi)
        if hydraulics.conveyance(elevations, mannings_n, spacing, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


@njit
def _repeat_conveyance(elevations, mannings_n, spacing, wse, calls):
    total = 0.0
    for i in range(calls):
        total += hydraulics.conveyance(elevations, mannings_n, spacing, wse + 1e-9 * (i % 7))
    return total


@njit
def _repeat_legacy_discharge(right, n_right, left, n_left, spacing, wse, calls):
    total = 0.0
    for i in range(calls):
        total += calculate_discharge_from_wse(wse + 1e-9 * (i % 7), 1.0, right, right.size, n_right,
                                              left, left.size, n_left, spacing, 6.0, 1.0, 1.0)
    return total


@njit
def _repeat_bisection(elevations, mannings_n, spacing, target, calls):
    total = 0.0
    for i in range(calls):
        total += _bisect_wse(elevations, mannings_n, spacing, target * (1.0 + 1e-9 * (i % 7)), 25)
    return total


@njit
def _repeat_wse_for_conveyance(elevations, mannings_n, spacing, target, calls):
    total = 0.0
    for i in range(calls):
        total += hydraulics.wse_for_conveyance(elevations, mannings_n, spacing, target * (1.0 + 1e-9 * (i % 7)))
    return total


@njit
def _repeat_table_wse_for_conveyance(table, target, calls):
    total = 0.0
    for i in range(calls):
        total += hydraulics.table_wse_for_conveyance(table, target * (1.0 + 1e-9 * (i % 7)))
    return total


@njit
def _bisect_compound_wse(elevations, mannings_n, spacing, bank, target, evaluations):
    lo = elevations[elevations.size // 2]
    hi = lo + 25.0
    for _ in range(evaluations):
        mid = 0.5 * (lo + hi)
        if hydraulics.compound_conveyance(elevations, mannings_n, spacing, bank, bank, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


@njit
def _repeat_compound_conveyance(elevations, mannings_n, spacing, bank, wse, calls):
    total = 0.0
    for i in range(calls):
        total += hydraulics.compound_conveyance(elevations, mannings_n, spacing, bank, bank, wse + 1e-9 * (i % 7))
    return total


@njit
def _repeat_legacy_compound_discharge(right, n_right, left, n_left, spacing, bank_index, wse, calls):
    total = 0.0
    for i in range(calls):
        total += calculate_discharge_from_wse(wse + 1e-9 * (i % 7), 1.0, right, right.size, n_right, left, left.size,
                                              n_left, spacing, 6.0, 1.0, 1.0, 1.0, bank_index, bank_index)
    return total


@njit
def _repeat_compound_bisection(elevations, mannings_n, spacing, bank, target, calls):
    total = 0.0
    for i in range(calls):
        total += _bisect_compound_wse(elevations, mannings_n, spacing, bank, target * (1.0 + 1e-9 * (i % 7)), 25)
    return total


@njit
def _repeat_wse_for_compound_conveyance(elevations, mannings_n, spacing, bank, target, calls):
    total = 0.0
    for i in range(calls):
        total += hydraulics.wse_for_compound_conveyance(elevations, mannings_n, spacing, bank, bank,
                                                        target * (1.0 + 1e-9 * (i % 7)))
    return total


@njit
def _repeat_compound_table_wse_for_conveyance(table, target, calls):
    total = 0.0
    for i in range(calls):
        total += hydraulics.compound_table_wse_for_conveyance(table, target * (1.0 + 1e-9 * (i % 7)))
    return total


def _seconds_per_call(repeat, *args, calls: int) -> float:
    """Best of five timed runs of a compiled loop, after one untimed call to compile it."""
    repeat(*args, 1)
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        repeat(*args, calls)
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def test_conveyance_is_much_faster_than_the_legacy_calculation() -> None:
    xs = typical_section()
    right, left = xs.elevations[250:].copy(), xs.elevations[250::-1].copy()
    n_right, n_left = xs.mannings_n[250:].copy(), xs.mannings_n[250::-1].copy()

    new = _seconds_per_call(_repeat_conveyance, xs.elevations, xs.mannings_n, 10.0, 104.0, calls=20000)
    legacy = _seconds_per_call(_repeat_legacy_discharge, right, n_right, left, n_left, 10.0, 104.0, calls=2000)

    assert new < 5e-6
    assert new < legacy / 3


def test_exact_wse_for_discharge_is_faster_than_bisection() -> None:
    xs = typical_section()
    target = 200.0 / math.sqrt(SLOPE)
    table = ConveyanceTable(xs, max_depth=25.0).table

    bisection = _seconds_per_call(_repeat_bisection, xs.elevations, xs.mannings_n, 10.0, target, calls=2000)
    one_off = _seconds_per_call(_repeat_wse_for_conveyance, xs.elevations, xs.mannings_n, 10.0, target, calls=20000)
    from_table = _seconds_per_call(_repeat_table_wse_for_conveyance, table, target, calls=20000)

    assert one_off < bisection / 2
    assert from_table < bisection / 5


def test_compound_conveyance_is_much_faster_than_the_legacy_calculation() -> None:
    """With banks at the channel's edges, one ordinate either side of the stream cell."""
    xs = typical_section()
    right, left = xs.elevations[250:].copy(), xs.elevations[250::-1].copy()
    n_right, n_left = xs.mannings_n[250:].copy(), xs.mannings_n[250::-1].copy()

    new = _seconds_per_call(_repeat_compound_conveyance, xs.elevations, xs.mannings_n, 10.0, 10.0, 104.0, calls=20000)
    legacy = _seconds_per_call(_repeat_legacy_compound_discharge, right, n_right, left, n_left, 10.0, 1, 104.0,
                               calls=2000)

    assert new < 5e-6
    assert new < legacy / 3


def test_exact_compound_wse_for_discharge_is_faster_than_bisection() -> None:
    xs = with_banks(typical_section(), 10.0, 10.0)
    target = 200.0 / math.sqrt(SLOPE)
    table = ConveyanceTable(xs, max_depth=25.0).table

    bisection = _seconds_per_call(_repeat_compound_bisection, xs.elevations, xs.mannings_n, 10.0, 10.0, target,
                                  calls=2000)
    one_off = _seconds_per_call(_repeat_wse_for_compound_conveyance, xs.elevations, xs.mannings_n, 10.0, 10.0, target,
                                calls=20000)
    from_table = _seconds_per_call(_repeat_compound_table_wse_for_conveyance, table, target, calls=20000)

    assert one_off < bisection / 2
    assert from_table < bisection / 5
