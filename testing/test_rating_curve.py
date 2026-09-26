from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.optimize import brentq

from arc.Automated_Rating_Curve_Generator import find_wse, flood_increments, objective_with_wse, safe_signs_differ
from arc.cross_section import calculate_discharge_from_wse
from arc.hydraulics import DepthRoughness, discharge, hydraulic_geometry, hydraulic_profile, wse_for_discharge
from arc.rating_curve import P, Q, T, V, WSE, max_flow_wse, rating_curve
from arc.xsection.xsection import XSection

BED = 100.0
N = 0.035
SLOPE = 0.001


def make_section(elevations, mannings_n=N, spacing=1.0) -> XSection:
    elevations = np.asarray(elevations, dtype=np.float64)
    return XSection(elevations, np.full(elevations.size, mannings_n), spacing)


def trapezoid(bottom_width=10.0, side_slope=2.0, half=60, spacing=1.0) -> XSection:
    x = spacing * np.arange(-half, half + 1)
    return make_section(BED + np.maximum(np.abs(x) - bottom_width / 2, 0.0) / side_slope, spacing=spacing)


def trapezoid_discharge(depth, bottom=10.0, side_slope=2.0, n=N, slope=SLOPE) -> float:
    top = bottom + 2 * side_slope * depth
    area = depth * (bottom + top) / 2
    perimeter = bottom + 2 * depth * math.hypot(1.0, side_slope)
    return area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / n


def channel_and_hollow() -> XSection:
    """A 10 m wide channel 2 m deep, whose right bank tops out at 102 m in front of a hollow 80 m wide whose floor
    is 1 m lower, so that at 102 m the water spills into it."""
    distance = np.arange(0.0, 101.0)
    left = BED + np.clip(distance - 5.0, 0.0, 2.0) + 0.1 * np.maximum(distance - 7.0, 0.0)
    right = BED + np.clip(distance - 5.0, 0.0, 2.0)
    right[8:88] = BED + 1.0
    right[88:] = BED + 10.0
    return make_section(np.concatenate([left[:0:-1], right]))


def test_the_last_increment_carries_the_maximum_flow_at_the_manning_depth() -> None:
    """A prismatic trapezoid, where the sampled cross section is exact."""
    xs = trapezoid()
    q_max = 30.0
    curve = rating_curve(xs, q_max, SLOPE, 10)
    depth = brentq(lambda y: trapezoid_discharge(y) - q_max, 1e-6, 20.0, xtol=1e-14)

    assert curve.valid and (curve.start, curve.last) == (0, 10)
    assert curve.max_wse == pytest.approx(BED + depth, abs=1e-9)
    assert curve.increments[-1, Q] == pytest.approx(q_max, rel=1e-9)
    assert np.allclose(np.diff(curve.increments[:, WSE]), depth / 10, atol=1e-12)
    for q, v, t, wse, p in curve.increments:
        y = wse - BED
        assert q == pytest.approx(trapezoid_discharge(y), rel=1e-9)
        assert t == pytest.approx(10.0 + 4.0 * y, rel=1e-12)
        assert p == pytest.approx(10.0 + 2 * y * math.hypot(1.0, 2.0), rel=1e-12)
        assert v == pytest.approx(q / (y * (10.0 + 2.0 * y)), rel=1e-12)


def test_the_slope_factor_and_roughness_reach_the_rating_curve() -> None:
    xs = trapezoid()
    roughness = DepthRoughness()
    curve = rating_curve(xs, 30.0, SLOPE, 8, roughness=roughness, slope_factor=1.3)

    assert curve.max_wse == pytest.approx(wse_for_discharge(xs, 30.0, SLOPE, roughness=roughness, slope_factor=1.3))
    for q, _, _, wse, _ in curve.increments:
        assert q == pytest.approx(discharge(xs, SLOPE, wse=wse, roughness=roughness, slope_factor=1.3), rel=1e-12)
    # Rougher shallow water needs a deeper channel for the same flow
    assert curve.max_wse > rating_curve(xs, 30.0, SLOPE, 8, slope_factor=1.3).max_wse


def test_the_baseflow_lowers_the_first_increment_as_legacy_did() -> None:
    xs = trapezoid()
    plain = rating_curve(xs, 30.0, SLOPE, 10)
    first = plain.increments[0, Q]

    above = rating_curve(xs, 30.0, SLOPE, 10, baseflow=first * 0.5)
    below = rating_curve(xs, 30.0, SLOPE, 10, baseflow=first * 2.0)

    assert above.increments[0, Q] == pytest.approx(first * 0.5 - 0.001)
    assert np.array_equal(above.increments[1:], plain.increments[1:])
    assert np.array_equal(below.increments, plain.increments)
    assert np.array_equal(rating_curve(xs, 30.0, SLOPE, 10, baseflow=0.0005).increments, plain.increments)


def test_no_flow_or_no_increments_give_a_curve_without_any() -> None:
    xs = trapezoid()

    for curve in (rating_curve(xs, 0.0, SLOPE, 10), rating_curve(xs, math.nan, SLOPE, 10),
                  rating_curve(xs, 30.0, SLOPE, 0)):
        assert (curve.start, curve.last) == (-1, 0) and not curve.valid
        assert np.isnan(curve.increments).all()


def test_a_section_that_can_t_carry_the_flow_has_no_rating_curve() -> None:
    walls = make_section([9999.0, BED, 9999.0])
    walls.mannings_n[[0, 2]] = 9999.0

    assert rating_curve(trapezoid(), 30.0, 0.0, 10) is None  # no slope
    assert math.isnan(max_flow_wse(trapezoid(), 30.0, -1.0))
    assert rating_curve(make_section([BED]), 30.0, SLOPE, 10) is None  # no width at all


def test_a_spill_far_below_the_maximum_flow_isn_t_acceptable() -> None:
    """Just below the bank top the channel carries less than half the maximum flow, so legacy's test for an
    acceptable result fails there, as it failed for legacy."""
    xs = channel_and_hollow()
    below = discharge(xs, SLOPE, wse=102.0)
    above = discharge(xs, SLOPE, wse=102.0 + 1e-9)
    assert above > 3 * below

    assert wse_for_discharge(xs, 2.5 * below, SLOPE) == 102.0
    assert rating_curve(xs, 2.5 * below, SLOPE, 10) is None
    curve = rating_curve(xs, 1.5 * below, SLOPE, 10)
    assert curve is not None and curve.max_wse == 102.0
    assert curve.increments[-1, Q] == pytest.approx(below, rel=1e-12)  # the channel's, just below the spill


def test_where_discharge_falls_the_last_increment_is_repeated_or_the_water_raised_a_centimetre() -> None:
    """Water spreading over a flat floodplain beside a small V channel carries less as it first rises above it."""
    distance = np.abs(np.arange(-60.0, 61.0))
    xs = make_section(np.where(distance > 52, BED + 2.0 + (distance - 52),
                               np.where(distance <= 2, BED + distance, BED + 2.0)))
    assert discharge(xs, SLOPE, wse=BED + 2.05) < discharge(xs, SLOPE, wse=BED + 2.0)
    q_max = discharge(xs, SLOPE, wse=BED + 2.5)
    outcomes = set()
    for count in range(5, 80):
        curve = rating_curve(xs, q_max, SLOPE, count)
        increments = curve.increments
        assert np.all(np.diff(increments[:, Q]) >= 0.0)
        assert increments[-1, Q] == pytest.approx(q_max, rel=1e-9)
        step = (curve.max_wse - BED) / count
        for i in range(1, count):
            if np.array_equal(increments[i], increments[i - 1]):
                outcomes.add("repeated")
            elif not math.isclose(increments[i, WSE], BED + step * (i + 1), abs_tol=1e-9):
                outcomes.add("raised")
                assert increments[i, WSE] < BED + step * (i + 2)  # no higher than the next increment
    assert outcomes == {"repeated", "raised"}


# --- Parity with legacy's flood_increments ------------------------------------------------------------------------


def legacy_increments(xs: XSection, thalweg, step, count, slope, q_cap, banks):
    """Legacy flood_increments on the same cross section, as legacy held it: side 1 to the right."""
    center = xs.elevations.size // 2
    side1, side2 = xs.elevations[center:].copy(), xs.elevations[center::-1].copy()
    n1, n2 = xs.mannings_n[center:].copy(), xs.mannings_n[center::-1].copy()
    out = np.full((1, 8 + 5 * count), np.nan)
    args = (side1, side1.size, n1, side2, side2.size, n2, float(xs.ordinate_distance))
    start, last = flood_increments(count + 1, step, args, thalweg, slope, q_cap, out, 0, False, 6.0, 1.0, 1.0, 1.0,
                                   banks[1], banks[0])
    return start, last, out[0, 8:].reshape(count, 5)


def random_valley(rng) -> XSection:
    """A valley with flats, rises and hollows, whose ends stand high enough to keep the water inside."""
    size = 2 * int(rng.integers(8, 40)) + 1
    center = size // 2
    steps = rng.choice([0.0, 0.0, 0.05, 0.3, 1.0, -0.4], size - 1)
    right = np.concatenate([[0.0], np.cumsum(steps[:size - 1 - center])])
    left = np.concatenate([[0.0], np.cumsum(steps[size - 1 - center:])])
    # Off the millimetre but at the stream cell, so that no increment on the millimetre ties with ground, where
    # legacy's rounding of the water surface would decide which side of it the water is
    elevations = np.round(BED + np.concatenate([left[:0:-1], right]), 3) + 0.0004
    elevations[center] = BED
    elevations[0] = elevations[-1] = BED + 50.0
    return XSection(elevations, np.full(size, N), float(rng.choice([1.0, 7.5, 30.0])))


def test_the_increments_match_legacy_s_where_the_hydraulics_agree() -> None:
    """Uniform n, water inside the section, banks on ordinates, and increments on the millimetre, so that legacy's
    rounding changes nothing but the last digit it keeps."""
    from arc.rating_curve import _increments

    rng = np.random.default_rng(20)
    compared = fixed = 0
    for _ in range(3000):
        xs = random_valley(rng)
        center = xs.elevations.size // 2
        banks = (-1, -1)
        if rng.random() < 0.5:
            banks = (int(rng.integers(1, center)), int(rng.integers(1, center)))
            xs.left_bank_distance, xs.right_bank_distance = banks[0] * xs.ordinate_distance, \
                banks[1] * xs.ordinate_distance
        count = int(rng.integers(5, 31))
        step = float(rng.integers(20, 200)) / 1000.0
        thalweg = float(xs.elevations[center])
        if thalweg + step * count >= BED + 50.0:
            continue
        top = thalweg + step * count
        legacy_q = calculate_discharge_from_wse(np.round(top, 3), 1.0, xs.elevations[center:].copy(),
                                                center + 1, xs.mannings_n[center:].copy(),
                                                xs.elevations[center::-1].copy(), center + 1,
                                                xs.mannings_n[center::-1].copy(), xs.ordinate_distance,
                                                6.0, 1.0, 1.0, 1.0, banks[1], banks[0]) * math.sqrt(SLOPE)
        q_cap = round(legacy_q, 3)
        start, last, expected = legacy_increments(xs, thalweg, step, count, SLOPE, q_cap, banks)
        out = np.full((count, 5), np.nan)
        got = _increments(*hydraulic_profile(xs), float(xs.left_bank_distance), float(xs.right_bank_distance), 1.0,
                          1.0, 1.0, thalweg, top, count, math.sqrt(SLOPE), q_cap, out)
        rounded = np.round(out, 3)
        # Where legacy's discharges rounded to 3 decimals tie, or round away a trickle to 0, it can decide
        # differently; leave those out
        if np.any(np.abs(np.diff(expected[:, Q])) < 0.002) or np.any(expected[:, Q] < 0.002):
            continue
        compared += 1
        fixed += int(np.any(np.diff(expected[:, WSE]) < step * 0.999)) + int(np.any(~np.isclose(
            np.diff(expected[:, WSE]), step)))
        assert got == (start, last)
        for column in (T, WSE, P):
            assert np.allclose(rounded[:, column], expected[:, column], atol=1.5e-3), column
        # Without banks, legacy worked out discharge from its area and perimeter rounded to 3 decimals, which moves
        # it by up to 5/3 and 2/3 of their rounding relative to them
        area, perimeter = out[:, Q] / out[:, V], out[:, P]
        tolerance = out[:, Q] * (5 / 3 * 0.0005 / area + 2 / 3 * 0.0005 / perimeter) + 1.1e-3
        assert np.all(np.abs(rounded[:, Q] - expected[:, Q]) <= tolerance)
        # Its velocity is that discharge over the rounded area, which is only close where the area is well above a
        # thousandth
        wide = area > 1.0
        assert np.allclose(rounded[wide, V], expected[wide, V], rtol=2e-3, atol=2e-3)
    assert compared > 2000 and fixed > 20  # and some had legacy's fix-ups


# --- Speed ------------------------------------------------------------------------------------------------------------


def legacy_cell(xs: XSection, q_max: float, slope: float, count: int, banks: tuple[int, int]) -> None:
    """Legacy calculate_hydraulic_data_for_cell's search for the maximum flow's water surface and its increments,
    without the slope search it only ran when those missed the maximum flow by half."""
    center = xs.elevations.size // 2
    side1, side2 = xs.elevations[center:].copy(), xs.elevations[center::-1].copy()
    n1, n2 = xs.mannings_n[center:].copy(), xs.mannings_n[center::-1].copy()
    args = (side1, side1.size, n1, side2, side2.size, n2, float(xs.ordinate_distance), 6.0, 2.0, 1.0, 1.0,
            banks[1], banks[0])
    thalweg = float(side1[0])
    sqrt_slope = slope ** 0.5
    f_lower = objective_with_wse(thalweg + 0.01, sqrt_slope, q_max, args)
    f_upper = objective_with_wse(thalweg + 24.99, sqrt_slope, q_max, args)
    wse_final, q_sum = -999.0, 0.0
    if safe_signs_differ(f_lower, f_upper):
        wse_final = np.round(brentq(objective_with_wse, thalweg + 0.01, thalweg + 24.99, xtol=0.001,
                                    args=(sqrt_slope, q_max, args)), 3)
        q_sum = calculate_discharge_from_wse(wse_final, sqrt_slope, *args)
    wse, _, _ = find_wse(101, thalweg, 0.5, q_max, args, slope)
    wse = max(wse - 0.5, thalweg)
    wse, _, _ = find_wse(101, wse, 0.05, q_max, args, slope)
    wse = max(wse - 0.05, thalweg)
    wse_test, q_test, _ = find_wse(2501, wse, 0.01, q_max, args, slope)
    if abs(q_test - q_max) < abs(q_sum - q_max):
        wse_final, q_sum = wse_test, q_test
    out = np.full((1, 8 + 5 * count), np.nan)
    flood_increments(count + 1, round((wse_final - thalweg) / count, 3), args[:7], thalweg, slope, round(q_sum, 3),
                     out, 0, False, 6.0, 2.0, 1.0, 1.0, banks[1], banks[0])


def test_a_rating_curve_is_faster_than_legacy_s() -> None:
    """A 5 km cross section at 10 m spacing, with a channel 20 m wide and 2 m deep in a valley with 2% side slopes,
    banks at its edges and legacy's default depth-varying roughness, to 30 increments."""
    distance = np.abs(10.0 * np.arange(-250, 251))
    xs = make_section(BED + np.minimum(distance, 10.0) * 0.2 + np.maximum(distance - 10.0, 0.0) * 0.02, spacing=10.0)
    xs.left_bank_distance = xs.right_bank_distance = 10.0
    roughness = DepthRoughness()
    rating_curve(xs, 200.0, SLOPE, 30, roughness=roughness)
    legacy_cell(xs, 200.0, SLOPE, 30, (1, 1))

    def best(function, repeats):
        times = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(repeats):
                function()
            times.append((time.perf_counter() - start) / repeats)
        return min(times)

    new = best(lambda: rating_curve(xs, 200.0, SLOPE, 30, roughness=roughness), 200)
    legacy = best(lambda: legacy_cell(xs, 200.0, SLOPE, 30, (1, 1)), 20)
    assert new < legacy / 3, (new, legacy)
