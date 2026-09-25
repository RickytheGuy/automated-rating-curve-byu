from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np
import pytest

from arc import Automated_Rating_Curve_Generator as legacy
from arc.bathymetry import Banks, banks_for_width, single_cell_banks
from arc.bathymetry.smoothing import (MIN_GRADE, ReachSections, ReachWidths, filter_reach_widths,
                                      network_outlet_elevations, order_reach, reach_bank_observations,
                                      reach_bank_surface, reach_baseline, reach_network, smooth_bank_elevations)
from arc.xsection.xsection import XSection

BED = 100.0
NO_BANKS = Banks("none", math.nan, math.nan, math.nan, math.nan, False, False)


def flat_section(spacing=1.0, ordinates=101, bed=BED) -> XSection:
    elevations = np.full(ordinates, bed)
    return XSection(elevations, np.full(ordinates, 0.035), spacing)


def section_with_banks(width, spacing=1.0) -> tuple[XSection, Banks]:
    xs = flat_section(spacing)
    return xs, banks_for_width(xs, width)


def banks_at(left_elevation, right_elevation, valid=True) -> Banks:
    return Banks("target_width", 5.0, 5.0, left_elevation, right_elevation, False, valid)


def line_graph(*reaches, lengths=None) -> nx.DiGraph:
    network = nx.DiGraph()
    for k, reach in enumerate(reaches):
        network.add_node(reach, length=100.0 if lengths is None else lengths[k])
    network.add_edges_from(zip(reaches[:-1], reaches[1:]))
    return network


# --- The width filter ---------------------------------------------------------------------------------------------


def test_outlying_widths_are_rebuilt_at_the_reach_median() -> None:
    """Widths of 10, 20, 20, 20 and 40 m have their 25th, 50th and 75th percentiles all at 20 m, so the 10 and 40 m
    channels are rebuilt 20 m wide, half each side. The others keep their own banks, even at a percentile. Legacy
    rounded banks to whole spacings."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 20.0, 20.0, 20.0, 40.0])))
    banks[2] = Banks("width_to_depth_ratio", 5.0, 15.0, BED, BED, False, True)

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths == ReachWidths(20.0, 20.0, 20.0)
    assert [b.top_width for b in filtered] == pytest.approx([20.0] * 5)
    assert filtered[1:4] == banks[1:4]
    assert filtered[0].method == filtered[4].method == "target_width"
    assert (filtered[0].left, filtered[0].right) == pytest.approx((10.0, 10.0))


def test_a_median_a_cross_section_can_t_hold_is_widened_a_spacing_at_a_time() -> None:
    """The median is 10 m, but a cross section with 8 m spacing can't make a channel between one and two spacings
    wide, so it is rebuilt 18 m wide, still within the 75th percentile, 20 m."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 20.0])))
    wide = flat_section(spacing=8.0)
    sections.append(wide)
    banks.append(banks_for_width(wide, 40.0))

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths == ReachWidths(10.0, 10.0, 20.0)
    assert filtered[4].top_width == pytest.approx(18.0)
    assert filtered[4].valid and not filtered[4].single_cell


def test_a_channel_that_can_t_be_rebuilt_within_the_75th_percentile_becomes_a_single_cell() -> None:
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 12.0])))
    wide = flat_section(spacing=8.0)
    sections.append(wide)
    banks.append(banks_for_width(wide, 40.0))

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths.q75 == pytest.approx(12.0)
    assert filtered[4] == single_cell_banks(wide)


def test_a_channel_that_can_t_be_a_single_cell_either_is_left_alone() -> None:
    """With no ordinate on the raster on one side of the stream cell, a single-cell channel isn't valid."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 12.0])))
    walled = flat_section(spacing=8.0)
    walled.elevations[:50] = 9999.0
    outlier = Banks("width_to_depth_ratio", 1.0, 40.0, BED, BED, False, True)
    sections.append(walled)
    banks.append(outlier)

    filtered, _ = filter_reach_widths(sections, banks)

    assert filtered[4] == outlier


def test_cross_sections_without_valid_banks_get_the_median_width() -> None:
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 12.0, 14.0])))
    sections += [flat_section(), flat_section(spacing=8.0)]
    banks += [NO_BANKS, NO_BANKS]

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths.median == pytest.approx(12.0)
    assert filtered[3].valid and filtered[3].top_width == pytest.approx(12.0)
    assert filtered[4] == NO_BANKS  # 12 m is between one and two of its spacings


def test_a_reach_without_valid_banks_is_left_alone() -> None:
    sections = [flat_section(), flat_section()]

    assert filter_reach_widths(sections, [NO_BANKS, NO_BANKS]) == ([NO_BANKS, NO_BANKS], None)


def test_a_single_cell_channel_is_two_spacings_wide() -> None:
    """Its banks are the ordinates either side, where legacy counted one spacing."""
    xs = flat_section(spacing=10.0)

    assert single_cell_banks(xs).top_width == pytest.approx(20.0)


# --- Observations -------------------------------------------------------------------------------------------------


def test_an_observation_is_the_lower_bank_above_the_stream_cell() -> None:
    """A bank at or below the stream cell is ignored, and banks that aren't valid give no observation. Legacy lost
    the cross section's observation when a bank was below the stream cell."""
    sections = [flat_section() for _ in range(5)]
    banks = [banks_at(103.0, 102.0), banks_at(99.0, 102.5), banks_at(BED, 101.5), banks_at(BED, BED),
             banks_at(104.0, 104.0, valid=False)]

    observations, lower, upper = reach_bank_observations(sections, banks)

    np.testing.assert_array_equal(observations, [102.0, 102.5, 101.5, np.nan, np.nan])
    assert (lower, upper) == (-math.inf, math.inf)


def test_observations_outside_the_2nd_and_97th_percentiles_are_left_out() -> None:
    elevations = [100.5, 101.0, 101.0, 101.0, 101.0, 110.0]
    sections = [flat_section(bed=99.0) for _ in elevations]

    observations, lower, upper = reach_bank_observations(sections, [banks_at(z, z) for z in elevations])

    assert (lower, upper) == pytest.approx(tuple(np.percentile(elevations, [2, 97])))
    np.testing.assert_array_equal(observations, [np.nan, 101.0, 101.0, 101.0, 101.0, np.nan])


def test_fewer_than_four_observations_are_all_kept() -> None:
    sections = [flat_section(bed=99.0) for _ in range(3)]

    observations, lower, upper = reach_bank_observations(sections, [banks_at(z, z) for z in [100.0, 101.0, 150.0]])

    np.testing.assert_array_equal(observations, [100.0, 101.0, 150.0])
    assert (lower, upper) == (-math.inf, math.inf)


# --- Ordering a reach's cross sections ----------------------------------------------------------------------------


def hairpin():
    """Cells east along row 10, down through column 16, and back west along row 13."""
    cells = [(10, c) for c in range(5, 16)] + [(11, 16), (12, 16)] + [(13, c) for c in range(15, 4, -1)]
    return np.array([r for r, _ in cells]), np.array([c for _, c in cells])


def test_a_reach_is_ordered_towards_the_reach_downstream_of_it() -> None:
    rows, cols = hairpin()
    network = line_graph(1, 2)

    order, stations = order_reach(network, 1, rows[::-1], cols[::-1], 10.0, 10.0,
                                  {1: (rows, cols), 2: (np.array([13]), np.array([4]))}, np.zeros(rows.size))

    assert order.tolist() == list(range(rows.size - 1, -1, -1))
    assert stations[-1] == pytest.approx(200.0 + 2 * math.hypot(10, 10) + 10.0)


def test_an_outlet_is_ordered_away_from_the_reach_upstream_of_it() -> None:
    rows, cols = hairpin()
    network = line_graph(9, 1)

    order, _ = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols), 9: (np.array([13]), np.array([4]))},
                           np.zeros(rows.size))

    assert order.tolist() == list(range(rows.size - 1, -1, -1))


def test_the_first_reach_downstream_with_cross_sections_sets_the_downstream_end() -> None:
    """Reach 1 splits into reaches 2 and 3, and reach 2 has no cross sections, so reach 3 sets the end."""
    rows, cols = hairpin()
    network = nx.DiGraph()
    network.add_edges_from([(1, 2), (1, 3)])
    cells = {1: (rows, cols), 2: (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
             3: (np.array([13]), np.array([4]))}

    order, _ = order_reach(network, 1, rows, cols, 10.0, 10.0, cells, np.zeros(rows.size))

    assert order.tolist() == list(range(rows.size))


def test_a_reach_by_itself_is_ordered_along_its_cells_from_its_highest_observation() -> None:
    """Legacy ordered a reach with no neighbour along a straight line through its cells, which mixes up the
    hairpin's two arms, and numbered its stations 0, 1, 2 and so on."""
    rows, cols = hairpin()
    observations = np.linspace(90.0, 110.0, rows.size)  # rising along the cells, so the far end is upstream
    network = nx.DiGraph()
    network.add_node(1, length=300.0)

    order, stations = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols)}, observations)

    assert order.tolist() == list(range(rows.size - 1, -1, -1))
    np.testing.assert_allclose(np.diff(stations)[:10], 10.0)
    projection = np.argsort(cols, kind="stable")
    legacy_order, legacy_stations = legacy._order_reach_stream_cells_from_network(
        network, 1, [{"row": int(r), "col": int(c)} for r, c in zip(rows, cols)], {}, projection, observations, 10.0,
        10.0)
    assert set(np.diff(rows[legacy_order])) != {0} and legacy_stations[1] == 1.0


def test_a_reach_whose_neighbours_have_no_cross_sections_is_turned_by_its_end_observations() -> None:
    """Its first ten cross sections' mean observation is higher than its last ten's, so that end is upstream."""
    rows, cols = np.zeros(30, dtype=np.int64), np.arange(30)
    observations = np.r_[np.full(10, 105.0), np.full(10, np.nan), np.full(10, 95.0)]
    network = line_graph(1, 2)

    order, stations = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols)}, observations)

    assert order.tolist() == list(range(30))
    np.testing.assert_allclose(stations, 10.0 * np.arange(30))


# --- The network --------------------------------------------------------------------------------------------------


def random_network(rng, size):
    network = nx.DiGraph()
    for reach in range(1, size + 1):
        network.add_node(reach, length=float(rng.choice([rng.uniform(50.0, 5000.0), 1.0])))
    for reach in range(1, size):
        if rng.random() > 0.12:
            network.add_edge(reach, int(rng.integers(reach + 1, size + 1)))
    return network


def test_the_network_matches_legacy() -> None:
    """Outlet elevations, grades and each reach's baseline, on random networks with reaches missing observations."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        network = random_network(rng, int(rng.integers(1, 30)))
        below = {reach: len(nx.descendants(network, reach)) for reach in network}
        minima, maxima = {}, {}
        for reach in network:
            if rng.random() < 0.75:
                minima[reach] = 100.0 + 3.0 * below[reach] + float(rng.normal(0.0, 4.0))
                maxima[reach] = minima[reach] + float(abs(rng.normal(2.0, 2.0)))
        legacy_network = network.copy()
        expected = legacy._estimate_network_smoothed_reach_min_bank_elevations(legacy_network, minima, maxima)

        outlets, grades = network_outlet_elevations(network, minima, maxima)

        assert outlets.keys() == expected.keys()
        for reach in outlets:
            assert outlets[reach] == pytest.approx(expected[reach], abs=1e-9)
        assert grades == pytest.approx({reach: data["bank_elevation_grade"]
                                        for reach, data in legacy_network.nodes(data=True)
                                        if "bank_elevation_grade" in data}, rel=1e-12)
        for reach in outlets:
            stations = np.sort(rng.uniform(0.0, 1000.0, int(rng.integers(1, 8))))
            want = legacy._interpolate_reach_bank_elevation_surface(legacy_network, reach, stations, expected)
            baseline, fractions = reach_baseline(network, reach, stations, outlets, grades)
            np.testing.assert_allclose(baseline, want[0], rtol=0, atol=1e-9)
            np.testing.assert_allclose(fractions, want[1], rtol=0, atol=1e-12)


def test_equal_reach_minima_fall_at_the_minimum_grade() -> None:
    """Legacy's test: three reaches in a line with the same lowest bank each fall 1 cm per 100 m."""
    outlets, grades = network_outlet_elevations(line_graph(1, 2, 3), {1: 100.0, 2: 100.0, 3: 100.0}, {})

    assert outlets == pytest.approx({1: 100.0, 2: 100.0 - 0.01, 3: 100.0 - 0.02})
    assert grades == pytest.approx({1: MIN_GRADE, 2: MIN_GRADE, 3: MIN_GRADE})


def test_a_headwater_falls_from_its_highest_bank_and_an_outlet_to_its_lowest() -> None:
    """Legacy's tests of a headwater, an outlet and a reach on its own."""
    outlets, grades = network_outlet_elevations(line_graph(1, 2, 3), {1: 100.0, 2: 90.0, 3: 89.0},
                                                {1: 105.0, 2: 95.0, 3: 90.0})

    assert outlets == pytest.approx({1: 100.0, 2: 90.0, 3: 89.0})
    assert grades == pytest.approx({1: 0.05, 2: 0.1, 3: 0.01})

    alone = nx.DiGraph()
    alone.add_node(1, length=100.0)
    assert network_outlet_elevations(alone, {1: 90.0}, {1: 100.0}) == ({1: 90.0}, pytest.approx({1: 0.1}))


def test_a_reach_without_observations_is_filled_in_along_the_network() -> None:
    """Reach 2 has none, so its outlet is interpolated between reaches 1 and 3 by length."""
    network = line_graph(1, 2, 3, lengths=[100.0, 300.0, 100.0])

    outlets, _ = network_outlet_elevations(network, {1: 110.0, 3: 100.0}, {1: 112.0, 3: 101.0})

    assert outlets[2] == pytest.approx(110.0 - 10.0 * 300.0 / 400.0)


def test_an_outlet_that_rises_above_the_reach_upstream_is_lowered() -> None:
    outlets, grades = network_outlet_elevations(line_graph(1, 2), {1: 100.0, 2: 105.0}, {1: 101.0, 2: 106.0})

    assert outlets[2] == pytest.approx(100.0 - MIN_GRADE * 100.0)
    assert grades[2] == pytest.approx(MIN_GRADE)


def test_a_network_is_built_from_the_stream_table() -> None:
    """Missing, self and unknown downstream IDs make outlets, and a reach's first row counts."""
    network = reach_network([1, 2, 3, 4, 2, 5], [2, 3, None, 99, 4, "4"], [100.0, -1.0, float("nan"), 50.0, 7.0, 20.0])

    assert sorted(network.edges) == [(1, 2), (2, 3), (5, 4)]
    assert [network.nodes[r]["length"] for r in [1, 2, 3, 4, 5]] == [100.0, 1.0, 1.0, 50.0, 20.0]
    assert reach_network([7], [7], [10.0]).out_degree(7) == 0


# --- The surface along a reach ------------------------------------------------------------------------------------


def test_an_observation_below_the_line_becomes_an_anchor() -> None:
    """Legacy's test: the stretch from the start is refitted through the low observation, and from it the line
    heads for the outlet again."""
    surface, grades, anchors = reach_bank_surface([np.nan, 8.5, np.nan, np.nan], [10.0, 9.0, 8.0, 7.0],
                                                  [0.0, 1 / 3, 2 / 3, 1.0], 300.0, 7.0, min_grade=0.001)

    np.testing.assert_allclose(surface, [10.0, 8.5, 7.75, 7.0])
    np.testing.assert_allclose(grades, [0.015, 0.0075, 0.0075, 0.0075])
    assert anchors.tolist() == [False, True, False, False]


def test_each_lower_observation_refits_from_the_last_anchor() -> None:
    surface, grades, anchors = reach_bank_surface([np.nan, np.nan, 7.5, 6.4, np.nan], [10.0, 9.0, 8.0, 7.0, 6.0],
                                                  [0.0, 0.25, 0.5, 0.75, 1.0], 400.0, 6.0, min_grade=0.001)

    np.testing.assert_allclose(surface, [10.0, 8.75, 7.5, 6.4, 6.0])
    np.testing.assert_allclose(grades, [0.0125, 0.0125, 0.011, 0.004, 0.004])
    assert anchors.tolist() == [False, False, True, True, False]


def test_an_anchor_can_t_be_too_low_to_reach_the_outlet() -> None:
    surface, _, _ = reach_bank_surface([np.nan, -100.0, np.nan], [10.0, 9.0, 8.0], [0.0, 0.5, 1.0], 100.0, 8.0,
                                       min_grade=0.001)

    np.testing.assert_allclose(surface, [10.0, 8.05, 8.0])


def test_only_observations_within_the_bounds_and_above_the_stream_cell_are_anchors() -> None:
    """Legacy's test: 10.5 is above the upper bound, 8.5 is level with its stream cell and 7.0 is below the lower
    bound."""
    surface, _, anchors = reach_bank_surface([np.nan, 10.5, 9.5, 8.5, 7.0], [12.0, 11.0, 10.0, 9.0, 8.0],
                                             [0.0, 0.25, 0.5, 0.75, 1.0], 100.0, 8.0, min_grade=0.001,
                                             thalwegs=[11.0, 9.0, 9.0, 8.5, 6.0], lower=8.0, upper=10.0)

    np.testing.assert_allclose(surface, [12.0, 10.75, 9.5, 8.75, 8.0])
    assert anchors.tolist() == [False, False, True, False, False]


def test_an_observation_just_above_its_stream_cell_can_be_an_anchor() -> None:
    """Legacy also needed it more than np.isclose's tolerance above, 1.02 mm here, where it is 1 mm above."""
    args = ([np.nan, 102.0, np.nan], [110.0, 105.0, 100.0], [0.0, 0.5, 1.0], 100.0, 100.0)
    thalwegs = [np.nan, 101.999, np.nan]

    assert reach_bank_surface(*args, thalwegs=thalwegs)[2].tolist() == [False, True, False]
    assert legacy._anchor_interpolated_bank_surface_to_cell_observations(
        *args, MIN_GRADE, None, np.asarray(thalwegs))[2].tolist() == [False, False, False]


def test_the_first_cross_section_is_no_higher_than_the_reaches_flowing_in() -> None:
    surface, _, anchors = reach_bank_surface([9.0, np.nan], [10.0, 9.0], [0.0, 1.0], 100.0, 9.0, min_grade=0.001,
                                             ceiling=10.0)

    np.testing.assert_allclose(surface, [9.1, 9.0])
    assert anchors.tolist() == [True, False]


def test_the_surface_matches_legacy() -> None:
    rng = np.random.default_rng(3)
    for _ in range(1000):
        n = int(rng.integers(1, 40))
        fractions = np.sort(rng.random(n))
        fractions[0] = 0.0
        if n > 1:
            fractions[-1] = 1.0
        outlet = float(rng.uniform(90.0, 100.0))
        upstream = outlet + float(rng.uniform(0.0, 20.0))
        baseline = np.minimum.accumulate(upstream + fractions * (outlet - upstream))
        observations = baseline + rng.normal(0.0, 3.0, n)
        observations[rng.random(n) < 0.3] = np.nan
        thalwegs = observations - rng.uniform(-1.0, 3.0, n)
        thalwegs[np.isclose(observations, thalwegs)] -= 0.1  # where the two differ on purpose (see above)
        lower, upper = sorted(rng.uniform(85.0, 115.0, 2)) if rng.random() < 0.5 else (-np.inf, np.inf)
        ceiling = float(upstream + rng.normal(0.0, 2.0)) if rng.random() < 0.5 else None
        length = float(rng.uniform(10.0, 5000.0))

        want = legacy._anchor_interpolated_bank_surface_to_cell_observations(
            observations, baseline, fractions, length, outlet, MIN_GRADE, ceiling, thalwegs, lower, upper)
        got = reach_bank_surface(observations, baseline, fractions, length, outlet, ceiling=ceiling,
                                 thalwegs=thalwegs, lower=lower, upper=upper)

        np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(got[1], want[1], rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(got[2], want[2])


def test_the_baseline_never_rises_even_with_stations_out_of_order() -> None:
    """Legacy's guard, kept: stations 0, 60, 30 and 100 m still give a baseline that only falls."""
    network = line_graph(1)
    network.nodes[1]["bank_elevation_grade"] = 0.1  # where legacy reads it
    stations = np.array([0.0, 60.0, 30.0, 100.0])

    baseline, _ = reach_baseline(network, 1, stations, {1: 90.0}, {1: 0.1})

    np.testing.assert_allclose(baseline, [100.0, 94.0, 94.0, 90.0])
    np.testing.assert_allclose(baseline, legacy._interpolate_reach_bank_elevation_surface(network, 1, stations,
                                                                                         {1: 90.0})[0])


def test_a_reach_of_one_cross_section_is_all_outlet() -> None:
    network = line_graph(1)

    baseline, fractions = reach_baseline(network, 1, [0.0], {1: 90.0}, {1: 0.01})

    assert baseline.tolist() == [90.0] and fractions.tolist() == [1.0]


# --- Everything ---------------------------------------------------------------------------------------------------


def channel_section(bank_top, bed, width=20.0, spacing=2.0, ordinates=81) -> XSection:
    """A 1:1 channel whose sides rise from the bed to bank_top at width / 2 from the stream cell, then a floodplain
    rising 1 cm per metre."""
    distance = spacing * np.abs(np.arange(ordinates) - ordinates // 2)
    side = np.clip(distance - (width / 2 - (bank_top - bed)), 0.0, bank_top - bed)
    elevations = bed + side + 0.01 * np.maximum(distance - width / 2, 0.0)
    return XSection(elevations, np.full(ordinates, 0.035), spacing)


def small_network():
    """Reaches 1 and 2 join as reach 3, each of 20 cross sections along a row of 10 m cells."""
    network = reach_network([1, 2, 3], [3, 3, None], [200.0, 200.0, 200.0])
    reaches = {}
    starts = {1: (0, 0), 2: (2, 0), 3: (1, 21)}
    tops = {1: 110.0, 2: 108.0, 3: 104.0}
    for reach, (row, col) in starts.items():
        sections = [channel_section(tops[reach] - 0.05 * k, tops[reach] - 2.0 - 0.05 * k) for k in range(20)]
        if reach == 3:
            sections[10] = channel_section(103.3, 101.3)  # a bank lower than the ones either side
        banks = [banks_for_width(xs, 20.0) for xs in sections]
        rows, cols = np.full(20, row), np.arange(col, col + 20)
        reaches[reach] = ReachSections(rows, cols, sections, banks)
    return network, reaches


def test_bank_elevations_fall_along_the_network() -> None:
    network, reaches = small_network()

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0)

    for reach, result in smoothed.items():
        ordered = result.bank_elevations[result.order]
        assert np.all(np.isfinite(ordered))
        assert np.all(np.diff(ordered) <= -MIN_GRADE * 10.0 * (1 - 1e-9))
    inflow = min(smoothed[1].bank_elevations[smoothed[1].order[-1]], smoothed[2].bank_elevations[smoothed[2].order[-1]])
    assert smoothed[3].bank_elevations[smoothed[3].order[0]] <= inflow
    assert smoothed[3].anchors[10]
    assert smoothed[3].bank_elevations[10] == pytest.approx(103.3, abs=1e-9)


def test_results_come_back_in_the_order_the_cross_sections_were_given() -> None:
    network, reaches = small_network()
    expected = smooth_bank_elevations(network, reaches, 10.0, 10.0)[3]
    shuffle = np.random.default_rng(0).permutation(20)
    given = reaches[3]
    reaches[3] = ReachSections(given.rows[shuffle], given.cols[shuffle], [given.sections[k] for k in shuffle],
                               [given.banks[k] for k in shuffle])

    shuffled = smooth_bank_elevations(network, reaches, 10.0, 10.0)[3]

    np.testing.assert_allclose(shuffled.bank_elevations, expected.bank_elevations[shuffle], rtol=1e-15)
    np.testing.assert_array_equal(shuffled.anchors, expected.anchors[shuffle])
    np.testing.assert_array_equal(shuffled.observations, expected.observations[shuffle])
    assert shuffled.order.tolist() == np.argsort(shuffle)[expected.order].tolist()


def test_each_cross_section_keeps_its_own_banks() -> None:
    network, reaches = small_network()

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0)

    assert smoothed[1].banks == list(reaches[1].banks)
    assert smoothed[1].widths == ReachWidths(20.0, 20.0, 20.0)


def test_a_reach_with_cross_sections_must_be_in_the_network() -> None:
    network, reaches = small_network()
    network.remove_node(2)

    with pytest.raises(ValueError, match="aren't in the stream network: 2"):
        smooth_bank_elevations(network, reaches, 10.0, 10.0)


def test_smoothing_needs_a_bank_elevation_somewhere() -> None:
    network, reaches = small_network()
    for reach, sections in reaches.items():
        reaches[reach] = sections._replace(banks=[NO_BANKS] * len(sections.banks))

    with pytest.raises(ValueError, match="No reach has a bank elevation"):
        smooth_bank_elevations(network, reaches, 10.0, 10.0)


# --- Speed --------------------------------------------------------------------------------------------------------


def _best_seconds(function, repeats: int = 5) -> float:
    function()
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def test_ordering_and_anchoring_are_faster_than_legacy_s() -> None:
    """A reach of 500 cross sections winding along its cells."""
    rng = np.random.default_rng(2)
    x = y = 0.0
    cells = []
    while len(cells) < 500:
        heading = 0.9 * math.sin(len(cells) / 12.0) + 0.7
        x += math.cos(heading) / 3
        y += math.sin(heading) / 3
        cell = (int(round(y)), int(round(x)))
        if not cells or cell != cells[-1]:
            cells.append(cell)
    rows, cols = np.array(cells)[:, 0] + 10, np.array(cells)[:, 1] + 10
    end = (rows[-1] + 1, cols[-1] + 1)
    network = line_graph(1, 2)
    entries = [{"row": int(r), "col": int(c)} for r, c in zip(rows, cols)]
    grouped = {1: entries, 2: [{"row": int(end[0]), "col": int(end[1])}]}
    cells_by_reach = {1: (rows, cols), 2: (np.array([end[0]]), np.array([end[1]]))}
    fractions = np.linspace(0.0, 1.0, 500)
    baseline = 120.0 - 20.0 * fractions
    observations = baseline + rng.normal(0.0, 1.0, 500)
    thalwegs = observations - 2.0

    new_order = _best_seconds(lambda: order_reach(network, 1, rows, cols, 10.0, 10.0, cells_by_reach, observations))
    old_order = _best_seconds(lambda: legacy._order_reach_stream_cells_from_network(
        network, 1, entries, grouped, np.arange(500), observations, 10.0, 10.0))
    new_surface = _best_seconds(lambda: reach_bank_surface(observations, baseline, fractions, 5000.0, 100.0,
                                                           thalwegs=thalwegs))
    old_surface = _best_seconds(lambda: legacy._anchor_interpolated_bank_surface_to_cell_observations(
        observations, baseline, fractions, 5000.0, 100.0, MIN_GRADE, None, thalwegs))

    assert new_order < old_order
    assert new_surface < old_surface


def test_the_network_pass_is_faster_than_legacy_s() -> None:
    rng = np.random.default_rng(4)
    network = random_network(rng, 300)
    minima = {reach: 100.0 + float(rng.normal(0.0, 5.0)) for reach in network}
    maxima = {reach: value + 2.0 for reach, value in minima.items()}

    new = _best_seconds(lambda: network_outlet_elevations(network, minima, maxima))
    old = _best_seconds(lambda: legacy._estimate_network_smoothed_reach_min_bank_elevations(network, minima, maxima))

    assert new < old
