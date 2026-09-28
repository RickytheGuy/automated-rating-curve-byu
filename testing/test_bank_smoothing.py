from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np
import pytest

from arc import Automated_Rating_Curve_Generator as legacy
from arc.bathymetry import Banks, banks_for_width, single_cell_banks
from arc.bathymetry.smoothing import (BANK_HEIGHT_QUANTILE, BANK_HEIGHT_WINDOW, METHODS, MIN_GRADE, ReachSections,
                                      ReachWidths, falling_fit, filter_reach_widths, network_outlet_elevations,
                                      order_reach, reach_bank_levels, reach_bank_observations, reach_bank_surface,
                                      reach_baseline, reach_network, smooth_bank_elevations)
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


def test_a_median_between_one_and_two_spacings_is_rebuilt_as_it_is() -> None:
    """The median is 10 m, on a cross section with 8 m spacing. Legacy's bank indices couldn't make a channel that
    wide, and widened it a cell at a time."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 20.0])))
    wide = flat_section(spacing=8.0)
    sections.append(wide)
    banks.append(banks_for_width(wide, 40.0))

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths == ReachWidths(10.0, 10.0, 20.0)
    assert (filtered[4].left, filtered[4].right) == pytest.approx((5.0, 5.0))
    assert filtered[4].valid and filtered[4].method == "target_width"


def test_a_median_wider_than_a_cross_section_holds_is_as_wide_as_it_can_be() -> None:
    """With walls a metre and two metres either side of the stream cell, a 10 m median makes a channel 3 m wide."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 12.0])))
    walled = flat_section()
    walled.elevations[:49] = walled.elevations[53:] = 9999.0
    sections.append(walled)
    banks.append(Banks("width_to_depth_ratio", 1.0, 1.0, BED, BED, False, True))

    filtered, _ = filter_reach_widths(sections, banks)

    assert (filtered[4].left, filtered[4].right) == pytest.approx((1.0, 2.0))
    assert filtered[4].valid


def test_a_channel_that_can_t_be_rebuilt_is_left_alone() -> None:
    """With no ordinate on the raster on one side of the stream cell, no channel there is valid."""
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
    assert filtered[4].valid and filtered[4].top_width == pytest.approx(12.0)  # one and a half of its spacings


def test_a_reach_without_valid_banks_is_left_alone() -> None:
    sections = [flat_section(), flat_section()]

    assert filter_reach_widths(sections, [NO_BANKS, NO_BANKS]) == ([NO_BANKS, NO_BANKS], None)


def test_a_single_cell_channel_is_one_spacing_wide() -> None:
    """The stream cell's own width, as legacy counted it, so a reach of them has a median of one spacing."""
    sections = [flat_section(spacing=10.0) for _ in range(4)]
    banks = [single_cell_banks(xs) for xs in sections]

    assert banks[0].top_width == pytest.approx(10.0)
    assert filter_reach_widths(sections, banks)[1] == ReachWidths(10.0, 10.0, 10.0)


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


def test_a_reach_is_turned_by_the_observations_nearest_its_ends() -> None:
    """Its high end's last dozen cross sections have no observation (no valid banks there), which legacy's ten at
    each end didn't allow for: it left the reach running uphill. The ten observations nearest each end still say
    which end is higher."""
    # Given from the high end, so the stations start at the low end and the reach has to be turned round
    rows, cols = np.zeros(40, dtype=np.int64), np.arange(39, -1, -1)
    observations = 95.0 + 10.0 * cols / 39.0
    observations[cols >= 28] = np.nan
    network = line_graph(1, 2)
    cells = {1: (rows, cols)}

    order, stations = order_reach(network, 1, rows, cols, 10.0, 10.0, cells, observations)

    assert order.tolist() == list(range(40))  # from column 39 down to column 0
    np.testing.assert_allclose(stations, 10.0 * np.arange(40))
    projection = np.argsort(cols, kind="stable")
    legacy_order, _ = legacy._order_reach_stream_cells_from_network(
        network, 1, [{"row": int(r), "col": int(c)} for r, c in zip(rows, cols)], {}, projection, observations,
        10.0, 10.0)
    assert cols[legacy_order].tolist() == list(range(40))  # legacy's, uphill


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


# --- The bank level: the water surface plus the banks' height -----------------------------------------------------


def reference_falling_fit(values) -> np.ndarray:
    """Pool adjacent violators the slow way: merge the first rising pair of runs at their mean, and start again."""
    runs = [[float(v)] for v in values]
    merged = True
    while merged:
        merged = False
        for k in range(len(runs) - 1):
            if np.mean(runs[k]) < np.mean(runs[k + 1]):
                runs[k:k + 2] = [runs[k] + runs[k + 1]]
                merged = True
                break
    return np.concatenate([[np.mean(run)] * len(run) for run in runs])


def test_the_falling_fit_pools_each_rise_at_its_mean() -> None:
    np.testing.assert_allclose(falling_fit([5.0, 3.0, 4.0, 2.0, 6.0, 1.0]), [5.0, 3.75, 3.75, 3.75, 3.75, 1.0])
    np.testing.assert_array_equal(falling_fit([3.0, 2.0, 2.0, 1.0]), [3.0, 2.0, 2.0, 1.0])


def test_the_falling_fit_is_the_least_squares_sequence_that_never_rises() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        values = rng.normal(0.0, 1.0, n) - rng.uniform(0.0, 0.2) * np.arange(n)

        fitted = falling_fit(values)

        np.testing.assert_allclose(fitted, reference_falling_fit(values), rtol=0.0, atol=1e-12)
        assert np.all(np.diff(fitted) <= 1e-12)


def test_the_falling_fit_leaves_nan_out() -> None:
    np.testing.assert_allclose(falling_fit([3.0, math.nan, 5.0, 1.0]), [4.0, math.nan, 4.0, 1.0])


def test_the_bank_height_is_the_10th_percentile_within_500_m() -> None:
    assert (BANK_HEIGHT_QUANTILE, BANK_HEIGHT_WINDOW) == (0.1, 500.0)
    assert METHODS[0] == "water_plus_height"


def test_a_steady_reach_s_bank_elevation_is_its_water_plus_its_banks_height() -> None:
    stations = 30.0 * np.arange(100)
    thalwegs = 100.0 - 0.001 * stations

    bank = reach_bank_levels(thalwegs, thalwegs + 1.5, stations, math.nan)

    np.testing.assert_allclose(bank, thalwegs + 1.5, atol=1e-9)


def test_the_height_comes_from_the_banks_nearby() -> None:
    """Banks 1 m above the water for the first kilometre and 2 m for the second: at each end, only its own are
    within 500 m."""
    stations = 10.0 * np.arange(201)
    thalwegs = 103.0 - 0.0015 * stations
    heights = np.where(stations < 1000.0, 1.0, 2.0)

    bank = reach_bank_levels(thalwegs, thalwegs + heights, stations, math.nan)

    assert bank[0] == pytest.approx(thalwegs[0] + 1.0)
    assert bank[-1] == pytest.approx(thalwegs[-1] + 2.0)


def test_the_height_is_a_low_percentile_of_the_banks() -> None:
    """Heights of 1 to 10 m in turn, all within 500 m of each other: their 10th percentile."""
    stations = 10.0 * np.arange(50)
    thalwegs = 100.0 - 0.01 * stations
    heights = 1.0 + np.arange(50) % 10

    bank = reach_bank_levels(thalwegs, thalwegs + heights, stations, math.nan)

    np.testing.assert_allclose(bank - thalwegs, np.quantile(heights, 0.1), atol=1e-9)


def test_a_reach_that_runs_mild_then_steep_keeps_its_banks_above_the_water() -> None:
    """Legacy's straight line to the outlet passed under the whole mild stretch (BS2)."""
    stations = 30.0 * np.arange(300)
    thalwegs = np.where(stations < 6000.0, 165.0 - 0.0001 * stations, 164.4 - 8.0 * (stations - 6000.0) / 3000.0)

    bank = reach_bank_levels(thalwegs, thalwegs + 1.5, stations, math.nan)

    np.testing.assert_allclose(bank, thalwegs + 1.5, atol=1e-9)


def test_the_bank_elevation_never_rises_downstream_or_goes_below_the_stream_cell() -> None:
    """A stream cell 2 m above its neighbours, as a bridge in the DEM: the water surface passes under it, and the
    bank elevation stops at the stream cell."""
    stations = 10.0 * np.arange(50)
    thalwegs = 100.0 - 0.001 * stations
    thalwegs[25] += 2.0

    bank = reach_bank_levels(thalwegs, thalwegs + 0.5, stations, math.nan)

    assert np.all(bank >= thalwegs)
    assert bank[25] == thalwegs[25]
    assert np.all(np.diff(np.delete(bank, 25)) <= 1e-12)


def test_observations_at_or_below_the_stream_cell_are_left_out() -> None:
    stations = 10.0 * np.arange(40)
    thalwegs = 100.0 - 0.01 * stations

    bank = reach_bank_levels(thalwegs, thalwegs + np.where(np.arange(40) % 2, -0.5, 1.0), stations, math.nan)

    np.testing.assert_allclose(bank, thalwegs + 1.0, atol=1e-9)


def test_a_reach_without_observations_takes_the_fallback_height() -> None:
    stations = 10.0 * np.arange(40)
    thalwegs = 100.0 - 0.01 * stations

    np.testing.assert_allclose(reach_bank_levels(thalwegs, np.full(40, math.nan), stations, 0.8), thalwegs + 0.8)
    assert np.isnan(reach_bank_levels(thalwegs, np.full(40, math.nan), stations, math.nan)).all()


def made_up_reach(thalwegs, observations, cell=30.0, phantom=False):
    """A reach of cross sections one cell apart along a row, flowing into a reach of one cross section without an
    observation of its own (as below the Du Page reach), and with phantom an inflow with no cross sections."""
    count = len(thalwegs)
    network = line_graph(1, 2, lengths=[count * cell, cell])
    if phantom:
        network.add_node(0, length=2000.0)
        network.add_edge(0, 1)

    def section(z):
        elevations = np.full(9, z + 5.0)
        elevations[4] = z
        return XSection(elevations, np.full(9, 0.035), cell)

    def banks(z):
        return Banks("test", 30.0, 30.0, z, z, False, True)
    last = float(thalwegs[-1]) - 0.01
    reaches = {1: ReachSections(np.zeros(count, np.int64), np.arange(count), [section(z) for z in thalwegs],
                                [banks(z) for z in observations]),
               2: ReachSections(np.zeros(1, np.int64), np.array([count]), [section(last)], [banks(last)])}
    return network, reaches


def test_an_inflow_without_cross_sections_doesn_t_cap_the_reach() -> None:
    """Legacy's network gave the inflow the reach's own lowest observation plus the minimum grade, and that held the
    reach's bank elevation down to about its outlet's all the way up (BS2)."""
    thalwegs = 100.0 - 0.001 * 30.0 * np.arange(300)
    network, reaches = made_up_reach(thalwegs, thalwegs + 1.5, phantom=True)

    new = smooth_bank_elevations(network, reaches, 30.0, 30.0)[1]
    old = smooth_bank_elevations(network, reaches, 30.0, 30.0, method="legacy")[1]

    np.testing.assert_allclose(new.bank_elevations, thalwegs + 1.5, atol=1e-9)
    assert old.bank_elevations[0] < thalwegs[0] - 5.0
    assert not new.anchors.any()


def test_the_site_s_height_fills_in_a_reach_without_observations() -> None:
    """Reach 1's banks are 1.2 m above its water where they're above it at all; reach 3, flat, has no banks."""
    thalwegs = 100.0 - 0.001 * 30.0 * np.arange(60)
    network, reaches = made_up_reach(thalwegs, np.r_[thalwegs[:30] + 1.2, thalwegs[30:] - 1.0])
    flat = XSection(np.r_[np.full(4, 55.0), 50.0, np.full(4, 55.0)], np.full(9, 0.035), 30.0)
    reaches[3] = ReachSections(np.full(10, 5, np.int64), np.arange(10), [flat] * 10, [NO_BANKS] * 10)
    network.add_node(3, length=300.0)

    smoothed = smooth_bank_elevations(network, reaches, 30.0, 30.0)

    np.testing.assert_allclose(smoothed[3].bank_elevations, 51.2, atol=1e-9)


def test_an_unknown_method_raises() -> None:
    network, reaches = made_up_reach(np.full(5, 100.0), np.full(5, 101.0))

    with pytest.raises(ValueError, match="method must be one of"):
        smooth_bank_elevations(network, reaches, 30.0, 30.0, method="lowest")


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


def test_legacy_s_bank_elevations_fall_along_the_network() -> None:
    network, reaches = small_network()

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0, method="legacy")

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


def test_without_an_observation_anywhere_no_reach_gets_a_bank_elevation() -> None:
    """As legacy, whose network estimate came back empty, so that it dropped every reach. What to do with that is up
    to the caller: the pipeline drops them only with bank elevations, which is where they're needed."""
    network, reaches = small_network()
    for reach, sections in reaches.items():
        reaches[reach] = sections._replace(banks=[NO_BANKS] * len(sections.banks))

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0)

    assert all(np.isnan(result.bank_elevations).all() for result in smoothed.values())


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
