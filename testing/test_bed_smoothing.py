from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np
import pytest

from arc import Automated_Rating_Curve_Generator as legacy
from arc.bathymetry import Banks
from arc.bathymetry.bed_smoothing import fill_reach_depths, smooth_channel_depths, smooth_reach_bed
from arc.bathymetry.smoothing import ReachSections, SmoothedReach, order_reach
from arc.xsection.xsection import XSection

CELL = 10.0
NO_BANKS = Banks("none", math.nan, math.nan, math.nan, math.nan, False, False)


def line_graph(*reaches) -> nx.DiGraph:
    network = nx.DiGraph()
    for reach in reaches:
        network.add_node(reach, length=100.0)
    network.add_edges_from(zip(reaches[:-1], reaches[1:]))
    return network


def network_inputs(network, cells, banks, thalwegs=None):
    """What smooth_channel_depths takes, for cross sections at cells (reach, row, col) in that order, each reach
    ordered as smooth_bank_elevations orders it, with these smoothed bank elevations and stream cell elevations."""
    thalwegs = np.zeros(len(cells)) if thalwegs is None else np.asarray(thalwegs, dtype=float)
    indices: dict[int, list[int]] = {}
    for k, (reach, _, _) in enumerate(cells):
        indices.setdefault(reach, []).append(k)
    positions = {reach: (np.array([cells[k][1] for k in ks]), np.array([cells[k][2] for k in ks]))
                 for reach, ks in indices.items()}
    reaches, smoothed = {}, {}
    for reach, ks in indices.items():
        rows, cols = positions[reach]
        sections = [XSection(np.array([thalwegs[k] + 5.0, thalwegs[k], thalwegs[k] + 5.0]), np.full(3, 0.035), CELL)
                    for k in ks]
        reaches[reach] = ReachSections(rows, cols, sections, [NO_BANKS] * len(ks))
        bank_elevations = np.array([banks[k] for k in ks], dtype=float)
        order, stations = order_reach(network, reach, rows, cols, CELL, CELL, positions, bank_elevations)
        smoothed[reach] = SmoothedReach([NO_BANKS] * len(ks), bank_elevations, bank_elevations,
                                        np.zeros(len(ks), dtype=bool), order, stations, None)
    return reaches, smoothed, indices


def smoothed_depths(network, cells, banks, depths, thalwegs=None, **options):
    """smooth_channel_depths' depths and beds, back in the order of cells."""
    reaches, smoothed, indices = network_inputs(network, cells, banks, thalwegs)
    options.setdefault("use_banks", True)
    result = smooth_channel_depths(network, reaches, smoothed, {r: [depths[k] for k in ks] for r, ks in indices.items()},
                                   CELL, CELL, **options)
    out_depths, out_beds = np.empty(len(cells)), np.empty(len(cells))
    for reach, ks in indices.items():
        out_depths[ks], out_beds[ks] = result[reach].depths, result[reach].beds
    return out_depths, out_beds


def run_legacy(monkeypatch, network, cells, banks, depths, steps=("fill", "beds")):
    """Legacy's depth fill and bed smoothing on cross sections at cells, with these bank elevations and depths."""
    monkeypatch.setattr(legacy, "_CELL_COMIDS", np.array([c[0] for c in cells], dtype=np.int64))
    monkeypatch.setattr(legacy, "_CELL_SOURCE_STREAM_IDS", None)
    monkeypatch.setattr(legacy, "_CELL_ROWS", np.array([c[1] for c in cells], dtype=np.int64))
    monkeypatch.setattr(legacy, "_CELL_COLS", np.array([c[2] for c in cells], dtype=np.int64))
    monkeypatch.setattr(legacy, "_build_reach_network_graph", lambda *args: (network, {}))
    records = [{"bank_search_result": {"smoothed_bank_elevation": b, "bathymetry_depth": d}}
               for b, d in zip(banks, depths)]
    if "fill" in steps:
        legacy._smooth_reach_bathymetry_depths(records, {"dx": CELL, "dy": CELL})
    if "beds" in steps:
        legacy._smooth_reach_excavated_bed_elevations(records, {"dx": CELL, "dy": CELL})
    return np.array([r["bank_search_result"]["bathymetry_depth"] for r in records]), \
        np.array([r["bank_search_result"].get("smoothed_bed_elevation", math.nan) for r in records])


def along_row(reach_lengths, row=0):
    """Reaches one after another along a row, one cross section per cell."""
    cells, col = [], 0
    for reach, length in reach_lengths:
        cells += [(reach, row, col + k) for k in range(length)]
        col += length
    return cells


# --- Filling in depths --------------------------------------------------------------------------------------------


def test_depths_that_aren_t_above_0_and_at_most_25_m_take_the_one_before() -> None:
    depths = [1.0, 0.0, -1.0, 2.0, 30.0, 25.5, 25.0, 3.0]

    filled = fill_reach_depths(line_graph(1), {1: depths}, {1: np.arange(8)})

    np.testing.assert_array_equal(filled[1], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 25.0, 3.0])


def test_depths_are_filled_in_down_the_order_given() -> None:
    filled = fill_reach_depths(line_graph(1), {1: [0.0, 1.0, 0.0, 2.0]}, {1: np.array([3, 2, 1, 0])})

    np.testing.assert_array_equal(filled[1], [1.0, 1.0, 2.0, 2.0])


def test_a_reach_s_first_depths_take_the_shallowest_last_depth_flowing_in() -> None:
    network = nx.DiGraph([(1, 3), (2, 3)])

    filled = fill_reach_depths(network, {1: [1.0, 1.5], 2: [0.8], 3: [0.0, -1.0, 2.0]},
                               {1: np.arange(2), 2: np.arange(1), 3: np.arange(3)})

    np.testing.assert_array_equal(filled[3], [0.8, 0.8, 2.0])


def test_with_no_depth_before_it_a_depth_becomes_half_a_metre() -> None:
    filled = fill_reach_depths(line_graph(1), {1: [0.0, 30.0, 1.2]}, {1: np.arange(3)})

    np.testing.assert_array_equal(filled[1], [0.5, 0.5, 1.2])


def test_a_nan_depth_is_filled_in_and_so_is_the_next_with_the_depth_before_them(monkeypatch) -> None:
    """Legacy kept the NaN, and gave the next bad depth 0.5 m."""
    depths = [1.2, math.nan, -1.0, 1.3]
    cells = along_row([(1, 4), (2, 1)])

    filled = fill_reach_depths(line_graph(1, 2), {1: depths, 2: [1.0]}, {1: np.arange(4), 2: np.arange(1)})
    old, _ = run_legacy(monkeypatch, line_graph(1, 2), cells, [100.0] * 5, depths + [1.0], steps=("fill",))

    np.testing.assert_array_equal(filled[1], [1.2, 1.2, 1.2, 1.3])
    np.testing.assert_array_equal(old[:4], [1.2, math.nan, 0.5, 1.3])


def random_network(rng, size) -> nx.DiGraph:
    network = nx.DiGraph()
    network.add_nodes_from(range(1, size + 1), length=100.0)
    network.add_edges_from((reach, int(rng.integers(1, reach))) for reach in range(2, size + 1))
    return network


def test_the_fill_matches_legacy_s_in_legacy_s_order(monkeypatch) -> None:
    """Random networks of reaches along rows, with depths of all kinds but NaN, filled in the order legacy put them
    in."""
    rng = np.random.default_rng(7)
    for _ in range(100):
        network = random_network(rng, int(rng.integers(1, 8)))
        cells = [(reach, 3 * reach, k) for reach in network for k in range(int(rng.integers(1, 12)))]
        depths = [float(rng.choice([-1.0, 0.0, 30.0, 25.0, rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0)]))
                  for _ in cells]
        expected, _ = run_legacy(monkeypatch, network, cells, [100.0] * len(cells), depths, steps=("fill",))

        entries: dict[int, list[dict]] = {}
        for k, (reach, row, col) in enumerate(cells):
            entries.setdefault(reach, []).append({"entry_index": k, "row": row, "col": col})
        orders = {reach: legacy._order_reach_stream_cells_from_network(network, reach, e, entries, np.arange(len(e)),
                                                                        np.zeros(len(e)), CELL, CELL)[0]
                  for reach, e in entries.items()}
        filled = fill_reach_depths(network, {r: [depths[x["entry_index"]] for x in e] for r, e in entries.items()},
                                   orders)

        got = np.empty(len(cells))
        for reach, e in entries.items():
            got[[x["entry_index"] for x in e]] = filled[reach]
        np.testing.assert_array_equal(got, expected)


# --- Smoothing a reach's bed --------------------------------------------------------------------------------------


def test_the_bed_is_the_median_of_the_five_cross_sections_around_it() -> None:
    beds = np.array([10.0, 11.0, 9.0, 14.0, 8.0, 12.0])

    smoothed = smooth_reach_bed(beds, CELL * np.arange(6), max_bed_grade=None)

    np.testing.assert_array_equal(smoothed, [10.0, 10.5, 10.0, 11.0, 10.5, 12.0])


def test_the_bed_cap_holds_the_bed_to_1_cm_a_metre() -> None:
    """A 1 m channel down a reach falling 2%: capped, its bed falls 1 m per 100 m, so the channel is gone after 120 m
    (1.2 m deep at first, from the median at the reach's end)."""
    banks = 100.0 - 0.02 * CELL * np.arange(30)

    capped = smooth_reach_bed(banks - 1.0, CELL * np.arange(30))
    uncapped = smooth_reach_bed(banks - 1.0, CELL * np.arange(30), max_bed_grade=None)

    np.testing.assert_allclose((banks - capped)[[0, 5, 10, 12]], [1.2, 0.7, 0.2, 0.0], atol=1e-9)
    assert np.all(banks[12:] - capped[12:] <= 1e-9)
    np.testing.assert_allclose((banks - uncapped)[2:-2], 1.0, atol=1e-9)
    np.testing.assert_allclose((banks - uncapped)[[0, -1]], [1.2, 0.8], atol=1e-9)


def test_the_cap_allows_at_least_a_tenth_of_a_metre_between_cross_sections() -> None:
    smoothed = smooth_reach_bed([100.0, 90.0, 90.0], [0.0, 0.0, 0.0], window=1)

    np.testing.assert_allclose(smoothed, [100.0, 99.999, 99.998])


def test_the_first_bed_is_held_to_the_lowest_bed_flowing_in() -> None:
    smoothed = smooth_reach_bed([95.0, 95.0], [0.0, 10.0], window=1, inflow_bed=98.0, inflow_distance=10.0)

    np.testing.assert_allclose(smoothed, [97.9, 97.8])


def test_a_missing_bed_is_left_out_of_the_median_and_the_cap() -> None:
    """The medians are 10.5, 10, none, 11 and 11, and the cap lets them change 0.1 m per 10 m, across the gap too."""
    smoothed = smooth_reach_bed([10.0, 11.0, math.nan, 9.0, 13.0], [0.0, 10.0, 20.0, 30.0, 40.0])

    np.testing.assert_allclose(smoothed, [10.5, 10.4, math.nan, 10.6, 10.7])


def test_the_bed_matches_legacy_s(monkeypatch) -> None:
    """One reach, flowing in from a reach of one cross section, its cells a step or two apart along a row, with legacy's
    1 mm hold on its first bed (inflow_distance 0)."""
    rng = np.random.default_rng(3)
    network = nx.DiGraph([(2, 1), (1, 3)])
    for _ in range(300):
        n = int(rng.integers(1, 40))
        cols = 5 + np.concatenate([[0], np.cumsum(rng.integers(1, 3, n - 1))])
        cells = [(2, 0, 3)] + [(1, 0, int(c)) for c in cols] + [(3, 0, int(cols[-1]) + 2)]
        grade = float(rng.choice([0.0005, 0.005, 0.02, 0.05]))
        banks = [100.0 - grade * CELL * c + rng.normal(0.0, 0.05) for _, _, c in cells]
        depths = [float(rng.uniform(0.2, 3.0)) for _ in cells]
        _, expected = run_legacy(monkeypatch, network, cells, banks, depths, steps=("beds",))

        smoothed = smooth_reach_bed(np.subtract(banks, depths)[1:n + 1], CELL * (cols - cols[0]),
                                    inflow_bed=banks[0] - depths[0], inflow_distance=0.0)

        np.testing.assert_array_equal(smoothed, expected[1:n + 1])


def test_a_bed_cap_or_median_window_that_can_t_be_raises() -> None:
    with pytest.raises(ValueError):
        smooth_reach_bed([1.0, 2.0], [0.0, 1.0], max_bed_grade=-0.01)
    with pytest.raises(ValueError):
        smooth_reach_bed([1.0, 2.0], [0.0, 1.0], window=0)
    with pytest.raises(ValueError):
        smooth_reach_bed([1.0, 2.0], [0.0])


# --- Along the network --------------------------------------------------------------------------------------------


def test_the_depth_is_the_bank_elevation_s_height_above_the_smoothed_bed() -> None:
    """Beds of 99, 98.9, 96.8, 98.7 and 98.6 m, whose medians are 98.9, 98.8, 98.7, 98.65 and 98.6."""
    cells = along_row([(1, 5), (2, 1)])
    banks = [100.0, 99.9, 99.8, 99.7, 99.6, 99.5]

    depths, beds = smoothed_depths(line_graph(1, 2), cells, banks, [1.0, 1.0, 3.0, 1.0, 1.0, 1.0])

    np.testing.assert_allclose(beds[:5], [98.9, 98.8, 98.7, 98.65, 98.6])
    np.testing.assert_allclose(depths[:5], [1.1, 1.1, 1.1, 1.05, 1.0])


def test_where_the_bed_is_above_the_bank_elevation_the_depth_is_0() -> None:
    cells = along_row([(1, 3), (2, 1)])

    depths, _ = smoothed_depths(line_graph(1, 2), cells, [100.0, 100.0, 90.0, 90.0], [1.0] * 4)

    assert depths[2] == 0.0


def test_without_bank_elevations_the_bed_is_the_stream_cell_s_elevation_less_the_depth(monkeypatch) -> None:
    """Stream cells 0.3 m up and down about smooth banks. Legacy smoothed the banks less the depth and carved below the
    stream cells, which left the carved bed as rough as they are."""
    rng = np.random.default_rng(0)
    cells = along_row([(1, 50), (2, 1)])
    banks = 100.0 - 0.005 * CELL * np.arange(51)
    thalwegs = banks - 1.0 + rng.normal(0.0, 0.3, 51)

    depths, beds = smoothed_depths(line_graph(1, 2), cells, banks, [0.8] * 51, thalwegs=thalwegs, use_banks=False)
    old, _ = run_legacy(monkeypatch, line_graph(1, 2), cells, banks, [0.8] * 51)

    known = depths > 0.0
    np.testing.assert_allclose((thalwegs - depths)[known], beds[known])
    assert np.std(np.diff(thalwegs[:50] - depths[:50])) < 0.1
    assert np.std(np.diff(thalwegs[:50] - old[:50])) > 0.3


def test_a_reach_without_neighbours_is_smoothed_along_its_cells_in_metres(monkeypatch) -> None:
    """A reach by itself falling 0.5%, a 1 m channel. Legacy took its cross sections a metre apart, so the cap let its
    bed fall 1 cm per cross section."""
    cells = along_row([(1, 50)])
    banks = 100.0 - 0.005 * CELL * np.arange(50)

    depths, _ = smoothed_depths(line_graph(1), cells, banks, [1.0] * 50)
    old, _ = run_legacy(monkeypatch, line_graph(1), cells, banks, [1.0] * 50)

    np.testing.assert_allclose(depths[[0, 25, 49]], [1.05, 1.0, 0.95], atol=1e-9)
    np.testing.assert_allclose(old[[0, 25, 49]], [1.05, 0.05, 0.0], atol=1e-9)


def test_where_the_banks_drop_3_m_the_capped_reach_below_starts_without_a_channel(monkeypatch) -> None:
    """Reach 1 falls into reach 2, whose banks are 3 m lower. Capped, reach 2's first bed is held within 0.1 m of
    reach 1's last, 10 m upstream (legacy: within 1 mm), and falls 1 cm a metre from there, so the channel starts
    220 m down (legacy: 230 m). Uncapped, it is 1 m deep all along."""
    network = line_graph(1, 2, 3)
    cells = along_row([(1, 20), (2, 40)])
    banks = 100.0 - 0.001 * CELL * np.arange(60) - np.where(np.arange(60) >= 20, 3.0, 0.0)

    capped, beds = smoothed_depths(network, cells, banks, [1.0] * 60)
    uncapped, _ = smoothed_depths(network, cells, banks, [1.0] * 60, max_bed_grade=None)
    old, old_beds = run_legacy(monkeypatch, network, cells, banks, [1.0] * 60)

    assert beds[20] == pytest.approx(beds[19] - 0.1)
    assert old_beds[20] == pytest.approx(old_beds[19] - 0.001)
    assert np.flatnonzero(capped[20:] > 0.0)[0] == 22
    assert np.flatnonzero(old[20:] > 0.0)[0] == 23
    np.testing.assert_allclose(uncapped[22:58], 1.0, atol=1e-9)


def test_the_first_bed_is_held_to_the_lower_of_two_reaches_flowing_in() -> None:
    """Reaches 1 and 2, with beds at 98 and 97 m, join to make reach 3, whose own bed is at 95 m. Its first cross
    section is a cell diagonal from each of their last."""
    network = nx.DiGraph([(1, 3), (2, 3)])
    cells = [(1, 0, k) for k in range(5)] + [(2, 2, k) for k in range(5)] + [(3, 1, 5 + k) for k in range(5)]
    banks = [100.0] * 10 + [97.0] * 5
    depths = [2.0] * 5 + [3.0] * 5 + [2.0] * 5

    _, beds = smoothed_depths(network, cells, banks, depths)

    assert beds[10] == pytest.approx(97.0 - 0.01 * math.hypot(CELL, CELL))


def test_a_missing_bank_elevation_takes_out_only_its_own_cross_section(monkeypatch) -> None:
    """Legacy's running median and cap spread its NaN to 49 of the reach's 50 cross sections."""
    cells = along_row([(1, 50), (2, 1)])
    banks = 100.0 - 0.005 * CELL * np.arange(51)
    banks[20] = math.nan

    depths, beds = smoothed_depths(line_graph(1, 2), cells, banks, [1.0] * 51)
    old, _ = run_legacy(monkeypatch, line_graph(1, 2), cells, banks, [1.0] * 51)

    assert np.isnan(beds[20]) and depths[20] == 1.0
    assert not np.isnan(beds[:20]).any() and not np.isnan(beds[21:]).any()
    assert np.isnan(old[:50]).sum() == 49


def test_the_headwaters_match_legacy(monkeypatch) -> None:
    """Random reaches along rows, each flowing into its own downstream reach, which has one cross section beside its
    end: nothing flows into them, so there's nothing to hold their first beds to."""
    rng = np.random.default_rng(1)
    for _ in range(30):
        network, cells = nx.DiGraph(), []
        for reach in range(1, int(rng.integers(2, 6))):
            network.add_edge(reach, 100 + reach)
            n = int(rng.integers(1, 30))
            cells += [(reach, 3 * reach, k) for k in range(n)] + [(100 + reach, 3 * reach, n)]
        grades = {reach: float(rng.choice([0.001, 0.005, 0.02])) for reach, _, _ in cells}
        banks = [100.0 - grades[reach] * CELL * col + rng.normal(0.0, 0.05) for reach, _, col in cells]
        depths = [float(rng.choice([rng.uniform(0.2, 3.0), rng.uniform(0.2, 3.0), 0.0, 30.0])) for _ in cells]

        got, _ = smoothed_depths(network, cells, banks, depths)
        expected, _ = run_legacy(monkeypatch, network, cells, banks, depths)

        headwaters = [k for k, (reach, _, _) in enumerate(cells) if reach < 100]
        np.testing.assert_array_equal(got[headwaters], expected[headwaters])


def test_depths_come_back_in_the_order_the_cross_sections_were_given() -> None:
    rng = np.random.default_rng(2)
    cells = along_row([(1, 30), (2, 1)])
    banks = list(100.0 - 0.01 * CELL * np.arange(31) + rng.normal(0.0, 0.1, 31))
    depths = list(rng.uniform(0.5, 2.0, 31))
    shuffle = rng.permutation(30)

    expected, _ = smoothed_depths(line_graph(1, 2), cells, banks, depths)
    got, _ = smoothed_depths(line_graph(1, 2), [cells[k] for k in shuffle] + cells[30:],
                             [banks[k] for k in shuffle] + banks[30:], [depths[k] for k in shuffle] + depths[30:])

    np.testing.assert_array_equal(got[:30], expected[shuffle])


def test_reaches_that_can_t_be_smoothed_raise() -> None:
    cells = along_row([(1, 3)])
    reaches, smoothed, _ = network_inputs(line_graph(1), cells, [100.0] * 3)

    with pytest.raises(ValueError, match="no cross sections"):
        smooth_channel_depths(line_graph(1), reaches, smoothed, {2: [1.0]}, CELL, CELL, use_banks=True)
    with pytest.raises(ValueError, match="3 cross sections"):
        smooth_channel_depths(line_graph(1), reaches, smoothed, {1: [1.0, 1.0]}, CELL, CELL, use_banks=True)
    with pytest.raises(ValueError, match="loop"):
        smooth_channel_depths(nx.DiGraph([(1, 1)]), reaches, smoothed, {1: [1.0] * 3}, CELL, CELL, use_banks=True)
    with pytest.raises(ValueError, match="aren't in the stream network"):
        smooth_channel_depths(line_graph(5), reaches, smoothed, {1: [1.0] * 3}, CELL, CELL, use_banks=True)


# --- Speed --------------------------------------------------------------------------------------------------------


def test_filling_and_smoothing_is_faster_than_legacy_s(monkeypatch) -> None:
    """Twenty reaches of 200 cross sections flowing into each other, counting the ordering for the new code too, which
    the bank smoothing has already done."""
    rng = np.random.default_rng(4)
    network = line_graph(*range(1, 21))
    cells = along_row([(reach, 200) for reach in range(1, 21)])
    banks = list(100.0 - 0.002 * CELL * np.arange(len(cells)) + rng.normal(0.0, 0.05, len(cells)))
    depths = list(rng.choice([0.0, 1.0, 1.5, 2.0], len(cells)))
    reaches, smoothed, indices = network_inputs(network, cells, banks)
    reach_depths = {r: [depths[k] for k in ks] for r, ks in indices.items()}
    positions = {r: (reaches[r].rows, reaches[r].cols) for r in reaches}

    def new():
        for r in reaches:
            order_reach(network, r, reaches[r].rows, reaches[r].cols, CELL, CELL, positions,
                        smoothed[r].bank_elevations)
        smooth_channel_depths(network, reaches, smoothed, reach_depths, CELL, CELL, use_banks=True)

    start = time.perf_counter()
    run_legacy(monkeypatch, network, cells, banks, depths)
    old = time.perf_counter() - start
    new()
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        new()
        best = min(best, time.perf_counter() - start)

    assert best < old
