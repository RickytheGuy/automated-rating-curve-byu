from __future__ import annotations

import math
import time

import numpy as np
import pytest
from numba import njit

from arc import bathymetry, hydraulics
from arc.Automated_Rating_Curve_Generator import _fill_bathymetry_nan_cells
from arc.bathymetry import (Banks, bank_control_elevation, banks_at_elevation, banks_by_flat_water,
                            banks_by_land_cover, banks_by_width_to_depth_ratio, banks_for_width, bathymetry_depth,
                            burn_into_raster, carve_channel, channel_depth, fill_bathymetry_gaps, find_banks, in_bank,
                            ordinate_cells, power_law_geometry, sample_land_cover, set_bank_distances,
                            set_in_bank_roughness, single_cell_banks, trapezoid_depth)
from arc.bathymetry import banks as banks_module
from arc.bathymetry import channel as channel_module
from arc.cross_section import (CrossSection, _adjust_one_side_for_bathymetry, _find_bank_using_width_to_depth_ratio,
                               find_depth_of_bathymetry)
from arc.xsection.xsection import XSection

BED = 100.0
SLOPE = 0.001
N = 0.03
WALL = 9999.0
WATER = 80
NO_BANKS = Banks("none", math.nan, math.nan, math.nan, math.nan, False, False)


def make_section(elevations, spacing=1.0) -> XSection:
    elevations = np.asarray(elevations, dtype=np.float64)
    return XSection(elevations.copy(), np.full(elevations.size, 0.035), spacing)


def symmetric(side_profile, spacing=1.0) -> XSection:
    """A cross section with the same profile out from the stream cell on both sides."""
    side = np.asarray(side_profile, dtype=np.float64)
    return make_section(np.concatenate([side[:0:-1], side]), spacing)


def channel_and_floodplains() -> XSection:
    """A 20 m bed, banks rising 1:1 to 2 m 12 m out, then floodplains rising 1 cm per metre, at 1 m spacing."""
    distance = np.arange(0.0, 31.0)
    return symmetric(BED + np.clip(distance - 10.0, 0.0, 2.0) + 0.01 * np.maximum(distance - 12.0, 0.0))


def trapezoid_discharge(depth, bottom, top, slope=SLOPE, n=N):
    side = (top - bottom) / 2
    area = depth * (bottom + top) / 2
    perimeter = bottom + 2 * math.hypot(side, depth)
    return area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / n


def random_section(rng, size, spacing) -> XSection:
    """A random valley whose banks rise, flatten, and sometimes dip, rounded so that some ordinates tie."""
    center = size // 2
    left = np.cumsum(rng.normal(0.2, 0.4, center))
    right = np.cumsum(rng.normal(0.2, 0.4, size - 1 - center))
    elevations = BED + np.concatenate([left[::-1], [0.0], right])
    return make_section(np.round(elevations, int(rng.integers(2, 5))), spacing)


def legacy_cross_section(xs: XSection, *, use_banks: bool, land_cover=None, trapezoid_height=0.2) -> CrossSection:
    """Legacy ARC's CrossSection holding the same two half profiles, left as side 1, and a DEM row holding them."""
    size = xs.elevations.size
    center = size // 2
    profile1, profile2 = xs.elevations[center::-1].copy(), xs.elevations[center:].copy()
    params = {
        "d_x_section_distance": 4.0 * size, "b_FindBanksBasedOnLandCover": land_cover is not None,
        "i_lc_water_value": WATER, "d_bathymetry_trapzoid_height": trapezoid_height, "b_bathy_use_banks": use_banks,
        "d_degree_manipulation": 0.0, "d_degree_interval": 0.0, "i_boundary_number": 0, "nrows": 3,
        "ncols": size + 2, "s_output_bathymetry_path": None,
    }
    dem = np.zeros((3, size + 2))
    legacy = CrossSection(1.0, 1.0, dem, np.zeros((3, size + 2), dtype=np.uint8),
                          np.zeros((3, size + 2), dtype=np.int64), params)
    legacy.xs1_n, legacy.xs2_n = profile1.size, profile2.size
    legacy.d_ordinate_dist = float(xs.ordinate_distance)
    legacy.da_xs_profile1 = np.append(profile1, 99999.9)  # the legacy sampler's sentinel after the last ordinate
    legacy.da_xs_profile2 = np.append(profile2, 99999.9)
    legacy.ia_xc_row1_index_main = np.ones(profile1.size, dtype=np.int64)
    legacy.ia_xc_column1_index_main = center + 1 - np.arange(profile1.size)
    legacy.ia_xc_row2_index_main = np.ones(profile2.size, dtype=np.int64)
    legacy.ia_xc_column2_index_main = center + 1 + np.arange(profile2.size)
    dem[1, 1:size + 1] = xs.elevations
    if land_cover is not None:
        land_cover = np.asarray(land_cover)
        legacy.ia_lc_xs1 = np.append(land_cover[center::-1], 0).astype(np.uint8)
        legacy.ia_lc_xs2 = np.append(land_cover[center:], 0).astype(np.uint8)
    return legacy


# --- Banks -------------------------------------------------------------------------------------------------------


def test_flat_water_banks_are_where_the_ground_is_0p1_m_above_the_stream_cell() -> None:
    """Banks rising 4 cm per metre reach 0.1 m 2.5 m out."""
    banks = banks_by_flat_water(symmetric(BED + 0.04 * np.arange(0.0, 21.0)))

    assert banks.left == pytest.approx(2.5, rel=1e-12)
    assert banks.right == pytest.approx(2.5, rel=1e-12)
    assert banks.left_elevation == pytest.approx(BED + 0.1, rel=1e-12)
    assert banks.valid


def test_width_to_depth_banks_are_where_the_water_spills_out_of_the_channel() -> None:
    """The ratio of top width to depth falls while the channel fills and rises once the water reaches the
    floodplains, so the banks are at the bank tops, 12 m out. Legacy ARC found bank ordinates 12 and 12, but counted
    its top width as 12 + 12 - 1 = 23 m."""
    banks = banks_by_width_to_depth_ratio(channel_and_floodplains())

    assert (banks.left, banks.right) == pytest.approx((12.0, 12.0), rel=1e-12)
    assert banks.top_width == pytest.approx(24.0, rel=1e-12)
    assert (banks.left_elevation, banks.right_elevation) == pytest.approx((BED + 2.0, BED + 2.0), rel=1e-12)
    assert banks.valid and not banks.single_cell


def test_width_to_depth_banks_come_from_the_same_stage_as_legacy() -> None:
    """Legacy returned each bank's ordinate as the whole number of spacings to the water's edge, after rounding the
    edge to the millimetre, at the stage it chose."""
    rng = np.random.default_rng(2)
    compared = 0
    for _ in range(400):
        spacing = float(rng.choice([1.0, 2.5, 10.0]))
        xs = random_section(rng, 2 * int(rng.integers(3, 50)) + 1, spacing)
        center = xs.elevations.size // 2
        profile1, profile2 = xs.elevations[center::-1].copy(), xs.elevations[center:].copy()
        legacy = _find_bank_using_width_to_depth_ratio(BED, profile1, profile2, profile1.size, profile2.size, spacing)
        left, right, _ = banks_module._width_to_depth_banks(xs.elevations, spacing)
        if math.isnan(left):
            assert legacy == (0, 0)
            continue
        compared += 1
        assert (math.floor(round(left, 3) / spacing), math.floor(round(right, 3) / spacing)) == legacy
    assert compared > 300


def test_flat_water_banks_match_the_legacy_ordinates() -> None:
    """Legacy returned the last ordinate below the flat water without bank elevations, and the first above it with
    them (or the ordinate count when the water reached the end)."""
    rng = np.random.default_rng(3)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 10.0]))
        xs = random_section(rng, 2 * int(rng.integers(3, 40)) + 1, spacing)
        center = xs.elevations.size // 2
        legacy = legacy_cross_section(xs, use_banks=False)
        banks = banks_by_flat_water(xs)
        for distance, profile, count in ((banks.left, legacy.da_xs_profile1, legacy.xs1_n),
                                         (banks.right, legacy.da_xs_profile2, legacy.xs2_n)):
            last_wet = legacy._find_bank(profile, count, wse=True)
            first_dry = legacy._find_bank(profile, count, wse=False)
            if first_dry == count:  # the water reaches the end of the side
                assert distance == pytest.approx(center * spacing, rel=1e-12)
            elif profile[first_dry] == BED + 0.1:  # the ground is exactly at the flat water
                assert distance == pytest.approx(first_dry * spacing, rel=1e-12)
            else:
                assert math.floor(distance / spacing) == last_wet == first_dry - 1


def test_land_cover_banks_are_half_way_to_the_first_land() -> None:
    """Water from the stream cell three ordinates out on the left and five on the right, at 10 m spacing."""
    land_cover = np.full(21, 30.0)
    land_cover[10 - 3:10 + 6] = WATER
    banks = banks_by_land_cover(make_section(np.full(21, BED), spacing=10.0), land_cover, WATER)

    assert (banks.left, banks.right) == pytest.approx((35.0, 55.0), rel=1e-12)
    assert banks.top_width == pytest.approx(90.0, rel=1e-12)  # the nine cells of water


def test_land_cover_banks_match_legacy_without_bank_elevations() -> None:
    """Legacy (without bank elevations) put each bank at the first ordinate that isn't water, and counted
    bank_index_1 + bank_index_2 - 1 cells of channel: the same width as the banks here."""
    rng = np.random.default_rng(4)
    for _ in range(300):
        size = 2 * int(rng.integers(3, 30)) + 1
        center = size // 2
        land_cover = np.where(rng.random(size) < 0.3, 30, WATER)
        land_cover[center] = WATER
        land_cover[0] = land_cover[-1] = 30  # land on both sides
        xs = random_section(rng, size, 10.0)
        _, index1, index2 = legacy_cross_section(xs, use_banks=False, land_cover=land_cover)._find_wse_and_banks_by_lc()
        banks = banks_by_land_cover(xs, land_cover, WATER)
        assert (banks.left, banks.right) == pytest.approx(((index1 - 0.5) * 10.0, (index2 - 0.5) * 10.0), rel=1e-12)
        assert banks.valid == (index1 + index2 - 1 > 1)


def test_land_cover_needs_water_at_the_stream_cell_and_land_on_each_side() -> None:
    """Legacy (without bank elevations) didn't check the stream cell, and gave a side that is water all the way
    out bank index 0, a bank at the stream cell. It also stopped the search there instead of trying the others."""
    xs = channel_and_floodplains()
    all_water_left = np.full(61, float(WATER))
    all_water_left[34:] = 30.0
    stream_cell_on_land = np.full(61, 30.0)
    stream_cell_on_land[25:30] = WATER

    assert not banks_by_land_cover(xs, all_water_left, WATER).valid
    assert math.isnan(banks_by_land_cover(xs, all_water_left, WATER).left)
    assert not banks_by_land_cover(xs, stream_cell_on_land, WATER).valid
    # So the search goes on to the width-to-depth ratio
    assert find_banks(xs, land_cover=all_water_left, water_value=WATER).method == "width_to_depth_ratio"


def test_land_cover_off_the_raster_is_no_bank() -> None:
    land_cover = np.full(21, float(WATER))
    land_cover[:3] = np.nan
    land_cover[15:] = 30.0

    banks = banks_by_land_cover(make_section(np.full(21, BED)), land_cover, WATER)

    assert math.isnan(banks.left)
    assert banks.right == pytest.approx(4.5, rel=1e-12)


@pytest.mark.parametrize("spacing, widest", [(1.0, 2.0), (14.9, 29.8), (15.0, 15.0), (30.0, 30.0)])
def test_a_width_prior_of_two_spacings_or_less_makes_a_single_cell_channel(spacing: float, widest: float) -> None:
    """One spacing or less from 15 m up. Legacy returned None, and so no single-cell channel, at exactly 15 m. The
    channel is the prior's width, and its banks' elevations are the ordinates' either side, as legacy's were."""
    xs = symmetric(BED + np.arange(0.0, 11.0), spacing)

    assert find_banks(xs, target_width=widest).single_cell
    assert not find_banks(xs, target_width=widest * 1.001).single_cell
    for width in (widest, 0.3 * spacing):
        banks = find_banks(xs, target_width=width)
        assert (banks.left, banks.right) == (width / 2, width / 2)
        assert (banks.left_elevation, banks.right_elevation) == (BED + 1.0, BED + 1.0)
        assert banks.valid and banks.method == "single_cell"


def test_a_single_cell_channel_is_the_stream_cell_s_width() -> None:
    """Half a spacing either side of the stream cell, with the neighbours' elevations. Legacy's banks were the
    neighbours, two spacings apart, which it counted as one."""
    xs = make_section([BED + 3.0, BED + 1.5, BED, BED + 0.5, BED + 3.0], spacing=10.0)
    banks = single_cell_banks(xs)

    assert (banks.left, banks.right, banks.top_width) == (5.0, 5.0, 10.0)
    assert (banks.left_elevation, banks.right_elevation) == (BED + 1.5, BED + 0.5)
    assert banks.valid and banks.single_cell


def test_banks_a_width_puts_within_a_spacing_take_the_neighbours_ground() -> None:
    """The DEM can't show a bank inside the stream cell, so the ordinate beside it stands in. Further out, the ground
    at the bank, as for any other bank."""
    xs = make_section([BED + 5.0, BED + 3.0, BED + 1.0, BED, BED + 2.0, BED + 4.0, BED + 6.0], spacing=10.0)

    for width in (4.0, 10.0, 16.0, 20.0):
        banks = banks_for_width(xs, width)
        assert (banks.left_elevation, banks.right_elevation) == (BED + 1.0, BED + 2.0)
    banks = banks_for_width(xs, 30.0)  # 15 m out, half way to the next ordinates
    assert (banks.left_elevation, banks.right_elevation) == pytest.approx((BED + 2.0, BED + 3.0))
    # Banks found from the DEM keep the ground where they are, however near
    assert banks_at_elevation(xs, BED + 0.5).left_elevation == pytest.approx(BED + 0.5)


def test_a_single_cell_channel_needs_an_ordinate_on_the_raster_each_side() -> None:
    xs = make_section([BED + 5.0, WALL, BED, BED + 1.0, BED + 2.0])

    assert not find_banks(xs, target_width=1.0).single_cell
    assert not single_cell_banks(xs).valid


def test_the_search_order() -> None:
    xs = channel_and_floodplains()
    land_cover = np.full(61, 30.0)
    land_cover[30 - 5:30 + 6] = WATER

    assert find_banks(xs, target_width=1.5, land_cover=land_cover, water_value=WATER).method == "single_cell"
    assert find_banks(xs, land_cover=land_cover, water_value=WATER).method == "land_cover"
    assert find_banks(xs).method == "width_to_depth_ratio"
    # A V's width-to-depth ratio never rises, so its banks are flat water's
    assert find_banks(symmetric(BED + 0.02 * np.arange(0.0, 41.0))).method == "flat_water"
    # A V too steep for its flat water to span two spacings has no valid banks
    none = find_banks(symmetric(BED + 0.5 * np.arange(0.0, 5.0)))
    assert (none.method, none.valid) == ("none", False)
    assert math.isnan(none.left)


def test_banks_need_a_channel_two_spacings_wide() -> None:
    """Flat water only 0.8 spacings out each side is too narrow to resolve."""
    xs = symmetric(BED + 0.125 * np.arange(0.0, 11.0))
    banks = banks_by_flat_water(xs)

    assert banks.left == pytest.approx(0.8, rel=1e-12)
    assert not banks.valid


def test_banks_for_a_width_split_it_between_the_sides() -> None:
    xs = make_section(np.full(41, BED))
    walled = make_section(np.concatenate([np.full(17, WALL), np.full(24, BED)]))  # 3 ordinates left of the stream cell

    assert (banks_for_width(xs, 10.0).left, banks_for_width(xs, 10.0).right) == pytest.approx((5.0, 5.0))
    assert (banks_for_width(walled, 10.0).left, banks_for_width(walled, 10.0).right) == pytest.approx((3.0, 7.0))
    # 3 ordinates left of the stream cell and 7 right: of 12 m, the right side can hold only 7
    short_both = make_section([WALL] * 7 + [BED] * 11 + [WALL] * 3)
    assert (banks_for_width(short_both, 12.0).left, banks_for_width(short_both, 12.0).right) == pytest.approx((3.0, 7.0))
    assert banks_for_width(xs, 100.0).top_width == pytest.approx(40.0)  # the whole section
    assert banks_for_width(xs, 100.0).valid
    # Any width, narrower than a cell included
    for width in (0.8, 1.5):
        banks = banks_for_width(xs, width)
        assert (banks.left, banks.right) == pytest.approx((width / 2, width / 2))
        assert banks.valid and banks.method == "target_width" and not banks.single_cell
    assert not banks_for_width(make_section([WALL, BED, BED + 1.0]), 0.5).valid  # off the raster to the left
    assert not banks_for_width(xs, 0.0).valid and not banks_for_width(xs, math.nan).valid


def test_banks_for_a_width_match_legacy() -> None:
    """Legacy rounded each side's share to whole spacings, at least one."""
    rng = np.random.default_rng(6)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 10.0]))
        size = 2 * int(rng.integers(2, 30)) + 1
        xs = random_section(rng, size, spacing)
        width = float(rng.uniform(1.1, 2.5) * spacing * rng.integers(1, size // 2 + 2))
        index1, index2, cells = legacy_cross_section(xs, use_banks=False)._find_bank_by_target_width(width)
        if cells <= 1:
            continue
        banks = banks_for_width(xs, width)
        clamp = lambda distance: min(max(round(distance / spacing), 1), size // 2)
        assert (clamp(banks.left), clamp(banks.right)) == (index1, index2)


def test_banks_at_an_elevation_are_the_water_edges_there() -> None:
    banks = banks_at_elevation(channel_and_floodplains(), BED + 1.5)

    assert (banks.left, banks.right) == pytest.approx((11.5, 11.5), rel=1e-12)
    assert not banks_at_elevation(channel_and_floodplains(), BED).valid


def test_valid_banks_divide_the_hydraulics() -> None:
    xs = channel_and_floodplains()
    banks = find_banks(xs)

    set_bank_distances(xs, banks)

    assert (xs.left_bank_distance, xs.right_bank_distance) == (banks.left, banks.right)
    assert hydraulics.discharge(xs, SLOPE, wse=103.0) == pytest.approx(
        hydraulics.compound_conveyance(xs.elevations, xs.mannings_n, 1.0, 12.0, 12.0, 103.0) * math.sqrt(SLOPE))
    set_bank_distances(xs, NO_BANKS)
    assert (xs.left_bank_distance, xs.right_bank_distance) == (-1.0, -1.0)


def test_the_bank_control_elevation_is_the_lower_bank_above_the_stream_cell() -> None:
    banks = Banks("test", 5.0, 7.0, 103.0, 102.5, False, True)

    assert bank_control_elevation(banks, BED) == 102.5
    assert bank_control_elevation(banks._replace(right_elevation=BED), BED) == 103.0  # at the stream cell: ignored
    assert math.isnan(bank_control_elevation(banks._replace(left_elevation=99.0, right_elevation=math.nan), BED))


def test_in_bank_ordinates_get_the_water_s_roughness() -> None:
    xs = make_section(np.full(21, BED), spacing=10.0)
    banks = Banks("test", 35.0, 20.0, BED, BED, False, True)

    assert np.flatnonzero(in_bank(xs, banks)).tolist() == [7, 8, 9, 10, 11, 12]
    set_in_bank_roughness(xs, banks, 0.02)
    assert xs.mannings_n[7:13].tolist() == [0.02] * 6
    assert xs.mannings_n[6] == xs.mannings_n[13] == 0.035
    assert not in_bank(xs, banks._replace(valid=False)).any()
    # A single-cell channel's banks are half a spacing out, so only its stream cell is between them. Legacy's bank
    # indices, the ordinates either side, were included too.
    assert np.flatnonzero(in_bank(xs, single_cell_banks(xs))).tolist() == [10]


# --- Depth --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.01, 1.0, 20.0, 500.0])
def test_a_trapezoid_s_depth_carries_the_baseflow(q: float) -> None:
    depth = trapezoid_depth(q, 16.0, 24.0, SLOPE, N)

    assert trapezoid_discharge(depth, 16.0, 24.0) == pytest.approx(q, rel=1e-12)


def test_a_trapezoid_s_depth_is_legacy_s_to_within_its_1_cm_steps() -> None:
    """Legacy stepped the depth up in steps down to 1 cm and returned the last depth not carrying the baseflow."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        top = float(rng.uniform(2.0, 200.0))
        bottom = top * (1 - 2 * float(rng.uniform(0.0, 0.5)))
        q = float(np.exp(rng.uniform(math.log(0.01), math.log(3000.0))))
        slope = float(np.exp(rng.uniform(math.log(1e-5), math.log(0.05))))
        legacy = find_depth_of_bathymetry(q, bottom, top, slope, N)
        if legacy < 24.0:
            assert legacy - 1e-9 < trapezoid_depth(q, bottom, top, slope, N) <= legacy + 0.01 + 1e-9


def test_every_channel_is_a_trapezoid_as_wide_as_its_banks() -> None:
    xs = channel_and_floodplains()
    banks = find_banks(xs)  # 24 m apart

    assert channel_depth(xs, banks, 20.0, SLOPE, trapezoid_height=0.2) == trapezoid_depth(20.0, 14.4, 24.0, SLOPE, N)
    # Without valid banks, one cell wide: at 1 m spacing, a trapezoid 1 m across the top
    assert channel_depth(xs, banks._replace(valid=False), 2.0, SLOPE, trapezoid_height=0.2) == \
        trapezoid_depth(2.0, 0.6, 1.0, SLOPE, N)


def test_a_single_cell_channel_is_a_trapezoid_whatever_the_ground_beside_it() -> None:
    """Legacy's was a triangle up to the ordinates either side, narrowed by how far they stood above the water. Here
    it's a trapezoid of the single cell's width, as wide as the width prior that made it, if one did."""
    xs = make_section([BED + 3.0, BED + 0.5, BED, BED + 0.8, BED + 3.0], spacing=10.0)

    assert channel_depth(xs, single_cell_banks(xs), 2.0, SLOPE, trapezoid_height=0.2) == \
        trapezoid_depth(2.0, 6.0, 10.0, SLOPE, N)
    prior = find_banks(xs, target_width=4.0)
    assert prior.single_cell
    assert channel_depth(xs, prior, 2.0, SLOPE, trapezoid_height=0.2) == trapezoid_depth(2.0, 2.4, 4.0, SLOPE, N)
    # Narrower is deeper, for the same baseflow
    assert channel_depth(xs, prior, 2.0, SLOPE, trapezoid_height=0.2) > \
        channel_depth(xs, single_cell_banks(xs), 2.0, SLOPE, trapezoid_height=0.2)


def test_no_baseflow_is_no_depth_and_no_slope_no_answer() -> None:
    xs = channel_and_floodplains()
    banks = find_banks(xs)

    assert channel_depth(xs, banks, 0.0, SLOPE, trapezoid_height=0.2) == 0.0
    assert math.isnan(channel_depth(xs, banks, 5.0, 0.0, trapezoid_height=0.2))
    assert math.isnan(trapezoid_depth(5.0, 0.0, 0.0, SLOPE, N))


def test_a_valid_target_depth_wins_and_always_applies() -> None:
    xs = channel_and_floodplains()
    banks = find_banks(xs)

    assert bathymetry_depth(xs, banks, 0.0, SLOPE, trapezoid_height=0.2, target_depth=1.7) == (1.7, True, "target_depth")
    solved = bathymetry_depth(xs, banks, 20.0, SLOPE, trapezoid_height=0.2, target_depth=math.nan)
    assert solved == (channel_depth(xs, banks, 20.0, SLOPE, trapezoid_height=0.2), True, "baseflow_manning")
    # Without a target or baseflow there is nothing to carve
    assert bathymetry_depth(xs, banks, 0.0, SLOPE, trapezoid_height=0.2) == (0.0, False, "baseflow_manning")


def test_power_law_geometry() -> None:
    assert power_law_geometry(100.0, 0.3, 0.5, 2.0, 0.4) == pytest.approx((3.0, 2.0 * 100.0 ** 0.4))
    assert power_law_geometry(100.0, 0.3, 0.5, None, 0.4) == (pytest.approx(3.0), None)


@pytest.mark.parametrize("height", [-0.1, 0.51, math.nan])
def test_the_trapezoid_height_is_at_most_a_half(height: float) -> None:
    xs = channel_and_floodplains()
    with pytest.raises(ValueError, match="trapezoid_height"):
        channel_depth(xs, find_banks(xs), 1.0, SLOPE, trapezoid_height=height)
    with pytest.raises(ValueError, match="trapezoid_height"):
        carve_channel(xs, find_banks(xs), 1.0, trapezoid_height=height)


# --- Carving the channel ------------------------------------------------------------------------------------------

BANKS_10_M_OUT = Banks("test", 10.0, 10.0, BED, BED, False, True)


def test_the_channel_is_a_trapezoid_between_its_banks() -> None:
    """Banks 10 m out, 2 m deep, with each side sloping over 0.2 of the 20 m top width: 4 m."""
    xs = make_section(np.full(25, BED))

    changed = carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2)

    expected = np.full(25, BED)
    expected[12 - 9:12 + 10] = [99.5, 99.0, 98.5] + [98.0] * 13 + [98.5, 99.0, 99.5]
    np.testing.assert_allclose(xs.elevations, expected, rtol=1e-15)
    assert np.flatnonzero(changed).tolist() == list(range(3, 22))  # the banks themselves stay at 100 m


def test_the_carved_channel_carries_the_baseflow() -> None:
    """A 10 m flat bed between walls rising 3 m per metre, at 5 cm spacing. Its flat-water banks are 0.1 m up the
    walls. Carved, it carries the baseflow with the water at the stream cell's old elevation."""
    offsets = 0.05 * np.arange(-400, 401)
    xs = make_section(np.where(np.abs(offsets) <= 5.0, BED, BED + 3.0 * (np.abs(offsets) - 5.0)), spacing=0.05)
    xs.mannings_n[:] = N
    banks = find_banks(xs)
    depth = channel_depth(xs, banks, 5.0, SLOPE, trapezoid_height=0.2)

    carve_channel(xs, banks, depth, trapezoid_height=0.2)

    assert banks.method == "flat_water"
    assert banks.top_width == pytest.approx(10.0 + 2 * 0.1 / 3, rel=1e-9)
    # Exactly, since the hydraulics see the carved channel's own shape
    assert hydraulics.discharge(xs, SLOPE, wse=BED) == pytest.approx(5.0, rel=1e-9)


def test_without_bank_elevations_the_channel_only_lowers_the_ground() -> None:
    """Ground 6 to 8 m from the stream cell is already below the carved slope."""
    xs = make_section(np.full(25, BED))
    xs.elevations[4:7] = 97.0

    changed = carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2)

    assert (xs.elevations[4:7] == 97.0).all()
    assert not changed[4:7].any()
    assert xs.elevations[12] == 98.0


def test_with_bank_elevations_the_channel_can_raise_the_ground_too() -> None:
    """The bank elevation, 101 m, is above this ground, so the bed at 99 m and the slopes up to 101 m at the banks
    raise it. (The user's ruling: bathymetry with bank elevations may fill as well as cut.)"""
    xs = make_section(np.full(25, BED))

    changed = carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2, bank_elevation=101.0)

    expected = np.full(25, BED)
    expected[12 - 10:12 + 11] = [101.0, 100.5, 100.0, 99.5] + [99.0] * 13 + [99.5, 100.0, 100.5, 101.0]
    np.testing.assert_allclose(xs.elevations, expected, rtol=1e-15)
    assert np.flatnonzero(changed).tolist() == list(range(2, 23))


def test_with_bank_elevations_the_channel_stays_between_its_banks() -> None:
    """Legacy ARC's staged test case: banks 2 m out at 11 m, 1.5 m deep. Legacy carved one ordinate too far, taking
    the bank ordinate down to the 9.5 m bed and the 12 m ground beyond it down to 11 m."""
    xs = make_section([12.0, 11.0, 10.4, 10.0, 10.5, 11.0, 12.0])
    banks = Banks("test", 2.0, 2.0, 11.0, 11.0, False, True)
    legacy = legacy_cross_section(xs, use_banks=True, trapezoid_height=0.1)
    legacy.Calculate_Bathymetry_Based_on_RiverBank_Elevations(
        np.full((3, 9), np.nan), bank_search_result={
            "function_used": "test", "i_bank_1_index": 2, "i_bank_2_index": 2, "bank_elev_1": 11.0,
            "bank_elev_2": 11.0, "smoothed_bank_elevation": 11.0, "is_valid": True, "bathymetry_depth": 1.5})

    carve_channel(xs, banks, 1.5, trapezoid_height=0.1, bank_elevation=11.0)

    assert xs.elevations.tolist() == pytest.approx([12.0, 11.0, 9.5, 9.5, 9.5, 11.0, 12.0])
    assert legacy.da_xs_profile2[:4].tolist() == pytest.approx([9.5, 9.5, 9.5, 11.0])


def test_a_lopsided_channel_carves_one_stream_cell_elevation() -> None:
    """Banks 20 m left and 1 m right of the stream cell, so it lies within the right bank's 4.2 m slope. Legacy carved
    each half from its own bank only, so its halves gave the stream cell 8.0 m and 9.5 m, and the raster their
    average."""
    elevations = np.full(43, 13.0)
    elevations[1:23] = 10.0  # the stream cell is ordinate 21
    xs = make_section(elevations)

    carve_channel(xs, Banks("test", 20.0, 1.0, 10.0, 10.0, False, True), 2.0, trapezoid_height=0.2)

    assert xs.elevations[21] == pytest.approx(8.0 + 2.0 * (1.0 - 1.0 / 4.2), rel=1e-15)
    assert xs.elevations[20] == pytest.approx(8.0 + 2.0 * (1.0 - 2.0 / 4.2), rel=1e-15)  # left, but 2 m from the right bank
    assert xs.elevations[11] == 8.0  # 10 m from the left bank and 11 m from the right, beyond both slopes
    assert xs.elevations[5] == pytest.approx(8.0 + 2.0 * (1.0 - 4.0 / 4.2), rel=1e-15)  # 4 m from the left bank

    # The same channel the other way round
    mirrored = make_section(elevations[::-1].copy())
    carve_channel(mirrored, Banks("test", 1.0, 20.0, 10.0, 10.0, False, True), 2.0, trapezoid_height=0.2)
    np.testing.assert_allclose(mirrored.elevations, xs.elevations[::-1], rtol=1e-15)


@pytest.mark.parametrize("use_banks", [False, True])
def test_symmetric_channels_carve_as_legacy_did_for_the_same_trapezoid(use_banks: bool) -> None:
    """Told a trapezoid of the same width, whose banks it places by index and side distance, legacy's per-side burn
    gives the same profile wherever each half is nearest its own bank."""
    rng = np.random.default_rng(9)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 3.0, 10.0]))
        center = int(rng.integers(3, 30))
        half = float(rng.uniform(1.0, center - 1) * spacing)
        trapezoid_height = float(rng.uniform(0.01, 0.5))
        depth = float(rng.uniform(0.1, 5.0))
        profile = BED + np.abs(np.arange(-center, center + 1)) * spacing * rng.uniform(0.01, 0.3)
        profile += rng.normal(0.0, 0.1, profile.size)
        profile[center] = BED
        xs = make_section(profile, spacing)
        reference = BED + 1.0 if use_banks else BED

        carve_channel(xs, Banks("test", half, half, np.nan, np.nan, False, True), depth,
                      trapezoid_height=trapezoid_height, bank_elevation=reference if use_banks else None)

        index = math.floor(half / spacing + 1e-12)
        left = profile[center::-1].copy()
        right = profile[center:].copy()
        for side in (left, right):
            _adjust_one_side_for_bathymetry(index, 2 * half, 2 * half * (1 - 2 * trapezoid_height),
                                            2 * half * trapezoid_height, np.zeros(side.size, dtype=np.int64),
                                            np.arange(side.size), side, np.full((1, side.size), np.nan),
                                            half - index * spacing, reference - depth, depth, spacing,
                                            side[np.newaxis, :].copy(), use_banks)
        np.testing.assert_allclose(xs.elevations, np.concatenate([left[::-1], right[1:]]), atol=1e-12)


def test_a_single_cell_channel_lowers_only_the_stream_cell() -> None:
    xs = make_section([BED + 3.0, BED + 1.0, BED, BED + 1.0, BED + 3.0])

    changed = carve_channel(xs, single_cell_banks(xs), 0.6, trapezoid_height=0.2)

    assert xs.elevations.tolist() == pytest.approx([BED + 3.0, BED + 1.0, BED - 0.6, BED + 1.0, BED + 3.0])
    assert np.flatnonzero(changed).tolist() == [2]
    # Banks that aren't valid carve the same way
    xs = make_section([BED + 3.0, BED + 1.0, BED, BED + 1.0, BED + 3.0])
    carve_channel(xs, NO_BANKS, 0.6, trapezoid_height=0.2)
    assert xs.elevations[2] == pytest.approx(BED - 0.6)


def test_what_isn_t_carved() -> None:
    flat = lambda: make_section(np.full(25, BED))

    # A depth of 25 m or more, without bank elevations (legacy's limit)
    assert not carve_channel(flat(), BANKS_10_M_OUT, 25.0, trapezoid_height=0.2).any()
    assert carve_channel(flat(), BANKS_10_M_OUT, 25.0, trapezoid_height=0.2, bank_elevation=BED).any()
    # A depth or bank elevation that isn't finite (legacy made the whole profile NaN)
    xs = flat()
    assert not carve_channel(xs, BANKS_10_M_OUT, math.nan, trapezoid_height=0.2, bank_elevation=BED).any()
    assert not carve_channel(xs, BANKS_10_M_OUT, 1.0, trapezoid_height=0.2, bank_elevation=math.nan).any()
    assert (xs.elevations == BED).all()
    # A single-cell channel without a neighbour on the raster
    walled = make_section([WALL, WALL, BED, BED + 1.0, BED + 1.0])
    assert not carve_channel(walled, single_cell_banks(walled), 1.0, trapezoid_height=0.2).any()


def test_a_trapezoid_height_of_zero_is_a_rectangle() -> None:
    """Legacy divided by zero at the bank ordinate, raising ZeroDivisionError without bank elevations."""
    xs = make_section(np.full(25, BED))

    carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.0, bank_elevation=BED + 1.0)

    assert xs.elevations[2] == xs.elevations[22] == BED + 1.0  # the banks are the rectangle's top
    assert (xs.elevations[3:22] == BED - 1.0).all()
    # Its walls are vertical faces in the profile
    geometry = hydraulics.hydraulic_geometry(xs, wse=BED)
    assert (geometry.area, geometry.top_width, geometry.wetted_perimeter) == pytest.approx((20.0, 20.0, 22.0))


# --- The carved channel's profile -----------------------------------------------------------------------------------


def profile_at(xs: XSection, offsets) -> np.ndarray:
    """The profile's elevation at these distances from the stream cell, taking a vertical face's inner end."""
    stations, elevations = xs.profile.stations, xs.profile.elevations
    values = []
    for offset in np.atleast_1d(offsets):
        at = np.flatnonzero(stations == offset)
        if at.size:
            values.append(elevations[at[-1] if offset < 0 else at[0]])
        else:
            values.append(np.interp(offset, stations, elevations))
    return np.array(values)


def test_a_channel_narrower_than_a_cell_is_its_own_shape_in_the_hydraulics() -> None:
    """A 12 m wide trapezoid 1.5 m deep in a stream cell 30 m wide. The raster gets the bed at the stream cell, and
    the ordinates either side keep their ground, but the hydraulics see the channel: at its banks' level its water
    is the trapezoid, 12 m across, and it carries the baseflow its depth was solved for."""
    xs = make_section([BED + 6.0, BED + 4.0, BED + 2.0, BED, BED + 2.0, BED + 4.0, BED + 6.0], spacing=30.0)
    xs.mannings_n[:] = N
    banks = find_banks(xs, target_width=12.0)
    depth = channel_depth(xs, banks, 3.0, SLOPE, trapezoid_height=0.2)

    changed = carve_channel(xs, banks, depth, trapezoid_height=0.2)

    assert banks.single_cell and banks.top_width == 12.0
    assert np.flatnonzero(changed).tolist() == [3]
    assert xs.elevations[3] == pytest.approx(BED - depth)
    assert xs.elevations[[2, 4]].tolist() == [BED + 2.0, BED + 2.0]
    assert xs.profile.stations[xs.profile.center] == 0.0
    geometry = hydraulics.hydraulic_geometry(xs, wse=BED)
    assert geometry.top_width == pytest.approx(12.0, rel=1e-12)
    assert geometry.area == pytest.approx(depth * (12.0 + 7.2) / 2, rel=1e-12)
    assert hydraulics.discharge(xs, SLOPE, wse=BED) == pytest.approx(3.0, rel=1e-9)
    # On the ordinates alone the channel would reach the neighbours, 60 m across
    plain = make_section(xs.elevations, spacing=30.0)
    assert hydraulics.hydraulic_geometry(plain, wse=BED).top_width == pytest.approx(60.0 * depth / (depth + 2.0))


def test_a_two_spacing_trapezoid_is_a_trapezoid_not_a_triangle() -> None:
    """Banks 1 m either side at 1 m spacing: the ordinates are the banks and the stream cell, which on their own make
    a triangle with 5/8 of the trapezoid's area at a trapezoid height of 0.2."""
    xs = make_section(np.full(9, BED))
    banks = Banks("test", 1.0, 1.0, BED, BED, False, True)

    carve_channel(xs, banks, 1.0, trapezoid_height=0.2, bank_elevation=BED)

    assert xs.elevations.tolist() == [BED] * 3 + [BED, BED - 1.0, BED] + [BED] * 3
    assert hydraulics.hydraulic_geometry(xs, wse=BED).area == pytest.approx(1.6)
    assert hydraulics.hydraulic_geometry(make_section(xs.elevations), wse=BED).area == pytest.approx(1.0)


def test_the_ground_runs_straight_from_each_bank_top_to_the_first_ordinate_beyond() -> None:
    """With bank elevations the channel's banks are at the bank elevation, here above the stream cell's ground. The
    ground beside the channel doesn't dip back down to the stream cell's old elevation, as it would if it were
    interpolated between the stream cell and its neighbour: it rises from the bank top to the neighbour."""
    xs = make_section([BED + 6.0, BED + 4.0, BED + 2.0, BED, BED + 2.0, BED + 4.0, BED + 6.0], spacing=30.0)
    banks = find_banks(xs, target_width=10.0)

    carve_channel(xs, banks, 1.0, trapezoid_height=0.2, bank_elevation=BED + 1.8)

    np.testing.assert_allclose(profile_at(xs, [-30.0, -15.0, -5.0, -3.0, 0.0, 3.0, 5.0, 15.0, 30.0]),
                               [BED + 2.0, BED + 1.88, BED + 1.8, BED + 0.8, BED + 0.8, BED + 0.8, BED + 1.8,
                                BED + 1.88, BED + 2.0])
    assert np.all(np.diff(xs.profile.stations) >= 0.0)


def test_without_bank_elevations_the_profile_is_the_lower_of_the_ground_and_the_channel() -> None:
    """A pool below the channel's bed keeps its ground, and where the ground crosses the channel's side the profile
    turns there."""
    xs = make_section(np.full(25, BED))
    xs.elevations[4:7] = 97.0  # 6 to 8 m left of the stream cell, below the 98 m bed
    original = xs.elevations.copy()

    carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2)

    offsets = np.linspace(-12.0, 12.0, 481)
    ground = np.interp(offsets, np.arange(-12.0, 13.0), original)
    channel = np.where(np.abs(offsets) <= 10.0, 98.0 + 2.0 * np.maximum(1.0 - (10.0 - np.abs(offsets)) / 4.0, 0.0),
                       np.inf)
    np.testing.assert_allclose(profile_at(xs, offsets), np.minimum(ground, channel), atol=1e-12)


@pytest.mark.parametrize("use_banks", [False, True])
def test_the_profile_goes_through_the_carved_ordinates(use_banks: bool) -> None:
    """On random sections, banks, depths and trapezoid heights: the profile is the carved ordinates' elevation at
    each ordinate, the ground outside the banks, and without bank elevations never above the ground."""
    rng = np.random.default_rng(10)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 7.5, 30.0]))
        xs = random_section(rng, 2 * int(rng.integers(2, 20)) + 1, spacing)
        ground = xs.elevations.copy()
        center = ground.size // 2
        left, right = (float(rng.uniform(0.05, center) * spacing) for _ in range(2))
        if rng.random() < 0.3:
            left = float(rng.integers(1, center + 1) * spacing)  # a bank on an ordinate
        banks = Banks("test", left, right, np.nan, np.nan, False, True)
        reference = BED + float(rng.uniform(0.0, 2.0)) if use_banks else None

        changed = carve_channel(xs, banks, float(rng.uniform(0.05, 3.0)),
                                trapezoid_height=float(rng.uniform(0.0, 0.5)), bank_elevation=reference)

        offsets = (np.arange(ground.size) - center) * spacing
        outside = (offsets < -left - 1e-9 * spacing) | (offsets > right + 1e-9 * spacing)
        assert not changed[outside].any()
        stations = xs.profile.stations
        assert np.all(np.diff(stations) >= 0.0) and stations[xs.profile.center] == 0.0
        interior = ~outside & (np.abs(offsets + left) > 1e-9 * spacing) & (np.abs(offsets - right) > 1e-9 * spacing)
        np.testing.assert_allclose(profile_at(xs, offsets[interior]), xs.elevations[interior], atol=1e-9)
        np.testing.assert_allclose(profile_at(xs, offsets[outside]), ground[outside], atol=1e-9)
        if not use_banks:
            dense = np.linspace(offsets[0], offsets[-1], 2001)
            assert np.all(profile_at(xs, dense) <= np.interp(dense, offsets, ground) + 1e-9)


def test_the_profile_keeps_the_ordinates_roughness() -> None:
    """Each vertex takes the Manning's n of the segment between ordinates it falls in, its inner ordinate's."""
    xs = make_section(np.full(9, BED), spacing=10.0)
    xs.mannings_n[:] = [0.1, 0.09, 0.08, 0.07, 0.03, 0.06, 0.05, 0.04, 0.02]

    carve_channel(xs, Banks("test", 15.0, 4.0, BED, BED, False, True), 1.0, trapezoid_height=0.2,
                  bank_elevation=BED)

    stations, n = xs.profile.stations, xs.profile.mannings_n
    expected = {-40.0: 0.1, -30.0: 0.09, -20.0: 0.08, -15.0: 0.07, -11.2: 0.07, 0.0: 0.03, 0.2: 0.03, 4.0: 0.03,
                10.0: 0.06, 20.0: 0.05, 30.0: 0.04, 40.0: 0.02}
    assert {float(s): float(v) for s, v in zip(np.round(stations, 9), n)} == pytest.approx(expected)


# --- Rasters ------------------------------------------------------------------------------------------------------


def test_ordinate_cells_follow_the_sampled_cross_section() -> None:
    """A stream flowing south (pi / 2) has its cross section along its row, from west (on the left, looking
    upstream) to east. Flowing south-east, each ordinate steps a cell in both directions, from the south-west."""
    rows, cols = ordinate_cells(5, 5, np.pi / 2, 40.0, 10.0, 10.0)
    assert rows.tolist() == [5] * 5
    assert cols.tolist() == [3, 4, 5, 6, 7]

    rows, cols = ordinate_cells(5, 5, np.pi / 4, 40.0, 10.0, 10.0)
    assert list(zip(rows.tolist(), cols.tolist())) == [(6, 4), (5, 5), (4, 6)]


def test_land_cover_is_sampled_from_the_nearest_cells_and_nan_off_the_raster() -> None:
    land_cover = np.arange(25, dtype=np.uint8).reshape(5, 5)

    values = sample_land_cover(land_cover, 2, 2, np.pi / 2, 80.0, 10.0, 10.0)

    assert np.isnan(values[:2]).all() and np.isnan(values[-2:]).all()
    assert values[2:7].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_the_raster_takes_the_first_value_then_averages() -> None:
    raster = np.full((2, 4), np.nan, dtype=np.float32)
    rows = np.array([0, 0, 0, 5])
    cols = np.array([0, 1, 2, 0])
    changed = np.array([True, True, False, True])

    burn_into_raster(raster, rows, cols, np.array([1.0, 2.0, 3.0, 4.0]), changed)
    burn_into_raster(raster, rows, cols, np.array([3.0, 2.0, 3.0, 4.0]), changed)

    assert raster[0, 0] == 2.0  # 1, then the average of 1 and 3
    assert raster[0, 1] == 2.0
    assert np.isnan(raster[0, 2])  # never changed
    assert np.isnan(raster[1]).all()  # the ordinate off the raster was skipped


def test_a_gap_with_bathymetry_in_four_of_the_cells_around_it_gets_their_mean() -> None:
    """The middle cell has bathymetry in four of the eight cells around it, and without the last of them, three. The
    others, next to the raster's edge, have at most three, since beyond the edge counts as no bathymetry."""
    raster = np.full((3, 3), np.nan)
    raster[0, 0], raster[0, 2], raster[2, 0], raster[2, 1] = 1.0, 2.0, 3.0, 6.0
    three = raster.copy()
    three[2, 1] = np.nan

    assert fill_bathymetry_gaps(raster) is raster
    assert raster[1, 1] == 3.0
    assert np.isnan(raster).sum() == 4
    assert np.isnan(fill_bathymetry_gaps(three)).sum() == 6


def test_every_gap_is_filled_from_the_raster_as_it_was() -> None:
    """The gap at (1, 1) has four neighbours with bathymetry, and the one at (1, 2) three, so it stays a gap even
    though filling (1, 1) gives it a fourth. Given a fourth of its own at (0, 2), it gets the mean of its four, leaving
    out (1, 1)'s fill."""
    raster = np.full((3, 4), np.nan)
    raster[0, 0], raster[0, 1], raster[1, 0], raster[2, 0] = 1.0, 2.0, 3.0, 6.0
    raster[1, 3], raster[2, 3] = 5.0, 5.0
    fourth = raster.copy()
    fourth[0, 2] = 8.0

    fill_bathymetry_gaps(raster)
    fill_bathymetry_gaps(fourth)

    assert raster[1, 1] == 3.0
    assert np.isnan(raster[1, 2])
    assert fourth[1, 1] == 4.0
    assert fourth[1, 2] == 5.0


def test_the_gap_fill_matches_legacy_s() -> None:
    """Random rasters, float32 and float64, from nearly empty to nearly full."""
    rng = np.random.default_rng(2)
    for trial in range(300):
        shape = tuple(int(s) for s in rng.integers(1, 30, 2))
        raster = rng.normal(95.0, 2.0, shape)
        raster[rng.random(shape) < rng.uniform(0.0, 1.0)] = np.nan
        if trial % 2 == 0:
            raster = raster.astype(np.float32)

        expected = _fill_bathymetry_nan_cells(raster.copy())

        np.testing.assert_array_equal(fill_bathymetry_gaps(raster), expected)
        assert raster.dtype == expected.dtype


def test_without_bank_elevations_no_gap_is_filled_above_the_ground() -> None:
    """A channel carved to 95 m through ground at 100 m, with a pool at 94 m that it wasn't carved into, and one cell
    whose bathymetry is above the ground. That cell is dropped and filled again from its neighbours, as legacy did,
    but where legacy then filled the pool to 95 m, above its ground, here it stays a gap. Bathymetry level with the
    ground stays, and so does a fill level with it."""
    ground = np.full((7, 9), 100.0, dtype=np.float32)
    ground[3, 4] = 94.0
    ground[3, 2] = ground[2, 3] = 95.0
    raster = np.full((7, 9), np.nan, dtype=np.float32)
    raster[2:5, 1:8] = 95.0
    raster[3, 4] = raster[3, 2] = np.nan
    raster[2, 6] = 100.5

    expected = _fill_bathymetry_nan_cells(np.where(raster > ground, np.nan, raster).astype(np.float32))
    fill_bathymetry_gaps(raster, ground)

    assert expected[3, 4] == 95.0
    assert np.isnan(raster[3, 4])
    assert raster[2, 6] == raster[3, 2] == raster[2, 3] == 95.0
    expected[3, 4] = np.nan
    np.testing.assert_array_equal(raster, expected)


def test_the_gap_fill_needs_a_2d_floating_raster_and_ground_like_it() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fill_bathymetry_gaps(np.full(4, np.nan))
    with pytest.raises(TypeError, match="floating"):
        fill_bathymetry_gaps(np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="ground"):
        fill_bathymetry_gaps(np.full((2, 2), np.nan), np.zeros((2, 3)))


# --- Speed (each compiled loop runs once first, so compilation isn't counted) --------------------------------------

# Compiled but not cached: a cache wouldn't pick up changes to the functions these call in other modules.


@njit
def _repeat_legacy_trapezoid(q, calls):
    total = 0.0
    for i in range(calls):
        total += find_depth_of_bathymetry(q * (1.0 + 1e-9 * (i % 7)), 16.0, 24.0, SLOPE, N)
    return total


@njit
def _repeat_trapezoid(q, calls):
    total = 0.0
    for i in range(calls):
        total += trapezoid_depth(q * (1.0 + 1e-9 * (i % 7)), 16.0, 24.0, SLOPE, N)
    return total


@njit
def _repeat_legacy_width_to_depth(profile1, profile2, spacing, calls):
    total = 0
    for _ in range(calls):
        total += _find_bank_using_width_to_depth_ratio(BED, profile1, profile2, profile1.size, profile2.size, spacing)[0]
    return total


@njit
def _repeat_width_to_depth(elevations, spacing, calls):
    total = 0.0
    for _ in range(calls):
        total += banks_module._width_to_depth_banks(elevations, spacing)[0]
    return total


@njit
def _repeat_legacy_burn(profile, rows, cols, dem, output, calls):
    total = 0.0
    for _ in range(calls):
        for _side in range(2):
            carved = profile.copy()
            _adjust_one_side_for_bathymetry(12, 240.0, 144.0, 48.0, rows, cols, carved, output, 0.0, 98.0, 2.0,
                                            10.0, dem, False)
            total += carved[0]
    return total


@njit
def _repeat_carve(elevations, calls):
    """The ordinates' carve, legacy's burn's job. The channel's profile is extra, and in the whole-section test."""
    total = 0.0
    changed = np.zeros(elevations.size, dtype=np.bool_)
    for _ in range(calls):
        carved = elevations.copy()
        channel_module._carve(carved, 10.0, 120.0, 120.0, BED, 2.0, 0.2, True, changed)
        total += carved[carved.size // 2]
    return total


def _seconds_per_call(repeat, *args, calls: int) -> float:
    """Best of five timed runs, after one untimed call to compile."""
    repeat(*args, 1)
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        repeat(*args, calls)
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def _seconds_per_python_call(function, calls: int) -> float:
    function()
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def typical_section() -> XSection:
    """5 km at 10 m spacing (501 ordinates): a 20 m wide, 2 m deep channel in a valley with 2% side slopes."""
    distance = np.abs(10.0 * np.arange(-250, 251))
    rng = np.random.default_rng(1)
    elevations = BED + np.minimum(distance, 10.0) * 0.2 + np.maximum(distance - 10.0, 0.0) * 0.02
    elevations += rng.normal(0.0, 0.05, distance.size)
    elevations[250] = BED
    return make_section(elevations, spacing=10.0)


def test_the_bank_search_is_faster_than_legacy_s() -> None:
    xs = typical_section()
    legacy = legacy_cross_section(xs, use_banks=False)

    new = _seconds_per_python_call(lambda: find_banks(xs), calls=2000)
    old = _seconds_per_python_call(legacy.get_wse_or_lc_bank_search_result, calls=500)
    kernel = _seconds_per_call(_repeat_width_to_depth, xs.elevations, 10.0, calls=2000)
    legacy_kernel = _seconds_per_call(_repeat_legacy_width_to_depth, xs.elevations[250::-1].copy(),
                                      xs.elevations[250:].copy(), 10.0, calls=500)

    assert new < old
    assert kernel < legacy_kernel


def test_the_trapezoid_depth_and_the_carve_are_faster_than_legacy_s() -> None:
    xs = typical_section()
    rows = np.zeros(251, dtype=np.int64)
    cols = np.arange(251)

    for q in [1.0, 20.0, 300.0]:
        assert _seconds_per_call(_repeat_trapezoid, q, calls=20000) < _seconds_per_call(
            _repeat_legacy_trapezoid, q, calls=20000)
    assert _seconds_per_call(_repeat_carve, xs.elevations, calls=20000) < _seconds_per_call(
        _repeat_legacy_burn, xs.elevations[250:].copy(), rows, cols, xs.elevations[250:][np.newaxis, :].copy(),
        np.full((1, 251), np.nan), calls=10000)


def test_a_cross_section_s_whole_bathymetry_is_faster_than_legacy_s() -> None:
    """Finding the banks, the depth, and carving the channel into a sampled cross section. Legacy reused one
    CrossSection, reloading each sampled profile into it."""
    xs = typical_section()
    raster = np.full((1, 501), np.nan)
    rows, cols = np.zeros(501, dtype=np.intp), np.arange(501, dtype=np.intp)
    legacy = legacy_cross_section(xs, use_banks=False)
    profile1, profile2 = legacy.da_xs_profile1.copy(), legacy.da_xs_profile2.copy()
    legacy_raster = np.full((3, 503), np.nan)

    def new() -> None:
        section = make_section(xs.elevations, spacing=10.0)
        banks = find_banks(section)
        set_bank_distances(section, banks)
        depth = bathymetry_depth(section, banks, 20.0, SLOPE, trapezoid_height=0.2)
        burn_into_raster(raster, rows, cols, section.elevations,
                         carve_channel(section, banks, depth.depth, trapezoid_height=0.2))

    def old() -> None:
        legacy.da_xs_profile1[:] = profile1
        legacy.da_xs_profile2[:] = profile2
        result = legacy.get_wse_or_lc_bank_search_result()
        result["bathymetry_depth"] = legacy.calculate_hydraulic_bathymetry_depth(20.0, SLOPE, result)
        legacy.Calculate_Bathymetry_Based_on_WSE_or_LC(legacy_raster, bank_search_result=result)

    assert _seconds_per_python_call(new, calls=1000) < _seconds_per_python_call(old, calls=300)


def test_the_gap_fill_is_faster_than_legacy_s() -> None:
    """A 2000 x 2000 raster, empty but for channels three cells wide every 50 columns, with 2% of their cells gaps."""
    rng = np.random.default_rng(0)
    raster = np.full((2000, 2000), np.nan, dtype=np.float32)
    for first in range(10, 2000, 50):
        raster[:, first:first + 3] = 95.0
    raster[rng.random(raster.shape) < 0.02] = np.nan

    new = _seconds_per_python_call(lambda: fill_bathymetry_gaps(raster.copy()), calls=3)
    old = _seconds_per_python_call(lambda: _fill_bathymetry_nan_cells(raster.copy()), calls=3)

    assert new < old
