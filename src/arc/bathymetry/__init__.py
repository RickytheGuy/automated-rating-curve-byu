"""Bathymetry for a sampled cross section: finding its banks, how deep its channel is, and carving the channel in.

These are legacy ARC's per-cross-section bathymetry steps (CrossSection's bank searches and bathymetry methods, and
the functions they call), for XSection and arc.hydraulics. A cross section goes through them in order:

    banks = find_banks(xs, target_width=..., land_cover=..., water_value=...)
    set_bank_distances(xs, banks)                      # for the hydraulics to divide the channel there
    depth = bathymetry_depth(xs, banks, baseflow, slope, trapezoid_height=..., bank_elevation=...)
    if depth.apply:
        changed = carve_channel(xs, banks, depth.depth, trapezoid_height=..., bank_elevation=...)
        burn_into_raster(bathymetry, *ordinate_cells(...), xs.elevations, changed)

bank_elevation is None without Bathy_Use_Banks. With it, it is the bank elevation to carve below, which legacy ARC
smoothed along the reach first.

Banks are distances from the stream cell, not ordinate indices, so a bank can fall between ordinates and a channel's
top width is the distance between its banks. That is the main way the results differ from legacy ARC. The others
are errors in the legacy code that aren't repeated here. All of them are listed below.

Where legacy ARC used bank indices
----------------------------------
- Each bank is where its method finds it. For the width-to-depth ratio and flat water, that's the water's edge. Legacy
  used the last ordinate under water, or with bank elevations the first above flat water. For the land cover it's
  half way between the last ordinate of water and the first that isn't. Legacy used the first that isn't, or with
  bank elevations the last water. A target width is used as it is, where legacy rounded it to whole spacings, at
  least one each side. A bank's elevation is the ground's at the bank, where legacy used its bank ordinate's
  elevation, which for the width-to-depth ratio was an ordinate under water.
- A channel's top width is the distance between its banks. Legacy counted bank_index_1 + bank_index_2 - 1 spacings.
  For width-to-depth and flat-water banks, which were the last ordinates under water, that is one spacing less than
  the distance between those ordinates. It is one to three spacings less than between the water's edges. These
  channels are wider here, and for the same baseflow shallower. On a 20 m bed with 1:1 banks 2 m high (banks 24 m
  apart), legacy found banks at ordinates 12 and 12 and a width of 23 m.
- A channel is resolved if it is at least two spacings wide, whichever method found it. Legacy needed more than one
  of its counted cells, which depended on the method's index convention; without bank elevations it was the same
  test for land-cover banks.
- With bank elevations, legacy moved each bank from its ordinate to where the ground reached the bank elevation
  within the next segment, or half a spacing out when it didn't. Here the banks stay where they were found.
  Legacy's own notes describe that as keeping the local bank indices and top widths.

Errors in the legacy code, not repeated here
--------------------------------------------
- With bank elevations the channel was carved one ordinate beyond its banks. Calculate_Bathymetry_Based_on_
  RiverBank_Elevations passes i_bank_1_index + 1 as well as the side distance. So the bank ordinate went down to
  the bed, and the next ordinate out went down to the bank elevation. In the staged test case (banks at 11 m, 1.5 m
  deep), the bank went from 11 m to 9.5 m and the ground beyond it from 12 m to 11 m.
- Each half of the cross section was carved by measuring from its own bank only. When the stream cell lies on the
  far bank's sloping side, the halves set it to different elevations and the raster took their average: with banks
  20 and 1 ordinates out, 8.0 m and 9.5 m, so 8.75 m. Here every ordinate is measured from the nearer bank.
- The single-cell width prior test returned None, so no single-cell channel, at a spacing of exactly 15 m. Here 15 m
  counts as coarse, allowing one spacing.
- Land-cover banks without bank elevations had three errors:
  - The stream cell didn't have to be water.
  - A side that was water all the way out got bank index 0, a bank at the stream cell.
  - A one-cell result ended the search as invalid, where with bank elevations the search went on.
  Here the stream cell has to be water, a side without land has no bank, and an unresolved land-cover result goes
  on to the next method either way.
- A trapezoid height of 0 divided by zero at the bank ordinate, which raised ZeroDivisionError without bank
  elevations. Above 0.5 the depth solve got a negative bottom width. Here 0 is a rectangle, and values outside 0 to
  0.5 raise ValueError.
- With bank elevations a NaN depth made the whole profile NaN. Here a depth or reference level that isn't finite
  carves nothing.
- A single-cell channel overwrote its raster cell, while wider channels averaged with any value already there.
  Here both average.
- With land-cover banks and without bank elevations, the bank index was the first ordinate that isn't water. So
  that land ordinate got the water's roughness. Here only the ordinates between the banks do.

Numerical differences
---------------------
- Depths are solved exactly. Legacy stepped the depth: a trapezoid came out up to 1 cm shallow and a triangle up to
  10 cm deep.
- A slope that isn't positive gives a NaN depth, which carves nothing. Legacy stepped to 25 m.
- Without bank elevations, the channel only lowers ground lower than the cross section's own elevations. Legacy
  compared against the DEM cell, and an XSection holds no DEM values.
- bank_control_elevation ignores banks at or below the stream cell. Legacy's np.isclose also ignored banks a
  little above it, up to about 1 cm at 1000 m elevation, since its tolerance grows with elevation.

Not here yet
------------
- The reach and network steps that run between these, since they work on all of a reach's cross sections together:
  - smoothing the bank elevations, including the reach width filters and fills
  - the reach depth and bed smoothing
  - the network depth solvers
  - filling gaps in the bathymetry raster
  The kernels they need for one cross section are here: banks_for_width, single_cell_banks, banks_at_elevation and
  bank_control_elevation.
- The INFLECT width-depth curve (_calculate_inflect_curve_with_depths). It feeds the representative cross section's
  terrace depth, and a reach bank depth that no bank search uses. Its moving-window slope takes 10 points, from 5
  before to 4 after, so it is centred half a step off, which is worth checking before porting it.
- Legacy functions nothing calls weren't ported:
  - _find_bank_inflection_point, get_representative_bank_indices and calc_bankfull_elevation
  - _find_bank_using_reach_scale_inflection and _find_bank_indices_from_elevation (banks_at_elevation does their job)
  - _estimate_minimum_smoothed_bank_elevation and extract_scalar_hydraulic_geometry
"""
from arc.bathymetry.banks import (Banks, bank_control_elevation, banks_at_elevation, banks_by_flat_water,
                                  banks_by_land_cover, banks_by_width_to_depth_ratio, banks_for_width, find_banks,
                                  in_bank, set_bank_distances, set_in_bank_roughness, single_cell_banks)
from arc.bathymetry.channel import carve_channel
from arc.bathymetry.depth import (BathymetryDepth, bathymetry_depth, channel_depth, power_law_geometry,
                                  trapezoid_depth, triangle_depth)
from arc.bathymetry.raster import burn_into_raster, ordinate_cells, sample_land_cover

__all__ = [
    "Banks", "BathymetryDepth", "bank_control_elevation", "banks_at_elevation", "banks_by_flat_water",
    "banks_by_land_cover", "banks_by_width_to_depth_ratio", "banks_for_width", "bathymetry_depth", "burn_into_raster",
    "carve_channel", "channel_depth", "find_banks", "in_bank", "ordinate_cells", "power_law_geometry",
    "sample_land_cover", "set_bank_distances", "set_in_bank_roughness", "single_cell_banks", "trapezoid_depth",
    "triangle_depth",
]
