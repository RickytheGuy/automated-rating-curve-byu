"""Moving a cross section from its stream cell to lower ground beside it (legacy Low_Spot_Range).

A stream raster can put a stream a cell or two to one side of the channel in the DEM. Legacy ARC looked along each
cross section, a few ordinates either side of the stream cell, for lower ground, and if there was some, moved the
cross section there: to the cell of the lowest ordinate, sampling it again in the same direction. The angle search,
the banks and the bathymetry then all started from the new cell.

Errors in the legacy code, not repeated here
--------------------------------------------
- Low_Spot_Range is how many ordinates either side to look at, but legacy looked at one fewer, because it counted
  the stream cell as the first: with 10 it looked 9 ordinates out, and with 1 it didn't look at all. Here a range of
  N looks at the N ordinates either side.

Where legacy used its own markers
---------------------------------
- Legacy only moved to ground above 0 m, which kept it off the zeros its profiles held beyond the raster's edge, and
  off nodata. Here ordinates beyond the edge are left out, and NaN, this code's nodata, is never lower, so any
  elevation can be the low spot, including at or below 0 m.

Numerical differences
---------------------
- The new cell is the one nearest the lowest ordinate. Legacy's oblique cross sections put their ordinates in the
  wrong places (see arc.xsection.sampling), so it agrees with this only for cross sections along rows and columns.

Later on, legacy's bank smoothing put each reach's cross sections in order by the cells they had moved to, but its
depth fill and bed smoothing by their stream cells. arc.bathymetry's ReachSections takes the stream cells, for all
three.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from arc.xsection.sampling import sample_elevations

_ALONG_A_ROW = 1e-12  # a cross section whose direction's sine is smaller than this runs along a row


@njit(cache=True, error_model="numpy")
def _low_spot_cell(dem, row, col, stream_direction, cross_section_length, dx, dy, low_spot_range):
    elevations, spacing = sample_elevations(dem, row, col, stream_direction, cross_section_length, dx, dy)
    center = elevations.size // 2
    xs_direction = stream_direction - np.pi / 2
    cos, sin = np.cos(xs_direction), np.sin(xs_direction)
    # Legacy looked at this side first: towards the higher rows, or along a row towards the higher columns
    first = 1 if sin > _ALONG_A_ROW or (abs(sin) <= _ALONG_A_ROW and cos > 0.0) else -1
    rows, cols = dem.shape
    lowest, best = elevations[center], 0
    for k in range(1, min(low_spot_range, center) + 1):
        for step in (first * k, -first * k):
            distance = spacing * step
            x, y = col + cos * distance / dx, row + sin * distance / dy
            if 0.0 <= x <= cols - 1 and 0.0 <= y <= rows - 1 and elevations[center + step] < lowest:
                lowest, best = elevations[center + step], step
    if best == 0:
        return row, col
    distance = spacing * best
    return int(np.rint(row + sin * distance / dy)), int(np.rint(col + cos * distance / dx))


def low_spot_cell(dem: np.ndarray, row: int, col: int, stream_direction: float, cross_section_length: float,
                  dx: float, dy: float, low_spot_range: int) -> tuple[int, int]:
    """The cell to centre the cross section on instead of the stream cell (legacy
    adjust_cross_section_to_lowest_point): the cell nearest the lowest ordinate within low_spot_range ordinates
    either side of the stream cell, along the cross section sample_cross_section takes with these arguments. The
    stream cell itself if none is lower, or low_spot_range isn't positive.

    Of ordinates equally low, the nearest the stream cell wins, and of two as near, the one towards the higher rows,
    or for a cross section along a row the higher columns, as legacy looked at that side first. The cross section is
    then sampled again at the new cell, in the same direction.
    """
    row, col = int(row), int(col)
    if low_spot_range <= 0:
        return row, col
    new_row, new_col = _low_spot_cell(dem, row, col, float(stream_direction), float(cross_section_length), float(dx),
                                      float(dy), int(low_spot_range))
    return int(new_row), int(new_col)
