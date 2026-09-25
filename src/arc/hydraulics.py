"""Hydraulic geometry and Manning's discharge for a sampled cross section.

The water surface starts at the centre ordinate (the stream cell) and spreads outward in both
directions until it meets ground higher than the water surface. Ground is linear between ordinates,
so the water's edge is found part way along the segment where it meets the ground. Low ground beyond
a higher point stays dry until the water surface rises above that point. Both ends of the cross
section act as vertical walls, like the off-raster walls from sampling.

Discharge uses Manning's equation in SI units. Each segment of wetted ground takes the Manning's n of
its end nearer the centre, so wall ordinates never contribute their own n. The section's roughness is
the Horton-Einstein composite n = (sum(P_i * n_i**1.5) / P) ** (2/3), which makes the conveyance
K = A**(5/3) / sum(P_i * n_i**1.5) ** (2/3) and the discharge Q = K * sqrt(slope).

A cross section with banks is divided at them into a left overbank, the channel and a right overbank,
and its conveyance is the sum of theirs (the divided channel method; see the notes above
compound_wetted_geometry). The banks don't change where the water is, only how its conveyance adds up.

The functions taking arrays are compiled with numba, so they can be called from other compiled code
in ARC's per-cell loop. The functions taking an XSection are conveniences on top of them, and use the
XSection's banks if it has any.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from arc.xsection.xsection import XSection


class HydraulicGeometry(NamedTuple):
    area: float
    wetted_perimeter: float
    hydraulic_radius: float
    top_width: float
    mannings_n: float  # composite roughness of the wetted perimeter


class CompoundGeometry(NamedTuple):
    """The wetted geometry of each subsection of a cross section divided at its banks."""
    left_overbank: HydraulicGeometry
    channel: HydraulicGeometry
    right_overbank: HydraulicGeometry


@njit(cache=True, error_model="numpy")
def _n_15(mannings_n):
    return mannings_n * math.sqrt(mannings_n)


@njit(cache=True, error_model="numpy")
def _length(spacing, rise):
    return math.sqrt(spacing * spacing + rise * rise)


_INVERSE_CBRT_MAGIC = np.int64(0x553EF0FF289DD796)


@njit(cache=True, error_model="numpy")
def _inverse_cbrt(x):
    """x**(-1/3) for 1e-300 < x < 1e300, to within about 1e-15, without dividing.

    Subtracting a third of the float's bits from a constant gives a first guess within 3.5%. Each step multiplies
    by the start of the binomial series for (1 - e)**(-1/3), where e = 1 - x * r**3, and so has fourth-order
    convergence: two steps reach float64 precision. Conveyance this way takes well under half the time of the two
    fractional powers it replaces, which matters most for a divided cross section's three subsections.
    """
    r = np.int64(_INVERSE_CBRT_MAGIC - np.float64(x).view(np.int64) // 3).view(np.float64)
    for _ in range(2):
        e = 1.0 - x * r * r * r
        r = r * (1.0 + e * (1.0 / 3.0 + e * (2.0 / 9.0 + e * (14.0 / 81.0))))
    return r


@njit(cache=True, error_model="numpy")
def _cubed_conveyance(area, weighted_perimeter):
    """Conveyance cubed, A**5 / W**2, where W = sum(P_i * n_i**1.5)."""
    if area <= 0.0:
        return 0.0
    area_squared = area * area
    return area_squared * area_squared * area / (weighted_perimeter * weighted_perimeter)


@njit(cache=True, error_model="numpy")
def _conveyance(area, weighted_perimeter):
    """A**(5/3) / W**(2/3), as A**2 * (A * W**2)**(-1/3)."""
    if area <= 0.0:
        return 0.0
    x = area * weighted_perimeter * weighted_perimeter
    if 1e-300 < x < 1e300:
        return area * area * _inverse_cbrt(x)
    return area ** (5.0 / 3.0) / weighted_perimeter ** (2.0 / 3.0)  # out of range, NaN, or no roughness


@njit(cache=True, error_model="numpy")
def _side_geometry(elevations, mannings_n, center, step, spacing, wse):
    """Area, wetted perimeter, top width and sum(P_i * n_i**1.5) from the centre out to one water edge."""
    area = 0.0
    perimeter = 0.0
    top_width = 0.0
    weighted_perimeter = 0.0
    end = elevations.size - 1 if step > 0 else 0
    j = center
    while j != end:
        z_in = elevations[j]
        z_out = elevations[j + step]
        depth_in = wse - z_in
        if wse > z_out:
            # Both ends of the segment are under water
            length = _length(spacing, z_out - z_in)
            area += spacing * (depth_in + wse - z_out) / 2
            perimeter += length
            weighted_perimeter += length * _n_15(mannings_n[j])
            top_width += spacing
            j += step
        else:
            # The water meets the ground part way along the segment
            fraction = depth_in / (z_out - z_in)
            length = fraction * _length(spacing, z_out - z_in)
            area += spacing * fraction * depth_in / 2
            perimeter += length
            weighted_perimeter += length * _n_15(mannings_n[j])
            top_width += spacing * fraction
            return area, perimeter, top_width, weighted_perimeter

    # The water reaches the end of the cross section, which acts as a vertical wall
    depth_end = wse - elevations[end]
    perimeter += depth_end
    weighted_perimeter += depth_end * _n_15(mannings_n[end])
    return area, perimeter, top_width, weighted_perimeter


@njit(cache=True, error_model="numpy")
def wetted_geometry(elevations, mannings_n, spacing, wse):
    """Area, wetted perimeter, top width and sum(P_i * n_i**1.5) at a water surface elevation.

    Everything is zero when the water surface is at or below the stream cell.
    """
    center = elevations.size // 2
    if not wse > elevations[center]:
        return 0.0, 0.0, 0.0, 0.0
    left = _side_geometry(elevations, mannings_n, center, -1, spacing, wse)
    right = _side_geometry(elevations, mannings_n, center, 1, spacing, wse)
    return left[0] + right[0], left[1] + right[1], left[2] + right[2], left[3] + right[3]


@njit(cache=True, error_model="numpy")
def conveyance(elevations, mannings_n, spacing, wse):
    """Manning's conveyance A * R**(2/3) / n at a water surface elevation, so that Q = K * sqrt(slope)."""
    area, _, _, weighted_perimeter = wetted_geometry(elevations, mannings_n, spacing, wse)
    return _conveyance(area, weighted_perimeter)


# The water surface can also be raised from the stream cell one interval at a time, where an interval
# is a range of water surface elevations (start, end] over which the same ordinates are wet. Within an
# interval, with u the height above its start, area is quadratic in u and everything else is linear:
#   A = AREA + TOP_WIDTH * u + HALF_DWIDTH * u**2      T = TOP_WIDTH + 2 * HALF_DWIDTH * u
#   P = PERIMETER + DPERIMETER * u                      W = WEIGHTED + DWEIGHTED * u,  W = sum(P_i * n_i**1.5)
# A conveyance table has a row of these per interval. Conveyance is stored cubed, as A**5 / W**2, which
# orders the same way and needs no fractional powers: K3_START and K3_END are at the ends of the
# interval, and K3_MAX is the highest at or below its end.
#
# Within an interval, dK/du has the sign of 5 * T * W - 2 * W' * A, which is c0 + c1 * u + c2 * u**2 with
# c1 and c2 >= 0, so it only increases for u >= 0. Conveyance therefore either rises throughout an
# interval, or falls and then rises. So it is highest at one of the ends, and crosses any level between
# its start and its end only once.
(_START, _END, _AREA, _TOP_WIDTH, _HALF_DWIDTH, _PERIMETER, _DPERIMETER, _WEIGHTED, _DWEIGHTED,
 _K3_START, _K3_END, _K3_MAX) = range(12)
_COLUMNS = 12


@njit(cache=True, error_model="numpy")
def _raise_water_surface(elevations, mannings_n, spacing, stop_k3, max_wse, table):
    """Raise the water surface from the stream cell one interval at a time, writing interval i to row
    min(i, len(table) - 1), until an interval's cubed conveyance reaches stop_k3 or the interval reaches max_wse.
    Returns the number of intervals.

    The water edge on each side moves past an ordinate once the water surface is above it and above every
    ordinate between it and the centre. So taking whichever side's next ordinate is lower visits every
    interval in order, in one pass outward from the centre.
    """
    n = elevations.size
    center = n // 2
    rows = table.shape[0]
    level = float(elevations[center])
    k3_max = 0.0
    count = 0

    # Outermost wet ordinate on each side, and totals over the fully wet segments inside it: the count,
    # the sum of their mid elevations, their length, and their n-weighted length
    left = center
    right = center
    left_count, left_mid, left_length, left_weighted = 0, 0.0, 0.0, 0.0
    right_count, right_mid, right_length, right_weighted = 0, 0.0, 0.0, 0.0

    # For the partly wet segment beyond each side's outermost wet ordinate: width and wetted length per
    # metre of depth, and the Manning's n**1.5 of its inner end. A wall is a segment of no width.
    left_ratio = right_ratio = 0.0
    left_length_per_depth = right_length_per_depth = 1.0
    left_n_15 = right_n_15 = 0.0
    left_moved = right_moved = True

    while True:
        # Once the water is above this level, each side's edge moves past any ground no higher than it
        while right < n - 1 and elevations[right + 1] <= level:
            rise = elevations[right + 1] - elevations[right]
            length = _length(spacing, rise)
            right_count += 1
            right_mid += elevations[right] + rise / 2
            right_length += length
            right_weighted += length * _n_15(mannings_n[right])
            right += 1
            right_moved = True
        while left > 0 and elevations[left - 1] <= level:
            rise = elevations[left - 1] - elevations[left]
            length = _length(spacing, rise)
            left_count += 1
            left_mid += elevations[left] + rise / 2
            left_length += length
            left_weighted += length * _n_15(mannings_n[left])
            left -= 1
            left_moved = True

        if right_moved:
            right_n_15 = _n_15(mannings_n[right])
            if right < n - 1:
                rise = elevations[right + 1] - elevations[right]
                right_ratio = spacing / rise
                right_length_per_depth = _length(spacing, rise) / rise
            else:
                right_ratio = 0.0
                right_length_per_depth = 1.0
            right_moved = False
        if left_moved:
            left_n_15 = _n_15(mannings_n[left])
            if left > 0:
                rise = elevations[left - 1] - elevations[left]
                left_ratio = spacing / rise
                left_length_per_depth = _length(spacing, rise) / rise
            else:
                left_ratio = 0.0
                left_length_per_depth = 1.0
            left_moved = False

        right_depth = level - elevations[right]
        left_depth = level - elevations[left]
        area = (spacing * ((right_count + left_count) * level - right_mid - left_mid)
                + 0.5 * (right_ratio * right_depth * right_depth + left_ratio * left_depth * left_depth))
        top_width = spacing * (right_count + left_count) + right_ratio * right_depth + left_ratio * left_depth
        half_dwidth = 0.5 * (right_ratio + left_ratio)
        perimeter = (right_length + left_length
                     + right_length_per_depth * right_depth + left_length_per_depth * left_depth)
        dperimeter = right_length_per_depth + left_length_per_depth
        weighted = (right_weighted + left_weighted + right_length_per_depth * right_n_15 * right_depth
                    + left_length_per_depth * left_n_15 * left_depth)
        dweighted = right_length_per_depth * right_n_15 + left_length_per_depth * left_n_15

        # The interval ends at the next blocking ordinate, or never once both sides reach their walls
        next_right = elevations[right + 1] if right < n - 1 else np.inf
        next_left = elevations[left - 1] if left > 0 else np.inf
        end = min(next_right, next_left)
        height = end - level

        k3_start = _cubed_conveyance(area, weighted)
        if height < np.inf:
            k3_end = _cubed_conveyance(area + height * (top_width + height * half_dwidth), weighted + height * dweighted)
        elif top_width > 0.0 or half_dwidth > 0.0:
            k3_end = np.inf
        else:
            k3_end = 0.0  # no width between the walls, so this section never carries any water
        k3_max = max(k3_max, k3_start, k3_end)

        row = table[min(count, rows - 1)]
        row[_START] = level
        row[_END] = end
        row[_AREA] = area
        row[_TOP_WIDTH] = top_width
        row[_HALF_DWIDTH] = half_dwidth
        row[_PERIMETER] = perimeter
        row[_DPERIMETER] = dperimeter
        row[_WEIGHTED] = weighted
        row[_DWEIGHTED] = dweighted
        row[_K3_START] = k3_start
        row[_K3_END] = k3_end
        row[_K3_MAX] = k3_max
        count += 1

        if end == np.inf or k3_max >= stop_k3 or end >= max_wse:
            return count
        level = end


@njit(cache=True, error_model="numpy")
def _log_cubed_conveyance(row, u):
    """log(A**5 / W**2) at height u above the start of an interval, and its derivative."""
    area = row[_AREA] + u * (row[_TOP_WIDTH] + u * row[_HALF_DWIDTH])
    weighted = row[_WEIGHTED] + u * row[_DWEIGHTED]
    top_width = row[_TOP_WIDTH] + 2.0 * u * row[_HALF_DWIDTH]
    return 5.0 * math.log(area) - 2.0 * math.log(weighted), 5.0 * top_width / area - 2.0 * row[_DWEIGHTED] / weighted


@njit(cache=True, error_model="numpy")
def _newton_or_bisect(u, step, lo, hi):
    """The next u after a Newton step from u, bisecting if the step leaves the bracket [lo, hi], and whether that
    is close enough. Newton's method converges quadratically, so a step under 1e-9 m leaves an error of about
    1e-18 m; a bisection needs the bracket itself small. A step too small to move u stays in the bracket."""
    newton = u - step
    if lo <= newton <= hi:
        return newton, abs(step) <= 1e-9 * (1.0 + abs(u))
    middle = 0.5 * (lo + hi)
    return middle, hi - lo <= 1e-12 * (1.0 + abs(middle))


@njit(cache=True, error_model="numpy")
def _wse_in_interval(row, target):
    """The lowest water surface elevation in an interval whose conveyance reaches target, given that the
    interval's highest conveyance does but nothing below the interval does."""
    target_k3 = target ** 3
    if row[_K3_START] >= target_k3:
        # Conveyance jumped past the target at the start of the interval (water spilling into low ground)
        return row[_START]
    log_target = math.log(target_k3)

    # Conveyance crosses the target once in the interval (see the notes above the table columns)
    lo = 0.0
    hi = row[_END] - row[_START]
    if hi == np.inf:
        hi = 1.0
        while _log_cubed_conveyance(row, hi)[0] < log_target:
            hi *= 2.0

    # Newton's method, falling back to bisection whenever a step would leave the bracket
    u = hi
    for _ in range(100):
        log_k3, slope = _log_cubed_conveyance(row, u)
        error = log_k3 - log_target
        if error == 0.0:
            break
        if error > 0.0:
            hi = u
        else:
            lo = u
        u, converged = _newton_or_bisect(u, error / slope, lo, hi)
        if converged:
            break
    return row[_START] + u


@njit(cache=True, error_model="numpy")
def wse_for_conveyance(elevations, mannings_n, spacing, target):
    """The lowest water surface elevation whose conveyance reaches target.

    This raises the water surface only as far as the answer, so it costs about as much as one
    conveyance calculation. Returns the stream cell's elevation for a target of zero or less, and NaN if
    the cross section can never carry the target. Where conveyance jumps past the target (water spilling
    over a high point into low ground beyond it), this returns the elevation of that high point.
    """
    if target <= 0.0:
        return float(elevations[elevations.size // 2])
    last = np.empty((1, _COLUMNS))
    _raise_water_surface(elevations, mannings_n, spacing, target ** 3, np.inf, last)
    if not last[0, _K3_MAX] >= target ** 3:
        return np.nan
    return _wse_in_interval(last[0], target)


@njit(cache=True, error_model="numpy")
def build_conveyance_table(elevations, mannings_n, spacing, max_wse=np.inf):
    """Tabulate conveyance at every water surface elevation up to max_wse (see the notes above)."""
    table = np.empty((elevations.size + 1, _COLUMNS))
    count = _raise_water_surface(elevations, mannings_n, spacing, np.inf, max_wse, table)
    return table[:count]


@njit(cache=True, error_model="numpy")
def _interval_index(table, wse):
    """The last row of a table (of either kind) whose interval starts below wse, for wse above the first."""
    lo, hi = 0, table.shape[0] - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if table[mid, _START] < wse:
            lo = mid
        else:
            hi = mid - 1
    return lo


@njit(cache=True, error_model="numpy")
def _first_row_reaching(table, column, target):
    """The first row of a table whose column, a running maximum, reaches target, or the number of rows."""
    lo, hi = 0, table.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if table[mid, column] >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True, error_model="numpy")
def _block_geometry(row, block, u):
    """Area, wetted perimeter, top width and sum(P_i * n_i**1.5) at height u above the start of a row's interval,
    from the coefficients in columns block + _AREA to block + _DWEIGHTED."""
    area = row[block + _AREA] + u * (row[block + _TOP_WIDTH] + u * row[block + _HALF_DWIDTH])
    perimeter = row[block + _PERIMETER] + u * row[block + _DPERIMETER]
    top_width = row[block + _TOP_WIDTH] + 2.0 * u * row[block + _HALF_DWIDTH]
    weighted = row[block + _WEIGHTED] + u * row[block + _DWEIGHTED]
    return area, perimeter, top_width, weighted


@njit(cache=True, error_model="numpy")
def table_geometry(table, wse):
    """Area, wetted perimeter, top width and sum(P_i * n_i**1.5) at a water surface elevation, from a table.

    Zero at or below the stream cell, and NaN above the table's last interval.
    """
    if not wse > table[0, _START]:
        return 0.0, 0.0, 0.0, 0.0
    if wse > table[table.shape[0] - 1, _END]:
        return np.nan, np.nan, np.nan, np.nan
    row = table[_interval_index(table, wse)]
    return _block_geometry(row, 0, wse - row[_START])


@njit(cache=True, error_model="numpy")
def table_conveyance(table, wse):
    """Conveyance at a water surface elevation, from a table."""
    area, _, _, weighted = table_geometry(table, wse)
    return _conveyance(area, weighted)


@njit(cache=True, error_model="numpy")
def table_wse_for_conveyance(table, target):
    """Like wse_for_conveyance, from a table. Also NaN when the answer is above the table's last interval."""
    if target <= 0.0:
        return table[0, _START]
    row = _first_row_reaching(table, _K3_MAX, target ** 3)
    if row == table.shape[0]:
        return np.nan
    return _wse_in_interval(table[row], target)


# Cross sections divided at the banks. The banks are vertical lines left_bank and right_bank metres either side of
# the centre, measured along the cross section. They divide it into a left overbank, the channel between them and
# a right overbank, and the conveyance is the sum of the three subsections' conveyances, each A**(5/3) / W**(2/3)
# over its own water and wetted ground (the divided channel method, as in cross_section._compound_section_conveyance).
# The dividing lines add no wetted perimeter.
#
# The banks don't change where the water is, so the subsections' geometry adds up to wetted_geometry's. A segment
# that a bank crosses is divided at the bank, and both parts keep the segment's Manning's n. An end wall belongs to
# the overbank on its side. A bank at or beyond the end of its side leaves that side all channel, walls included, and
# so does a bank that is negative or NaN, which XSection uses for a bank that hasn't been found.


@njit(cache=True, error_model="numpy")
def _bank_position(bank, spacing, segments):
    """The segment of a side of `segments` segments that a bank `bank` metres from the centre is in, and how far
    along that segment it is. A bank that doesn't divide the side gives segments + 1, putting the side in the channel."""
    position = bank / spacing
    if not 0.0 <= position < segments:
        return segments + 1, 0.0
    segment = int(position)
    return segment, position - segment


@njit(cache=True, error_model="numpy")
def _divided_side_geometry(elevations, mannings_n, center, step, spacing, wse, bank):
    """_side_geometry, divided at a bank: area, wetted perimeter, top width and sum(P_i * n_i**1.5) of the part
    in the channel, then of the part in the overbank."""
    end = elevations.size - 1 if step > 0 else 0
    bank_segment, bank_fraction = _bank_position(bank, spacing, abs(end - center))
    channel_area = channel_perimeter = channel_top_width = channel_weighted = 0.0
    overbank_area = overbank_perimeter = overbank_top_width = overbank_weighted = 0.0
    segment = 0
    j = center
    while j != end:
        z_in = elevations[j]
        z_out = elevations[j + step]
        rise = z_out - z_in
        depth_in = wse - z_in
        full = wse > z_out
        if full:
            wet = 1.0  # the fraction of the segment under water
            depth_out = wse - z_out
        else:
            wet = depth_in / rise  # the water meets the ground part way along the segment
            depth_out = 0.0
        length = _length(spacing, rise)
        n_15 = _n_15(mannings_n[j])

        if segment < bank_segment or (segment == bank_segment and wet <= bank_fraction):
            channel_area += spacing * wet * (depth_in + depth_out) / 2
            channel_perimeter += wet * length
            channel_weighted += wet * length * n_15
            channel_top_width += spacing * wet
        elif segment > bank_segment or bank_fraction == 0.0:
            overbank_area += spacing * wet * (depth_in + depth_out) / 2
            overbank_perimeter += wet * length
            overbank_weighted += wet * length * n_15
            overbank_top_width += spacing * wet
        else:
            # The bank divides the wet part of the segment
            depth_bank = depth_in - bank_fraction * rise
            outer = wet - bank_fraction
            channel_area += spacing * bank_fraction * (depth_in + depth_bank) / 2
            channel_perimeter += bank_fraction * length
            channel_weighted += bank_fraction * length * n_15
            channel_top_width += spacing * bank_fraction
            overbank_area += spacing * outer * (depth_bank + depth_out) / 2
            overbank_perimeter += outer * length
            overbank_weighted += outer * length * n_15
            overbank_top_width += spacing * outer

        if not full:
            return (channel_area, channel_perimeter, channel_top_width, channel_weighted,
                    overbank_area, overbank_perimeter, overbank_top_width, overbank_weighted)
        j += step
        segment += 1

    # The water reaches the wall at the end, in the overbank if the bank is inside the section
    depth_end = wse - elevations[end]
    n_15 = _n_15(mannings_n[end])
    if segment > bank_segment:
        overbank_perimeter += depth_end
        overbank_weighted += depth_end * n_15
    else:
        channel_perimeter += depth_end
        channel_weighted += depth_end * n_15
    return (channel_area, channel_perimeter, channel_top_width, channel_weighted,
            overbank_area, overbank_perimeter, overbank_top_width, overbank_weighted)


@njit(cache=True, error_model="numpy")
def compound_wetted_geometry(elevations, mannings_n, spacing, left_bank, right_bank, wse):
    """wetted_geometry for each subsection of a cross section divided at its banks (see the notes above), as
    three tuples: the left overbank, the channel, and the right overbank."""
    center = elevations.size // 2
    if not wse > elevations[center]:
        dry = (0.0, 0.0, 0.0, 0.0)
        return dry, dry, dry
    left = _divided_side_geometry(elevations, mannings_n, center, -1, spacing, wse, left_bank)
    right = _divided_side_geometry(elevations, mannings_n, center, 1, spacing, wse, right_bank)
    channel = (left[0] + right[0], left[1] + right[1], left[2] + right[2], left[3] + right[3])
    return (left[4], left[5], left[6], left[7]), channel, (right[4], right[5], right[6], right[7])


@njit(cache=True, error_model="numpy")
def compound_conveyance(elevations, mannings_n, spacing, left_bank, right_bank, wse):
    """The conveyance of a cross section divided at its banks: the sum of its subsections' conveyances."""
    left, channel, right = compound_wetted_geometry(elevations, mannings_n, spacing, left_bank, right_bank, wse)
    return _conveyance(left[0], left[3]) + _conveyance(channel[0], channel[3]) + _conveyance(right[0], right[3])


# A compound table has a row per interval like a conveyance table, with each subsection's coefficients in turn,
# at these offsets from a conveyance table's columns (so the right overbank's area is in column
# _RIGHT_OVERBANK + _AREA). The total conveyance follows, not cubed, since the subsections' conveyances add: at the
# start and end of the interval, and the highest at or below its end.
#
# Within an interval, each subsection's conveyance is convex, which is more than being highest at an end:
#   K'' / K = 5/3 * A'' / A + 10/9 * (A' / A - W' / W)**2 >= 0,  since A'' = 2 * HALF_DWIDTH >= 0 and W'' = 0.
# So the total is convex too. It is highest at one of the ends, and crosses any level between its start and its end
# only once, as in a conveyance table.
_LEFT_OVERBANK, _CHANNEL, _RIGHT_OVERBANK = 0, 7, 14
_K_START, _K_END, _K_MAX = 23, 24, 25
_COMPOUND_COLUMNS = 26


@njit(cache=True, error_model="numpy")
def _edge_constants(elevations, mannings_n, spacing, step, pos):
    """For the partly wet segment beyond ordinate pos: its width and wetted length per metre of depth, and the
    Manning's n**1.5 of its inner end. At the end of the cross section, this is the wall, of no width."""
    end = elevations.size - 1 if step > 0 else 0
    if pos == end:
        return 0.0, 1.0, _n_15(float(mannings_n[end]))
    rise = float(elevations[pos + step]) - float(elevations[pos])
    return spacing / rise, _length(spacing, rise) / rise, _n_15(float(mannings_n[pos]))


@njit(cache=True, error_model="numpy")
def _next_node(elevations, step, pos, segment, overbank, bank_segment, bank_fraction):
    """The elevation of the next node beyond a side's outermost node: the bank if it divides the segment beyond pos
    and the water's edge isn't past it, or the next ordinate, or infinity at the wall."""
    end = elevations.size - 1 if step > 0 else 0
    if pos == end:
        return np.inf
    z_next = float(elevations[pos + step])
    if segment == bank_segment and bank_fraction > 0.0 and not overbank:
        z = float(elevations[pos])
        return z + bank_fraction * (z_next - z)
    return z_next


@njit(cache=True, error_model="numpy")
def _piece(elevations, mannings_n, spacing, step, pos, fraction, inner, node):
    """The piece of ground from elevation inner to a node, covering a fraction of the segment beyond pos: its
    width, width times mid elevation, length, and n-weighted length."""
    width = fraction * spacing
    length = fraction * _length(spacing, float(elevations[pos + step]) - float(elevations[pos]))
    return width, width * (inner + node) / 2, length, length * _n_15(mannings_n[pos])


_NO_EDGE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@njit(cache=True, error_model="numpy")
def _edge(level, inner, ratio, per_depth, n_15):
    """The partly wet piece of ground beyond a side's outermost node, with the water at level: its area, top width,
    half the rate its top width grows, wetted perimeter and its rate, and n-weighted perimeter and its rate."""
    depth = level - inner
    weighted_per_depth = per_depth * n_15
    return (0.5 * ratio * depth * depth, ratio * depth, 0.5 * ratio, per_depth * depth, per_depth,
            weighted_per_depth * depth, weighted_per_depth)


@njit(cache=True, error_model="numpy")
def _add_edges(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5], a[6] + b[6]


@njit(cache=True, error_model="numpy")
def _write_block(row, block, level, width, width_mid, length, weighted, edge):
    """Write a subsection's coefficients into a compound table row, from the totals over its fully wet pieces and
    the sum of its partly wet pieces' _edge terms."""
    row[block + _AREA] = level * width - width_mid + edge[0]
    row[block + _TOP_WIDTH] = width + edge[1]
    row[block + _HALF_DWIDTH] = edge[2]
    row[block + _PERIMETER] = length + edge[3]
    row[block + _DPERIMETER] = edge[4]
    row[block + _WEIGHTED] = weighted + edge[5]
    row[block + _DWEIGHTED] = edge[6]


@njit(cache=True, error_model="numpy")
def _block_conveyance(row, block, u):
    """A subsection's conveyance at height u above the start of a compound table row's interval."""
    area = row[block + _AREA] + u * (row[block + _TOP_WIDTH] + u * row[block + _HALF_DWIDTH])
    return _conveyance(area, row[block + _WEIGHTED] + u * row[block + _DWEIGHTED])


@njit(cache=True, error_model="numpy")
def _compound_row_conveyance(row, u):
    """Total conveyance at height u above the start of a compound table row's interval."""
    return (_block_conveyance(row, _LEFT_OVERBANK, u) + _block_conveyance(row, _CHANNEL, u)
            + _block_conveyance(row, _RIGHT_OVERBANK, u))


@njit(cache=True, error_model="numpy")
def _raise_compound_water_surface(elevations, mannings_n, spacing, left_bank, right_bank, stop_k, max_wse, table):
    """_raise_water_surface for a cross section divided at its banks, writing interval i to row
    min(i, len(table) - 1) of a compound table, until an interval's total conveyance reaches stop_k or the interval
    reaches max_wse. Returns the number of intervals.

    Each side is a sequence of nodes: its ordinates, and its bank if the bank divides a segment. Each fully wet
    piece of ground between two nodes is either in the channel or in its side's overbank.
    """
    n = elevations.size
    center = n // 2
    rows = table.shape[0]
    level = float(elevations[center])
    k_end = 0.0
    k_max = 0.0
    count = 0

    # For each side: the outermost wet ordinate, the number of segments inside it, and whether the water's edge is
    # past the bank (the bank's segment and how far along it is are fixed). Then, for the partly wet piece of
    # ground beyond the outermost node, the elevation it starts at and the fraction of its segment it covers, and
    # its width and wetted length per metre of depth and its Manning's n**1.5. Then totals over the fully wet pieces
    # in the channel and in the overbank: their width, width times mid elevation, length, and n-weighted length.
    # And the elevation of the next node.
    right_bank_segment, right_bank_fraction = _bank_position(right_bank, spacing, n - 1 - center)
    right, right_segment = center, 0
    right_overbank = right_bank_segment == 0 and right_bank_fraction == 0.0
    right_inner, right_piece = level, 1.0
    right_ratio, right_per_depth, right_n_15 = _edge_constants(elevations, mannings_n, spacing, 1, center)
    right_channel_width = right_channel_mid = right_channel_length = right_channel_weighted = 0.0
    right_overbank_width = right_overbank_mid = right_overbank_length = right_overbank_weighted = 0.0
    next_right = _next_node(elevations, 1, right, right_segment, right_overbank, right_bank_segment,
                            right_bank_fraction)

    left_bank_segment, left_bank_fraction = _bank_position(left_bank, spacing, center)
    left, left_segment = center, 0
    left_overbank = left_bank_segment == 0 and left_bank_fraction == 0.0
    left_inner, left_piece = level, 1.0
    left_ratio, left_per_depth, left_n_15 = _edge_constants(elevations, mannings_n, spacing, -1, center)
    left_channel_width = left_channel_mid = left_channel_length = left_channel_weighted = 0.0
    left_overbank_width = left_overbank_mid = left_overbank_length = left_overbank_weighted = 0.0
    next_left = _next_node(elevations, -1, left, left_segment, left_overbank, left_bank_segment, left_bank_fraction)

    while True:
        # Once the water is above this level, each side's edge moves past any node no higher than it. The ground
        # from the start of the partly wet piece to the node is then all under water.
        right_passed = 0
        while next_right <= level:
            at_bank = right_segment == right_bank_segment and right_bank_fraction > 0.0 and not right_overbank
            width, width_mid, length, weighted = _piece(elevations, mannings_n, spacing, 1, right,
                                                        right_bank_fraction if at_bank else right_piece,
                                                        right_inner, next_right)
            if right_overbank:
                right_overbank_width += width
                right_overbank_mid += width_mid
                right_overbank_length += length
                right_overbank_weighted += weighted
            else:
                right_channel_width += width
                right_channel_mid += width_mid
                right_channel_length += length
                right_channel_weighted += weighted
            right_passed += 1
            right_inner = next_right
            if at_bank:
                right_overbank = True
                right_piece = 1.0 - right_bank_fraction
            else:
                right += 1
                right_segment += 1
                right_piece = 1.0
                right_overbank = right_overbank or (right_segment == right_bank_segment and right_bank_fraction == 0.0)
            next_right = _next_node(elevations, 1, right, right_segment, right_overbank, right_bank_segment,
                                    right_bank_fraction)
        if right_passed > 0:
            right_ratio, right_per_depth, right_n_15 = _edge_constants(elevations, mannings_n, spacing, 1, right)

        left_passed = 0
        while next_left <= level:
            at_bank = left_segment == left_bank_segment and left_bank_fraction > 0.0 and not left_overbank
            width, width_mid, length, weighted = _piece(elevations, mannings_n, spacing, -1, left,
                                                        left_bank_fraction if at_bank else left_piece,
                                                        left_inner, next_left)
            if left_overbank:
                left_overbank_width += width
                left_overbank_mid += width_mid
                left_overbank_length += length
                left_overbank_weighted += weighted
            else:
                left_channel_width += width
                left_channel_mid += width_mid
                left_channel_length += length
                left_channel_weighted += weighted
            left_passed += 1
            left_inner = next_left
            if at_bank:
                left_overbank = True
                left_piece = 1.0 - left_bank_fraction
            else:
                left -= 1
                left_segment += 1
                left_piece = 1.0
                left_overbank = left_overbank or (left_segment == left_bank_segment and left_bank_fraction == 0.0)
            next_left = _next_node(elevations, -1, left, left_segment, left_overbank, left_bank_segment,
                                   left_bank_fraction)
        if left_passed > 0:
            left_ratio, left_per_depth, left_n_15 = _edge_constants(elevations, mannings_n, spacing, -1, left)

        end = min(next_right, next_left)
        height = end - level

        # Each side's partly wet piece belongs to the subsection its water edge is in
        right_edge = _edge(level, right_inner, right_ratio, right_per_depth, right_n_15)
        left_edge = _edge(level, left_inner, left_ratio, left_per_depth, left_n_15)
        row = table[min(count, rows - 1)]
        row[_START] = level
        row[_END] = end
        _write_block(row, _LEFT_OVERBANK, level, left_overbank_width, left_overbank_mid, left_overbank_length,
                     left_overbank_weighted, left_edge if left_overbank else _NO_EDGE)
        _write_block(row, _CHANNEL, level, left_channel_width + right_channel_width,
                     left_channel_mid + right_channel_mid, left_channel_length + right_channel_length,
                     left_channel_weighted + right_channel_weighted,
                     _add_edges(_NO_EDGE if left_overbank else left_edge, _NO_EDGE if right_overbank else right_edge))
        _write_block(row, _RIGHT_OVERBANK, level, right_overbank_width, right_overbank_mid, right_overbank_length,
                     right_overbank_weighted, right_edge if right_overbank else _NO_EDGE)

        # Conveyance only jumps where a side's edge passed more than one node, spilling past the node that stopped
        # it onto ground no higher. Otherwise the interval starts with the conveyance the last one ended with.
        count += 1
        if count > 1 and right_passed <= 1 and left_passed <= 1:
            k_start = k_end
        else:
            k_start = _compound_row_conveyance(row, 0.0)
        if height < np.inf:
            k_end = _compound_row_conveyance(row, height)
        elif row[_LEFT_OVERBANK + _TOP_WIDTH] + row[_CHANNEL + _TOP_WIDTH] + row[_RIGHT_OVERBANK + _TOP_WIDTH] > 0.0:
            k_end = np.inf
        else:
            k_end = k_start  # no width between the walls, so this section never carries any water
        k_max = max(k_max, k_start, k_end)
        row[_K_START] = k_start
        row[_K_END] = k_end
        row[_K_MAX] = k_max
        if end == np.inf or k_max >= stop_k or end >= max_wse:
            return count
        level = end


@njit(cache=True, error_model="numpy")
def _compound_row_conveyance_and_slope(row, u):
    """_compound_row_conveyance, and its derivative with respect to u."""
    total = 0.0
    slope = 0.0
    for block in (_LEFT_OVERBANK, _CHANNEL, _RIGHT_OVERBANK):
        area = row[block + _AREA] + u * (row[block + _TOP_WIDTH] + u * row[block + _HALF_DWIDTH])
        if area > 0.0:
            weighted = row[block + _WEIGHTED] + u * row[block + _DWEIGHTED]
            top_width = row[block + _TOP_WIDTH] + 2.0 * u * row[block + _HALF_DWIDTH]
            k = _conveyance(area, weighted)
            total += k
            slope += k * (5.0 * top_width / area - 2.0 * row[block + _DWEIGHTED] / weighted)
    return total, slope / 3.0


@njit(cache=True, error_model="numpy")
def _compound_wse_in_interval(row, target):
    """_wse_in_interval for a compound table row."""
    if row[_K_START] >= target:
        return row[_START]

    lo = 0.0
    hi = row[_END] - row[_START]
    if hi == np.inf:
        hi = 1.0
        while _compound_row_conveyance(row, hi) < target:
            lo = hi
            hi *= 2.0
        u = hi
    else:
        # Start where the chord between the interval's ends crosses the target, which is at or below the answer
        # as conveyance is convex in the interval. Newton's method then steps past the answer, and comes back down
        # to it. Rounding could put a step outside the bracket, so bisect then.
        u = hi * (target - row[_K_START]) / (row[_K_END] - row[_K_START])
    for _ in range(100):
        k, slope = _compound_row_conveyance_and_slope(row, u)
        error = k - target
        if error == 0.0:
            break
        if error > 0.0:
            hi = u
        else:
            lo = u
        u, converged = _newton_or_bisect(u, error / slope, lo, hi)
        if converged:
            break
    return row[_START] + u


@njit(cache=True, error_model="numpy")
def wse_for_compound_conveyance(elevations, mannings_n, spacing, left_bank, right_bank, target):
    """wse_for_conveyance for a cross section divided at its banks."""
    if target <= 0.0:
        return float(elevations[elevations.size // 2])
    last = np.empty((1, _COMPOUND_COLUMNS))
    _raise_compound_water_surface(elevations, mannings_n, spacing, left_bank, right_bank, target, np.inf, last)
    if not last[0, _K_MAX] >= target:
        return np.nan
    return _compound_wse_in_interval(last[0], target)


@njit(cache=True, error_model="numpy")
def build_compound_conveyance_table(elevations, mannings_n, spacing, left_bank, right_bank, max_wse=np.inf):
    """build_conveyance_table for a cross section divided at its banks (see the notes above)."""
    # An interval ends at each node, and there are at most two banks as well as the ordinates
    table = np.empty((elevations.size + 2, _COMPOUND_COLUMNS))
    count = _raise_compound_water_surface(elevations, mannings_n, spacing, left_bank, right_bank, np.inf, max_wse, table)
    return table[:count]


@njit(cache=True, error_model="numpy")
def compound_table_geometry(table, wse):
    """compound_wetted_geometry from a compound table. NaN above the table's last interval."""
    if not wse > table[0, _START]:
        dry = (0.0, 0.0, 0.0, 0.0)
        return dry, dry, dry
    if wse > table[table.shape[0] - 1, _END]:
        above = (np.nan, np.nan, np.nan, np.nan)
        return above, above, above
    row = table[_interval_index(table, wse)]
    u = wse - row[_START]
    return (_block_geometry(row, _LEFT_OVERBANK, u), _block_geometry(row, _CHANNEL, u),
            _block_geometry(row, _RIGHT_OVERBANK, u))


@njit(cache=True, error_model="numpy")
def compound_table_conveyance(table, wse):
    """compound_conveyance from a compound table. NaN above the table's last interval."""
    if not wse > table[0, _START]:
        return 0.0
    if wse > table[table.shape[0] - 1, _END]:
        return np.nan
    row = table[_interval_index(table, wse)]
    return _compound_row_conveyance(row, wse - row[_START])


@njit(cache=True, error_model="numpy")
def compound_table_wse_for_conveyance(table, target):
    """wse_for_compound_conveyance from a compound table. Also NaN when the answer is above its last interval."""
    if target <= 0.0:
        return table[0, _START]
    row = _first_row_reaching(table, _K_MAX, target)
    if row == table.shape[0]:
        return np.nan
    return _compound_wse_in_interval(table[row], target)


def hydraulic_geometry(xs: XSection, *, wse: float | None = None, depth: float | None = None) -> HydraulicGeometry:
    """Wetted geometry at a water surface elevation, or at a depth above the stream cell. Zero when dry.

    The banks don't change the geometry, so this is the whole cross section's either way. See compound_geometry
    for each subsection's.
    """
    wse = _resolve_wse(float(xs.elevations[xs.elevations.size // 2]), wse, depth)
    return _to_hydraulic_geometry(*wetted_geometry(xs.elevations, xs.mannings_n, xs.ordinate_distance, wse))


def compound_geometry(xs: XSection, *, wse: float | None = None, depth: float | None = None) -> CompoundGeometry:
    """Wetted geometry of the left overbank, the channel and the right overbank. Without banks, it's all channel."""
    wse = _resolve_wse(float(xs.elevations[xs.elevations.size // 2]), wse, depth)
    return _to_compound_geometry(compound_wetted_geometry(xs.elevations, xs.mannings_n, xs.ordinate_distance,
                                                          *_bank_distances(xs), wse))


def discharge(xs: XSection, slope: float, *, wse: float | None = None, depth: float | None = None) -> float:
    """Manning's discharge in m^3/s at a water surface elevation, or at a depth above the stream cell.

    With banks, this is the sum of the subsections' discharges, which is not Manning's equation applied to the
    whole section's hydraulic_geometry.
    """
    wse = _resolve_wse(float(xs.elevations[xs.elevations.size // 2]), wse, depth)
    if _has_banks(xs):
        k = compound_conveyance(xs.elevations, xs.mannings_n, xs.ordinate_distance, *_bank_distances(xs), wse)
    else:
        k = conveyance(xs.elevations, xs.mannings_n, xs.ordinate_distance, wse)
    return k * _sqrt_slope(slope)


def wse_for_discharge(xs: XSection, q: float, slope: float) -> float:
    """The lowest water surface elevation at which the cross section carries q m^3/s.

    Discharge doesn't always rise with the water level: when water spreads onto flat ground, the wetted
    perimeter can grow faster than the area. Taking the lowest match gives the level that a rising water
    surface reaches first. See wse_for_conveyance for the other special cases.
    """
    target = q / _sqrt_slope(slope)
    if _has_banks(xs):
        return wse_for_compound_conveyance(xs.elevations, xs.mannings_n, xs.ordinate_distance,
                                           *_bank_distances(xs), target)
    return wse_for_conveyance(xs.elevations, xs.mannings_n, xs.ordinate_distance, target)


class ConveyanceTable:
    """A cross section's conveyance at every water surface elevation, for many lookups on one section.

    Building the table costs about as much as raising the water surface to the top of it once. After that,
    the discharge at a water surface elevation, and the water surface elevation for a discharge, each take a
    binary search and at most a few Newton steps, with no loss of accuracy. The table doesn't depend on
    slope, so one table serves every slope. Limit max_depth to the deepest water you need, to build less.
    A cross section with banks gets a compound table, divided at its banks.
    """

    def __init__(self, xs: XSection, max_depth: float | None = None):
        self.center_elevation = float(xs.elevations[xs.elevations.size // 2])
        max_wse = np.inf if max_depth is None else self.center_elevation + max_depth
        self.compound = _has_banks(xs)
        if self.compound:
            self.table = build_compound_conveyance_table(xs.elevations, xs.mannings_n, xs.ordinate_distance,
                                                         *_bank_distances(xs), max_wse)
        else:
            self.table = build_conveyance_table(xs.elevations, xs.mannings_n, xs.ordinate_distance, max_wse)

    def geometry(self, *, wse: float | None = None, depth: float | None = None) -> HydraulicGeometry:
        wse = _resolve_wse(self.center_elevation, wse, depth)
        if self.compound:
            return _to_hydraulic_geometry(*(sum(values) for values in zip(*compound_table_geometry(self.table, wse))))
        return _to_hydraulic_geometry(*table_geometry(self.table, wse))

    def compound_geometry(self, *, wse: float | None = None, depth: float | None = None) -> CompoundGeometry:
        wse = _resolve_wse(self.center_elevation, wse, depth)
        if self.compound:
            return _to_compound_geometry(compound_table_geometry(self.table, wse))
        whole = table_geometry(self.table, wse)
        none = (0.0, 0.0, 0.0, 0.0) if whole[1] >= 0.0 else whole  # NaN above the table
        return _to_compound_geometry((none, whole, none))

    def discharge(self, slope: float, *, wse: float | None = None, depth: float | None = None) -> float:
        wse = _resolve_wse(self.center_elevation, wse, depth)
        if self.compound:
            return compound_table_conveyance(self.table, wse) * _sqrt_slope(slope)
        return table_conveyance(self.table, wse) * _sqrt_slope(slope)

    def wse_for_discharge(self, q: float, slope: float) -> float:
        """The lowest water surface elevation at which the cross section carries q m^3/s (see wse_for_discharge)."""
        target = q / _sqrt_slope(slope)
        if self.compound:
            return compound_table_wse_for_conveyance(self.table, target)
        return table_wse_for_conveyance(self.table, target)


def _bank_distances(xs: XSection) -> tuple[float, float]:
    return float(xs.left_bank_distance), float(xs.right_bank_distance)


def _has_banks(xs: XSection) -> bool:
    """Whether either bank is set: XSection uses -1 for a bank that hasn't been found."""
    left, right = _bank_distances(xs)
    return left >= 0.0 or right >= 0.0


def _to_hydraulic_geometry(area: float, perimeter: float, top_width: float, weighted_perimeter: float) -> HydraulicGeometry:
    if perimeter > 0.0:
        return HydraulicGeometry(area, perimeter, area / perimeter, top_width, (weighted_perimeter / perimeter) ** (2 / 3))
    if perimeter == 0.0:
        return HydraulicGeometry(0.0, 0.0, 0.0, 0.0, 0.0)  # dry
    return HydraulicGeometry(math.nan, math.nan, math.nan, math.nan, math.nan)  # above a table's last interval


def _to_compound_geometry(subsections) -> CompoundGeometry:
    return CompoundGeometry(*(_to_hydraulic_geometry(*subsection) for subsection in subsections))


def _resolve_wse(center_elevation: float, wse: float | None, depth: float | None) -> float:
    if (wse is None) == (depth is None):
        raise TypeError("Give exactly one of wse or depth.")
    return float(wse) if depth is None else center_elevation + float(depth)


def _sqrt_slope(slope: float) -> float:
    if not slope > 0.0:
        raise ValueError(f"slope must be positive, not {slope!r}.")
    return math.sqrt(slope)
