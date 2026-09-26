"""A cross section's rating curve: the flow increments legacy ARC wrote to its VDT database.

The hydraulics are the cross section's profile's (arc.hydraulics.hydraulic_profile): a carved channel's exact shape
where it has one (arc.bathymetry.carve_channel), and otherwise its ordinates. The water surface that carries a stream
cell's maximum flow is the lowest at which the discharge reaches it. The rating curve is then taken at equal steps of
depth from the stream cell up to that water surface, as legacy ARC's flood_increments did: at each, the discharge,
the velocity (discharge over area), the top width, the water surface elevation and the wetted perimeter. A cross
section with banks has its discharge summed over its channel and overbanks. Manning's n can vary with the depth
(arc.hydraulics.DepthRoughness), and the square root of the slope is multiplied by slope_factor, legacy's
slope_adjustment_factor.

Where discharge falls as the water rises, as it can where water spreads onto flat ground, legacy looked a centimetre
at a time up to the next increment for a water surface whose discharge was higher than the last increment's, and if
there wasn't one, repeated the last increment. That is kept, and so is its lowering of the first increment's discharge
to just below the baseflow when it was above it.

The maximum flow isn't always carried at the top: where the water spills over a bank into lower ground, the discharge
jumps, and the lowest water surface carrying the maximum flow is the bank's top, where just below it the channel
carries less. As in legacy, a cross section is only given a rating curve (it's "acceptable") if the discharge at its
top is within half of the maximum flow either way, and no increment may carry more than 1% more than that discharge.

Numerical differences
---------------------
- The maximum flow's water surface is solved exactly, where legacy stepped to it (by 0.5 m, 0.05 m and then 0.01 m,
  interpolating the last step) and also ran Brent's method, keeping whichever came closer. So the last increment
  carries the maximum flow unless the water spills there. Legacy's slope search, which it ran when its steps missed
  the maximum flow by more than half, isn't here, so the slope is the cell's slope. Where legacy's steps found the
  maximum flow's water surface, its slope search only changed the slope when the steps were at least a centimetre
  off; where the water spills, changing the slope moves the jump, not the bank's top.
- The increments are the exact depths, and their values aren't rounded until they're written. Legacy rounded the
  depth step and each water surface to the millimetre, and discharge, area, perimeter, top width and velocity to 3
  decimals, and so could end a centimetre or so from the maximum flow's water surface.
- A carved channel is its exact shape, where legacy's was what its ordinates could draw (see arc.bathymetry).

Errors in the legacy code, not repeated here
--------------------------------------------
- A cell whose slope legacy's first slope search changed got no rating curve at all: it returned there before
  writing the increments, a return its own comment doubted. (Here no slope search is needed.)
- A water surface had to be above 0 m, which dropped cells below sea level; legacy only kept them by raising a DEM
  with any elevation below 0 by 100 m. Here any found water surface counts.
- An increment whose area rounded to 0 at 3 decimals, as the first of a rating curve stepping a couple of centimetres
  can, was written as all zeros, its water surface 0 m included, and the rating curve kept: 126 of legacy's VDT rows
  on the 51 real sites start at a water surface of 0 m. Here nothing is rounded, so water above the stream cell
  always has area.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from arc.hydraulics import (DepthRoughness, _check_slope_factor, _conveyance, _parameters, hydraulic_profile,
                            profile_compound_geometry, wse_for_profile_conveyance)
from arc.xsection.xsection import XSection

FIX_UP_STEP = 0.01  # metres: how far at a time legacy looked above an increment whose discharge fell
DISCHARGE_CAP = 1.01  # an increment's discharge may exceed the discharge at the top by 1%
ACCEPTABLE = (0.5, 1.5)  # the discharge at the top, as a share of the maximum flow, that gives a rating curve
BASEFLOW_MARGIN = 0.001  # the first increment's discharge is set this far below the baseflow
Q, V, T, WSE, P = range(5)


class RatingCurve(NamedTuple):
    """A cross section's flow increments (see the notes above). increments is (count, 5): discharge, velocity, top
    width, water surface elevation and wetted perimeter, in legacy's order q, v, t, wse, p."""
    increments: np.ndarray
    max_wse: float  # the lowest water surface carrying the maximum flow
    start: int  # the last increment, counting from 1, whose geometry wasn't valid, or 0
    last: int  # the last increment, counting from 1, with its own values, or 0

    @property
    def valid(self) -> bool:
        """Whether legacy would have written the increments to the VDT database."""
        return self.last > max(self.start, 0)


@njit(cache=True, error_model="numpy")
def section_hydraulics(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay, shallow_factor,
                       deep_factor):
    """Area, wetted perimeter, top width and conveyance of a cross section's profile at a water surface elevation,
    and the channel's area and conveyance. The conveyance is the sum of the subsections' where there are banks, and
    all channel where not."""
    left, channel, right = profile_compound_geometry(stations, elevations, mannings_n, center, left_bank, right_bank,
                                                     wse, k_decay, shallow_factor, deep_factor)
    area = left[0] + channel[0] + right[0]
    perimeter = left[1] + channel[1] + right[1]
    top_width = left[2] + channel[2] + right[2]
    channel_k = _conveyance(channel[0], channel[3])
    k = _conveyance(left[0], left[3]) + channel_k + _conveyance(right[0], right[3])
    return area, perimeter, top_width, k, channel[0], channel_k


@njit(cache=True, error_model="numpy")
def _increments(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
                thalweg, max_wse, count, discharge_scale, q_top, out):
    """legacy flood_increments, on the exact hydraulics: fills out (count, 5) and returns the last increment whose
    geometry wasn't valid and the last with its own values, counting from 1. q_top is the discharge at max_wse."""
    step = (max_wse - thalweg) / count
    start = 0
    last = 0
    prev_t = prev_a = prev_p = prev_q = prev_v = prev_wse = 0.0
    for i in range(1, count + 1):
        wse = thalweg + step * i
        area, perimeter, top_width, k, _, _ = section_hydraulics(stations, elevations, mannings_n, center, left_bank,
                                                                 right_bank, wse, k_decay, shallow_factor, deep_factor)
        row = out[i - 1]
        if top_width > 0.0 and area > 0.0 and perimeter > 0.0:
            q = k * discharge_scale
            v = q / area
            if q < prev_q:
                # Look a centimetre at a time, up to the next increment, for a discharge above the last one
                candidate = wse + FIX_UP_STEP
                upper = thalweg + step * (i + 1)
                while candidate < upper:
                    area, perimeter, top_width, k, _, _ = section_hydraulics(
                        stations, elevations, mannings_n, center, left_bank, right_bank, candidate, k_decay,
                        shallow_factor, deep_factor)
                    q_candidate = k * discharge_scale
                    if area > prev_a and perimeter > prev_p and q_candidate > prev_q and q_candidate <= q_top:
                        wse = candidate
                        q = q_candidate
                        v = q_candidate / area
                        break
                    candidate += FIX_UP_STEP
            if q <= prev_q or q > q_top * DISCHARGE_CAP:
                # Repeat the last increment
                row[Q], row[V], row[T], row[WSE], row[P] = prev_q, prev_v, prev_t, prev_wse, prev_p
                continue
            row[Q], row[V], row[T], row[WSE], row[P] = q, v, top_width, wse, perimeter
            prev_t, prev_a, prev_p, prev_q, prev_v, prev_wse = top_width, area, perimeter, q, v, wse
            last = i
        else:
            start = i
            row[Q] = row[V] = row[T] = row[WSE] = row[P] = 0.0
    return start, last


def _banks(xs: XSection) -> tuple[float, float]:
    return float(xs.left_bank_distance), float(xs.right_bank_distance)


def max_flow_wse(xs: XSection, q_max: float, slope: float, *, roughness: DepthRoughness | None = None,
                 slope_factor: float = 1.0) -> float:
    """The lowest water surface elevation at which the cross section carries q_max (arc.hydraulics'
    wse_for_discharge), or NaN if it never does."""
    if not slope > 0.0:
        return math.nan
    target = q_max / (math.sqrt(slope) * _check_slope_factor(slope_factor))
    return float(wse_for_profile_conveyance(*hydraulic_profile(xs), *_banks(xs), target, *_parameters(roughness)))


def rating_curve(xs: XSection, q_max: float, slope: float, increments: int, *, baseflow: float = 0.0,
                 roughness: DepthRoughness | None = None, slope_factor: float = 1.0) -> RatingCurve | None:
    """A cross section's rating curve up to its maximum flow (see the notes above), or None if it isn't
    acceptable: it never carries the maximum flow, or the discharge at the top is more than half of it away. A
    maximum flow that isn't above 0, or no increments, give a curve without any (start -1 and last 0, as legacy left
    them).

    increments is legacy's VDT_Database_NumIterations. With a baseflow above 0.001 m^3/s, the first valid increment's
    discharge is lowered to just below it if it was at or above it, as legacy did.
    """
    count = max(int(increments), 0)
    out = np.full((count, 5), np.nan)
    if not q_max > 0.0:
        return RatingCurve(out, math.nan, -1, 0)
    wse = max_flow_wse(xs, q_max, slope, roughness=roughness, slope_factor=slope_factor)
    if not math.isfinite(wse):
        return None
    profile = hydraulic_profile(xs)
    left, right = _banks(xs)
    parameters = _parameters(roughness)
    discharge_scale = math.sqrt(slope) * _check_slope_factor(slope_factor)
    q_top = section_hydraulics(*profile, left, right, wse, *parameters)[3] * discharge_scale
    if not ACCEPTABLE[0] * q_max <= q_top <= ACCEPTABLE[1] * q_max:
        return None
    if count == 0:
        return RatingCurve(out, wse, -1, 0)
    thalweg = float(profile.elevations[profile.center])
    start, last = _increments(*profile, left, right, *parameters, thalweg, wse, count, discharge_scale,
                              float(q_top), out)
    if last > start and baseflow > BASEFLOW_MARGIN and out[start, Q] >= baseflow:
        out[start, Q] = baseflow - BASEFLOW_MARGIN
    return RatingCurve(out, wse, int(start), int(last))
