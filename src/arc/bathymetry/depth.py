"""How deep a cross section's channel is below its reference level, from baseflow or a drainage-area power law.

The reference level is the water surface the DEM shows at the stream cell, or with bank elevations
(Bathy_Use_Banks) the bank elevation given. The channel carries the baseflow with its water at the reference level,
by Manning's equation with legacy ARC's fixed roughness of 0.03:

- A channel with valid banks is a trapezoid as wide as its banks at the reference level, whose sloping sides each
  take up trapezoid_height of that width, whatever the depth.
- A single-cell channel, or one without valid banks, is a triangle whose deepest point is the stream cell and whose
  sides run up to the ordinates either side. Without bank elevations those ordinates keep their ground elevations
  (so the water reaches only part way up a side that stands above the reference level); with bank elevations they
  are at the reference level.

Legacy ARC stepped the depth up until the discharge passed the baseflow, in steps down to 1 cm for a trapezoid
(stopping just short of the answer) and 10 cm for a triangle (stopping just past it). This solves for the depth
exactly.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from arc.bathymetry.banks import Banks, _on_raster_ordinates
from arc.hydraulics import _newton_or_bisect
from arc.xsection.xsection import XSection

MANNINGS_N = 0.03  # legacy ARC's roughness for the bathymetry depth, whatever the land cover
_TRAPEZOID, _TRIANGLE = 0, 1


class BathymetryDepth(NamedTuple):
    depth: float
    apply: bool  # whether to carve the channel at all
    source: str  # "target_depth" or "baseflow_manning"


@njit(cache=True, error_model="numpy")
def _log_conveyance(shape, y, a, b, c):
    """ln(A**(5/3) / P**(2/3)) of the channel at depth y, and its derivative with respect to y.

    For the trapezoid a and b are the bottom and top widths. For the triangle a is the ordinate spacing, and b and c
    are how far the ordinates either side stand above the reference level.
    """
    if shape == _TRAPEZOID:
        side = 0.5 * (b - a)
        area = 0.5 * y * (a + b)
        slant = math.sqrt(side * side + y * y)
        perimeter = a + 2.0 * slant
        return (5.0 / 3.0) * math.log(area) - (2.0 / 3.0) * math.log(perimeter), \
            (5.0 / 3.0) / y - (4.0 / 3.0) * y / (slant * perimeter)
    # Each side of the triangle is wet from the stream cell out to where the water surface meets it
    left_width = a * y / (y + b)
    right_width = a * y / (y + c)
    d_left_width = a * b / ((y + b) * (y + b))
    d_right_width = a * c / ((y + c) * (y + c))
    area = 0.5 * y * (left_width + right_width)
    d_area = 0.5 * (left_width + right_width) + 0.5 * y * (d_left_width + d_right_width)
    left_length = math.sqrt(left_width * left_width + y * y)
    right_length = math.sqrt(right_width * right_width + y * y)
    perimeter = left_length + right_length
    d_perimeter = (left_width * d_left_width + y) / left_length + (right_width * d_right_width + y) / right_length
    return (5.0 / 3.0) * math.log(area) - (2.0 / 3.0) * math.log(perimeter), \
        (5.0 / 3.0) * d_area / area - (2.0 / 3.0) * d_perimeter / perimeter


@njit(cache=True, error_model="numpy")
def _depth(shape, q, slope, mannings_n, a, b, c):
    """The depth at which the channel carries q, by Manning's equation: 0 for no flow, NaN for no slope."""
    if not q > 0.0:
        return 0.0
    if not slope > 0.0 or not mannings_n > 0.0:
        return math.nan
    target = math.log(q * mannings_n / math.sqrt(slope))

    # Start from shallow water across the channel's whole width W at the reference level, where A = W_mean * y and
    # P is about W, so that ln K = 5/3 * ln(W_mean * y) - 2/3 * ln W
    if shape == _TRAPEZOID:
        u = 0.6 * target + 0.4 * math.log(b) - math.log(0.5 * (a + b))
    else:
        u = 0.6 * target + 0.4 * math.log(2.0) - 0.6 * math.log(a)
        if b > 0.0 and c > 0.0:
            # A triangle whose sides stand above the water is narrower than that while it's shallow, with
            # A = y**2 * a/2 * (1/b + 1/c) and P = y * (sqrt(1 + (a/b)**2) + sqrt(1 + (a/c)**2)), so K goes as
            # y**(8/3). Being narrower, it's deeper than either estimate on its own.
            area_per_y2 = 0.5 * a * (1.0 / b + 1.0 / c)
            perimeter_per_y = math.sqrt(1.0 + (a / b) ** 2) + math.sqrt(1.0 + (a / c) ** 2)
            u = max(u, 0.375 * target + 0.125 * math.log(perimeter_per_y ** 2 / area_per_y2 ** 5))

    # ln K rises with ln y at a rate between 1 and 8/3, so Newton's method on ln y converges in a few steps. It
    # bisects if a step would leave the bracket of the steps so far, which rounding might cause.
    u_lo, u_hi = -np.inf, np.inf
    for _ in range(100):
        y = math.exp(u)
        value, slope_in_y = _log_conveyance(shape, y, a, b, c)
        error = value - target
        if error == 0.0:
            break
        if error > 0.0:
            u_hi = u
        else:
            u_lo = u
        u, converged = _newton_or_bisect(u, error / (y * slope_in_y), u_lo, u_hi)
        if converged or not math.isfinite(u):
            break
    return math.exp(u)


@njit(cache=True, error_model="numpy")
def trapezoid_depth(q, bottom_width, top_width, slope, mannings_n):
    """The depth at which a trapezoid with these bottom and top widths carries q (legacy find_depth_of_bathymetry)."""
    if not top_width > 0.0 or not 0.0 <= bottom_width <= top_width:
        return math.nan
    return _depth(_TRAPEZOID, q, slope, mannings_n, bottom_width, top_width, 0.0)


@njit(cache=True, error_model="numpy")
def triangle_depth(q, spacing, left_height, right_height, slope, mannings_n):
    """The depth of the stream cell below the water surface at which a triangle reaching the ordinates either side
    carries q, when those ordinates stand these heights above the water (legacy
    find_depth_of_bathymetry_triangle). A height below the water counts as level with it."""
    if not spacing > 0.0:
        return math.nan
    return _depth(_TRIANGLE, q, slope, mannings_n, spacing, max(left_height, 0.0), max(right_height, 0.0))


def channel_depth(xs: XSection, banks: Banks, q: float, slope: float, *, trapezoid_height: float,
                  bank_elevation: float | None = None, mannings_n: float = MANNINGS_N) -> float:
    """How deep the channel is below its reference level to carry the baseflow q (see the notes above).

    Without bank_elevation the reference level is the stream cell's elevation. Zero for no baseflow, and zero for
    a single-cell channel without an ordinate on the raster each side. NaN for a slope that isn't positive.
    """
    trapezoid_height = _check_trapezoid_height(trapezoid_height)
    if banks.valid and not banks.single_cell:
        top_width = banks.top_width
        return float(trapezoid_depth(q, top_width * (1.0 - 2.0 * trapezoid_height), top_width, slope, mannings_n))

    center = xs.elevations.size // 2
    if _on_raster_ordinates(xs.elevations, center, -1) < 1 or _on_raster_ordinates(xs.elevations, center, 1) < 1:
        return 0.0
    if bank_elevation is None:
        reference = float(xs.elevations[center])
        left_height = float(xs.elevations[center - 1]) - reference
        right_height = float(xs.elevations[center + 1]) - reference
    else:
        left_height = right_height = 0.0
    return float(triangle_depth(q, float(xs.ordinate_distance), left_height, right_height, slope, mannings_n))


def bathymetry_depth(xs: XSection, banks: Banks, q: float, slope: float, *, trapezoid_height: float,
                     bank_elevation: float | None = None, target_depth: float | None = None,
                     mannings_n: float = MANNINGS_N) -> BathymetryDepth:
    """The depth to carve the channel to: a drainage-area target depth if there's a valid one, which always applies,
    or otherwise channel_depth for the baseflow, which applies only with baseflow."""
    if target_depth is not None and math.isfinite(target_depth) and target_depth > 0.0:
        return BathymetryDepth(float(target_depth), True, "target_depth")
    depth = channel_depth(xs, banks, q, slope, trapezoid_height=trapezoid_height, bank_elevation=bank_elevation,
                          mannings_n=mannings_n)
    return BathymetryDepth(depth, bool(q > 0.0), "baseflow_manning")


def power_law_geometry(drainage_area: float, coefficient_depth: float | None, exponent_depth: float | None,
                       coefficient_width: float | None, exponent_width: float | None
                       ) -> tuple[float | None, float | None]:
    """The bankfull depth and width that the drainage-area power laws give, each None without its coefficients."""
    depth = None
    if coefficient_depth is not None and exponent_depth is not None:
        depth = float(coefficient_depth * drainage_area ** exponent_depth)
    width = None
    if coefficient_width is not None and exponent_width is not None:
        width = float(coefficient_width * drainage_area ** exponent_width)
    return depth, width


def _check_trapezoid_height(trapezoid_height: float) -> float:
    """The share of the top width each sloping side takes: at most a half, when the trapezoid becomes a triangle."""
    if not 0.0 <= trapezoid_height <= 0.5:
        raise ValueError(f"trapezoid_height must be between 0 and 0.5, not {trapezoid_height!r}.")
    return float(trapezoid_height)
