"""Carving the bathymetry into a cross section.

A channel with valid banks is carved as a trapezoid between its banks. Its top is at the reference level at the
banks, and its sides slope down to the bed over trapezoid_height of the top width, measured from the nearer bank.
Its bed is flat, depth below the reference level. A single-cell channel, or one without valid banks, lowers just the
stream cell to the bed.

Without bank elevations the reference level is the stream cell's elevation, the channel only ever lowers the ground,
and a depth of 25 m or more isn't carved at all. With bank elevations (Bathy_Use_Banks) the reference level is the
bank elevation given, and the channel sets the ground whether that raises or lowers it.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from arc.bathymetry.banks import Banks, _on_raster_ordinates
from arc.bathymetry.depth import _check_trapezoid_height
from arc.xsection.xsection import XSection

MAX_DEPTH = 25.0  # without bank elevations, a deeper channel isn't carved


@njit(cache=True, error_model="numpy")
def _carve(elevations, spacing, left, right, trapezoid, reference, depth, trapezoid_height, lower_only, changed):
    """Carve the channel into elevations in place, marking the ordinates it sets in changed."""
    center = elevations.size // 2
    bed = reference - depth
    if not trapezoid:
        if not lower_only or bed < elevations[center]:
            elevations[center] = bed
            changed[center] = True
        return

    slope_width = trapezoid_height * (left + right)
    tolerance = 1e-9 * spacing
    first = max(center - int(math.floor((left + tolerance) / spacing)), 0)
    last = min(center + int(math.floor((right + tolerance) / spacing)), elevations.size - 1)
    for k in range(first, last + 1):
        offset = (k - center) * spacing
        to_bank = max(min(offset + left, right - offset), 0.0)  # the distance to the nearer bank
        if slope_width > 0.0:
            elevation = bed + depth * max(1.0 - to_bank / slope_width, 0.0)
        else:
            elevation = reference if to_bank <= tolerance else bed  # a rectangle: the banks are its top
        if not lower_only or elevation < elevations[k]:
            elevations[k] = elevation
            changed[k] = True


def carve_channel(xs: XSection, banks: Banks, depth: float, *, trapezoid_height: float,
                  bank_elevation: float | None = None) -> np.ndarray:
    """Carve a channel depth below its reference level into the cross section's elevations (see the notes above).

    Returns which ordinates it set, for burn_into_raster. Sets none for a depth or reference level that isn't
    finite, or a single-cell channel without an ordinate on the raster each side.
    """
    trapezoid_height = _check_trapezoid_height(trapezoid_height)
    elevations = xs.elevations
    center = elevations.size // 2
    changed = np.zeros(elevations.size, dtype=bool)
    reference = float(elevations[center]) if bank_elevation is None else float(bank_elevation)
    if not (math.isfinite(depth) and math.isfinite(reference)):
        return changed
    if bank_elevation is None and depth >= MAX_DEPTH:
        return changed

    trapezoid = bool(banks.valid and not banks.single_cell)
    if not trapezoid and (_on_raster_ordinates(elevations, center, -1) < 1
                          or _on_raster_ordinates(elevations, center, 1) < 1):
        return changed
    _carve(elevations, float(xs.ordinate_distance), float(banks.left), float(banks.right), trapezoid, reference,
           float(depth), trapezoid_height, bank_elevation is None, changed)
    return changed
