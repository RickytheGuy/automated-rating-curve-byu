"""Bank shelves: an experiment the user asked for on 2026-10-01, not in the source.

Where a channel is carved narrower than the channel the DEM shows (as a drainage-area width prior can make it, C6),
the ground beyond each bank top runs straight down to the first ordinate beyond it, which can leave a hollow between
the carved channel and the DEM's own bank: the water tops the carved bank and spills into it. Here the ground beyond
the bank top runs horizontally at the bank elevation instead, until it meets the ground coming back up to that
elevation, which fills the hollow to the bank elevation in the profile and in the bathymetry raster. That happens on a
side only where:

- the width-to-depth ratio's banks, found on the ground before the carve, hold more ordinates than the carved channel
  (the banks included), and
- the first ordinate beyond the carved bank is below the bank elevation, and the ground comes back up to it, on the
  raster, at or within the width-to-depth bank on that side.

The user's rule is for channels whose width a drainage-area power law set (POWER_LAW, the single cells a width
prior makes); methods=None gives the shelves to every carved channel, a reach's median width included. Only with
bank elevations (Bathy_Use_Banks), whose bank elevation the shelf is at. carving_with_shelves wraps
arc.pipeline.carve_channel; vc_variants' "bank_shelf" setting applies it (True for POWER_LAW, "all" for every
channel), and record collects what it found.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from arc.bathymetry.banks import OFF_RASTER_ELEVATION, _width_to_depth_banks, single_cell_banks
from arc.xsection.xsection import Profile

TOLERANCE = 1e-9  # of a spacing, for being at a bank
POWER_LAW = ("single_cell",)  # the banks a width prior gives a channel the DEM can't resolve


class Shelf(NamedTuple):
    distance: float  # from the stream cell to where the ground comes back up to the bank elevation
    offsets: tuple  # the ordinates the shelf covers, in spacings from the stream cell
    area: float  # the hollow's area below the bank elevation, in m^2 (the shelf fills it)
    depth: float  # how far the hollow's lowest ordinate is below the bank elevation


class Found(NamedTuple):
    """What the shelf rule found at one carved cross section."""
    method: str  # the carved banks' method
    channel_cells: int
    wd_left: float  # the width-to-depth ratio's banks on the ground before the carve (NaN if none)
    wd_right: float
    wd_cells: int  # 0 where the width-to-depth ratio doesn't resolve a channel
    wider: bool  # the width-to-depth banks hold more ordinates than the carved channel
    dip_left: bool  # the first ordinate beyond the carved bank is below the bank elevation
    dip_right: bool
    left: Shelf | None
    right: Shelf | None


def cells_within(left: float, right: float, spacing: float) -> int:
    """How many ordinates lie between banks this far out, the banks included."""
    tolerance = TOLERANCE * spacing
    return int(math.floor((left + tolerance) / spacing)) + int(math.floor((right + tolerance) / spacing)) + 1


def _first_beyond(bank: float, spacing: float) -> int:
    return int(math.floor((bank + TOLERANCE * spacing) / spacing)) + 1


def side_shelf(ground: np.ndarray, spacing: float, step: int, bank: float, elevation: float,
               limit: float) -> Shelf | None:
    """One side's shelf (see the notes above), or None."""
    center = ground.size // 2
    last = ground.size - 1 - center if step > 0 else center
    first = _first_beyond(bank, spacing)
    if first > last or not ground[center + step * first] < elevation:
        return None
    j = first
    while j <= last and ground[center + step * j] < elevation:
        j += 1
    if j > last or ground[center + step * j] >= OFF_RASTER_ELEVATION:
        return None  # never comes back up on the raster
    below, above = float(ground[center + step * (j - 1)]), float(ground[center + step * j])
    distance = spacing * (j - 1 + (elevation - below) / (above - below))
    if not distance <= limit + TOLERANCE * spacing:
        return None
    # The hollow's area below the bank elevation, from the bank top to where the ground comes back up
    stations = np.concatenate([[bank], spacing * np.arange(first, j), [distance]])
    heights = np.concatenate([[0.0], elevation - ground[center + step * np.arange(first, j)], [0.0]])
    return Shelf(float(distance), tuple(range(first, j)), float(np.trapezoid(heights, stations)),
                 float(heights.max()))


def find_shelves(ground: np.ndarray, spacing: float, banks, elevation: float) -> Found:
    """Both sides' shelves for a channel carved between banks below a bank elevation, on ground (the ordinates
    before the carve)."""
    wd_left, wd_right, _ = _width_to_depth_banks(ground, spacing)
    resolved = wd_left >= 0.0 and wd_right >= 0.0 and wd_left + wd_right >= 2.0 * spacing
    wd_cells = cells_within(wd_left, wd_right, spacing) if resolved else 0
    channel_cells = cells_within(banks.left, banks.right, spacing)
    center = ground.size // 2
    dips = []
    for step, bank in ((-1, banks.left), (1, banks.right)):
        first = _first_beyond(bank, spacing)
        last = ground.size - 1 - center if step > 0 else center
        dips.append(first <= last and bool(ground[center + step * first] < elevation))
    wider = resolved and wd_cells > channel_cells
    left = side_shelf(ground, spacing, -1, banks.left, elevation, wd_left) if wider else None
    right = side_shelf(ground, spacing, 1, banks.right, elevation, wd_right) if wider else None
    return Found(banks.method, channel_cells, float(wd_left), float(wd_right), wd_cells, bool(wider), dips[0],
                 dips[1], left, right)


def apply_shelves(xs, changed: np.ndarray, elevation: float, found: Found) -> None:
    """Give a carved cross section its shelves: in its profile, the ground beyond each bank top runs horizontally at
    the bank elevation to where the ground comes back up, and the ordinates under the shelf take the bank elevation
    (marked in changed, for the raster)."""
    stations, elevations, mannings_n, _ = xs.profile
    spacing = float(xs.ordinate_distance)
    center = xs.elevations.size // 2
    keep = np.ones(stations.size, dtype=bool)
    new_stations, new_elevations, new_n = [], [], []
    for shelf, sign in ((found.left, -1), (found.right, 1)):
        if shelf is None:
            continue
        for k in shelf.offsets:
            keep &= ~np.isclose(stations, sign * k * spacing, rtol=0.0, atol=TOLERANCE * spacing)
            xs.elevations[center + sign * k] = elevation
            changed[center + sign * k] = True
        new_stations.append(sign * shelf.distance)
        new_elevations.append(elevation)
        new_n.append(xs.mannings_n[center + sign * shelf.offsets[-1]])  # the ground's under the shelf's far end
    all_stations = np.concatenate([stations[keep], new_stations])
    order = np.argsort(all_stations, kind="stable")
    all_stations = all_stations[order]
    profile = np.concatenate([elevations[keep], new_elevations])[order]
    n = np.concatenate([mannings_n[keep], new_n])[order]
    xs.profile = Profile(all_stations, profile, n, int(np.flatnonzero(all_stations == 0.0)[0]))


def carving_with_shelves(carve, apply: bool = True, record: list | None = None, methods=POWER_LAW):
    """carve_channel, with the shelves added (apply) on channels whose banks came by one of methods (None for all),
    and what was found appended to record as (xs, Found, the ordinates before the carve)."""
    def carve_channel(xs, banks, depth, *, trapezoid_height, bank_elevation=None):
        ground = xs.elevations.copy()
        changed = carve(xs, banks, depth, trapezoid_height=trapezoid_height, bank_elevation=bank_elevation)
        if bank_elevation is None or xs.profile is None or not changed.any():
            return changed
        used = banks if banks.valid else single_cell_banks(xs)
        found = find_shelves(ground, float(xs.ordinate_distance), used, float(bank_elevation))
        if record is not None:
            record.append((xs, found, ground))
        if apply and (found.left is not None or found.right is not None) and (methods is None or
                                                                           used.method in methods):
            apply_shelves(xs, changed, float(bank_elevation), found)
        return changed
    return carve_channel
