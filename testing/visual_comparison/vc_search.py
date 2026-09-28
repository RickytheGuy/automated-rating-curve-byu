"""The angle search's test depth: the search generalised to any depth, or several, and the experiment's data.

The new pipeline turns each cross section to the direction, of those Degree_Manip and Degree_Interval give, whose
water 0.5 m above the stream cell is narrowest (arc.xsection.orientation, as legacy did). candidate_metrics measures
every candidate's water width and flow area at any depths instead, with the pipeline's rule for sides that reach the
raster's edge, so a test depth or a combination of them can be chosen afterwards. At a single depth, the narrowest
width is the pipeline's own choice, bit for bit.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from arc.xsection.orientation import _SAME_DIRECTION, _reach
from arc.xsection.sampling import sample_elevations

DEPTHS = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
BANKFULL, FLOOD = DEPTHS.size, DEPTHS.size + 1  # the table's two per-cell depths, after DEPTHS


@njit(cache=True, error_model="numpy")
def side_metrics(elevations, center, step, spacing, wse, limit):
    """The distance from the stream cell out to the water's edge on one side, no further than limit, and the area of
    water between the stream cell and there."""
    end = elevations.size - 1 if step > 0 else 0
    j = center
    area = 0.0
    while j != end:
        z_in, z_out = elevations[j], elevations[j + step]
        inner = spacing * abs(j - center)
        if inner >= limit:
            return limit, area
        if not wse > z_out:  # the edge is in this segment
            fraction = (wse - z_in) / (z_out - z_in)
            edge = inner + spacing * fraction
            if edge > limit:
                z_limit = z_in + (limit - inner) / spacing * (z_out - z_in)
                return limit, area + 0.5 * ((wse - z_in) + (wse - z_limit)) * (limit - inner)
            return edge, area + 0.5 * (wse - z_in) * spacing * fraction
        if inner + spacing > limit:
            z_limit = z_in + (limit - inner) / spacing * (z_out - z_in)
            return limit, area + 0.5 * ((wse - z_in) + (wse - z_limit)) * (limit - inner)
        area += 0.5 * ((wse - z_in) + (wse - z_out)) * spacing
        j += step
    return min(spacing * abs(end - center), limit), area


@njit(cache=True, error_model="numpy")
def candidate_metrics(dem, row, col, direction, length, dx, dy, offsets, depths):
    """Each distinct candidate's offset from the stream direction, and its water width and flow area at each depth
    above the stream cell, each side no further out than every candidate reaches on that side (the pipeline's
    rule)."""
    n = offsets.size
    tried = np.empty(n)
    reaches = np.empty((n, 2))
    count = 0
    for offset in offsets:
        offset = offset % np.pi
        if offset > np.pi / 2:
            offset -= np.pi
        repeat = False
        for j in range(count):
            if abs(tried[j] - offset) < _SAME_DIRECTION:
                repeat = True
        if repeat:
            continue
        elevations, spacing = sample_elevations(dem, row, col, direction + offset, length, dx, dy)
        center = elevations.size // 2
        reaches[count, 0] = _reach(elevations, center, -1, spacing)
        reaches[count, 1] = _reach(elevations, center, 1, spacing)
        tried[count] = offset
        count += 1
    left_limit, right_limit = reaches[:count, 0].min(), reaches[:count, 1].min()
    widths = np.empty((count, depths.size))
    areas = np.empty((count, depths.size))
    for j in range(count):
        elevations, spacing = sample_elevations(dem, row, col, direction + tried[j], length, dx, dy)
        center = elevations.size // 2
        for d in range(depths.size):
            wse = elevations[center] + depths[d]
            wl, al = side_metrics(elevations, center, -1, spacing, wse, left_limit)
            wr, ar = side_metrics(elevations, center, 1, spacing, wse, right_limit)
            widths[j, d] = wl + wr
            areas[j, d] = al + ar
    return tried[:count], widths, areas


@njit(cache=True, error_model="numpy")
def best_candidate(metric):
    """The candidate with the smallest metric, or with several depths the smallest sum of each depth's metric over
    its smallest among the candidates; the first of equals."""
    count, depths = metric.shape
    if depths == 1:
        best, smallest = 0, np.inf
        for j in range(count):
            if metric[j, 0] < smallest:
                best, smallest = j, metric[j, 0]
        return best
    lows = np.empty(depths)
    for d in range(depths):
        lows[d] = max(metric[:, d].min(), 1e-12)
    best, smallest = 0, np.inf
    for j in range(count):
        score = 0.0
        for d in range(depths):
            score += metric[j, d] / lows[d]
        if score < smallest:
            best, smallest = j, score
    return best


def narrowest_by(dem, row, col, direction, length, dx, dy, offsets, depths, use_area=False) -> float:
    """The pipeline's narrowest_direction, at these depths (by width, or by flow area)."""
    offsets = np.asarray(offsets, dtype=np.float64).ravel()
    if offsets.size == 0 or (offsets.size == 1 and offsets[0] == 0.0):
        return float(direction)
    tried, widths, areas = candidate_metrics(dem, int(row), int(col), float(direction), float(length), float(dx),
                                             float(dy), offsets, np.asarray(depths, dtype=np.float64))
    return float(direction + tried[best_candidate(areas if use_area else widths)])


# --- Made-up valleys ---------------------------------------------------------------------------------------------


def made_up_valley(angle, floodplain_half, amplitude, noise, seed, dx, dy, size=240, wavelength=1800.0,
                   channel_half=18.0, bank_height=1.0, wall_slope=0.12, wall_height=30.0, fall=0.001):
    """A DEM and its stream raster: a straight valley at angle (radians from the +column axis towards the +row
    axis) with a flat floodplain floodplain_half metres either side of its axis, walls rising at wall_slope to
    wall_height, and a channel 2 * channel_half wide around a centreline meandering amplitude metres either side of
    the axis. The DEM shows the channel's water bank_height below the floodplain, and noise (its sd in metres,
    smoothed over about a cell). Also each stream cell's channel direction."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size]
    x = (cols - size / 2) * dx
    y = (rows - size / 2) * dy
    along = x * math.cos(angle) + y * math.sin(angle)
    across = -x * math.sin(angle) + y * math.cos(angle)
    center = amplitude * np.sin(2 * math.pi * along / wavelength)
    distance = np.abs(across)
    ground = np.where(distance <= floodplain_half, 0.0,
                      np.minimum((distance - floodplain_half) * wall_slope, wall_height))
    channel = np.abs(across - center) <= channel_half
    ground = ground - fall * along
    ground = np.where(channel, -fall * along - bank_height, ground)
    ground = ground + gaussian_filter(rng.normal(0.0, 1.0, ground.shape), 1.0) * noise / 0.282
    t = np.arange(-size * max(dx, dy), size * max(dx, dy), 0.2 * min(dx, dy))
    wiggle = amplitude * np.sin(2 * math.pi * t / wavelength)
    sc = np.rint((t * math.cos(angle) - wiggle * math.sin(angle)) / dx + size / 2).astype(int)
    sr = np.rint((t * math.sin(angle) + wiggle * math.cos(angle)) / dy + size / 2).astype(int)
    keep = (sr >= 0) & (sr < size) & (sc >= 0) & (sc < size)
    streams = np.zeros((size, size), dtype=np.int64)
    streams[sr[keep], sc[keep]] = 1
    heading = angle + np.arctan(amplitude * 2 * math.pi / wavelength * np.cos(2 * math.pi * t / wavelength))
    channel_direction = {}
    for r, c, a in zip(sr[keep], sc[keep], heading[keep]):
        channel_direction.setdefault((int(r), int(c)), float(a))
    return ground, streams, channel_direction
