"""Linking a sampled cross section's ordinates to raster cells, to read land cover and to write bathymetry.

Each ordinate belongs to the cell nearest to it. Sampling steps one cell along whichever axis the cross section
crosses fastest, so consecutive ordinates are in different cells.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from arc.xsection.sampling import _compute_dem_coordinates


def ordinate_cells(row: int, col: int, stream_direction: float, cross_section_length: float, dx: float, dy: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The row and column of the cell nearest each ordinate of the cross section that sample_cross_section takes
    with these arguments. Ordinates off the raster get cells off it too."""
    columns, rows, _ = _compute_dem_coordinates(row, col, stream_direction, cross_section_length, dx, dy)
    return np.rint(rows).astype(np.intp), np.rint(columns).astype(np.intp)


def sample_land_cover(land_cover: np.ndarray, row: int, col: int, stream_direction: float,
                      cross_section_length: float, dx: float, dy: float) -> np.ndarray:
    """The land cover class at each ordinate, from its nearest cell, as floats that are NaN off the raster."""
    rows, cols = ordinate_cells(row, col, stream_direction, cross_section_length, dx, dy)
    on_raster = (rows >= 0) & (rows < land_cover.shape[0]) & (cols >= 0) & (cols < land_cover.shape[1])
    values = np.full(rows.size, np.nan)
    values[on_raster] = land_cover[rows[on_raster], cols[on_raster]]
    return values


@njit(cache=True, error_model="numpy")
def burn_into_raster(raster, rows, cols, elevations, changed):
    """Write the changed ordinates' elevations into their cells of an output raster. A cell that already has a value
    from another cross section takes the average of that and the new one, as legacy ARC did."""
    for k in range(elevations.size):
        if not changed[k]:
            continue
        r, c = rows[k], cols[k]
        if 0 <= r < raster.shape[0] and 0 <= c < raster.shape[1]:
            current = raster[r, c]
            raster[r, c] = elevations[k] if np.isnan(current) else current + (elevations[k] - current) * 0.5
