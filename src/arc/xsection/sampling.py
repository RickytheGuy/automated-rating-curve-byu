
import numpy as np
from scipy.ndimage import map_coordinates

from .xsection import XSection

def sample_cross_section(
        dem: np.ndarray,
        manning_n: np.ndarray,
        row: int,
        col: int,
        stream_direction: float,
        cross_section_length: float,
        dx: float,
        dy: float) -> XSection:
    """
    Sample a cross section from a DEM and Manning's n raster. We assume that left and right are defined
    looking upstream

    The cross section is centered on (row, col) and runs perpendicular to the stream. stream_direction is
    the direction the water flows, in radians, measured in map units from the +column axis toward the +row
    axis (rows increase downward, so this is clockwise on a north-up raster). The sampled values run from
    the left end of the cross section to the right end, with the stream cell in the middle.
    """
    dem_x_coords, dem_y_coords, ordinate_distance = _compute_dem_coordinates(
        row, col, stream_direction, cross_section_length, dx, dy)

    # Sample the DEM and Manning's n raster at the calculated coordinates
    elevations = map_coordinates(dem, [dem_y_coords, dem_x_coords], order=1, mode='constant', cval=9999)
    mannings_n_values = map_coordinates(manning_n, [dem_y_coords, dem_x_coords], order=1, mode='constant', cval=9999)

    return XSection(elevations, mannings_n_values, ordinate_distance)

def _compute_dem_coordinates(
        row: int,
        col: int,
        stream_direction: float,
        cross_section_length: float,
        dx: float,
        dy: float):
    """
    Compute the coordinates of the DEM grid points for a given cross section.

    Returns the column and row coordinates of the ordinates, ordered from the left end of the cross
    section to the right end looking upstream, and the distance between ordinates.
    """
    # The cross section runs perpendicular to the stream, from the left bank to the right bank
    xs_direction = stream_direction - np.pi / 2

    # Step one cell along whichever axis the cross section crosses fastest, so every ordinate lands on a
    # row or column of cell centres. The same number of ordinates go out to each side, so the stream cell
    # is always sampled
    ordinate_distance = 1 / max(abs(np.cos(xs_direction)) / dx, abs(np.sin(xs_direction)) / dy)
    num_points_per_side = int(cross_section_length / 2 / ordinate_distance)
    x_coords = ordinate_distance * np.arange(-num_points_per_side, num_points_per_side + 1)
    y_coords = np.zeros_like(x_coords)

    # Rotate the coordinates onto the cross section
    rotation_matrix = np.array([[np.cos(xs_direction), -np.sin(xs_direction)],
                                [np.sin(xs_direction), np.cos(xs_direction)]])
    rotated_coords = rotation_matrix @ np.vstack((x_coords, y_coords))

    # Translate the coordinates to the DEM grid
    dem_x_coords = col + rotated_coords[0, :] / dx
    dem_y_coords = row + rotated_coords[1, :] / dy

    return dem_x_coords, dem_y_coords, ordinate_distance
