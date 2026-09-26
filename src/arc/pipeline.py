"""ARC's main workflow, from its input rasters to its rating curves, cross sections and bathymetry.

It does what legacy Automated_Rating_Curve_Generator.main did, with the overhaul's modules, in one process and with
nothing kept between runs:

    from arc.config import Configs
    from arc.pipeline import run
    results = run(Configs.from_file("ARC_Input_File.yaml"))

or from the command line, ``python -m arc.pipeline ARC_Input_File.yaml``.

The steps
---------
1. Inputs: the flow file, the DEM, stream and land cover rasters, which must share one grid, and the stream network
   where the bathymetry, the slopes or the drainage-area power laws need it.
2. Stream cells: the stream raster's cells whose ID is in the flow file (every stream cell for representative cross
   sections), in the raster's row order, or the manual cross sections' cells.
3. Slopes, by Stream_Slope_Method (arc.xsection.slope). The cells of reaches whose slope can't be resolved are
   skipped, as legacy skipped them.
4. Roughness: the stream cells become water in the land cover, and each land cover class takes its n from the
   Manning's n table.
5. Cross sections (arc.xsection): the stream's direction at each stream cell, the low spot, the direction where the
   channel is narrowest, and the cross section there.
6. Banks (arc.bathymetry.find_banks), and the water's n between them.
7. Bathymetry, if Bathy_Out_File is set (arc.bathymetry): each reach's banks and bank elevations smoothed along the
   network, each channel's depth, the depths and beds smoothed along the network, and the channel carved into each
   cross section and burned into the bathymetry raster, whose gaps are then filled.
8. Rating curves (arc.rating_curve), or the representative cross sections (arc.outputs.representative).
9. Outputs (arc.outputs).

The modules' notes list how each step differs from legacy ARC. These are the workflow's own differences.

Where legacy used its own markers
---------------------------------
- DEM cells without data are walls, like the ground beyond the raster's edge (arc.xsection.sampling). Legacy read
  them as elevations of -9999 m, deep pits the water fell into.
- Legacy raised a DEM with any elevation below 0 m by 100 m, and took the 100 m off again in its outputs, so that its
  checks for elevations above 0 m worked. Nothing here needs that, so elevations are as they are.

Channels nothing resolves
-------------------------
- A channel whose banks neither the DEM, the land cover nor its reach's widths resolved is carved as wide as its
  width prior, where it has one, and otherwise one cell wide. Legacy carved one cell either way.

Errors in the legacy code, not repeated here
--------------------------------------------
- Legacy flipped south-up rasters to north-up but kept their geotransform, so their rows and columns in the VDT
  database, and the rasters it wrote, were upside down. Here rasters stay as they are, whichever way up.
- On rasters of different shapes, or in units other than metres or degrees, legacy logged an error and returned as
  if it had finished, writing nothing. Here it raises, and it also raises for rasters on different grids or rotated.
- A reach whose end-point slope couldn't be measured, a stream ID missing from the stream network's table, or a
  Reach_Average_Curve_File without a Print_VDT_Database made legacy fail with a KeyError or an AttributeError. Here
  the first two are skipped with a warning, and the third works.
- For a cell without a usable cross section, the reach-average curve file's metadata came from the cell before it.
  Here it's the cell's own.
- A reach without a smoothed bank elevation was dropped from the bathymetry and the rating curves even without
  Bathy_Use_Banks, where the bathymetry doesn't use it (arc.bathymetry.bed_smoothing). Here only with it.

Not here
--------
- Legacy's INFLECT curves and the reach bank and terrace depths it took from them: nothing it wrote used them.
- Its temporary INFLECT plots, and its processes and shared memory.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
import tqdm

from arc import LOG
from arc.bathymetry import (Banks, ReachSections, banks_for_width, bathymetry_depth, burn_into_raster, carve_channel,
                            fill_bathymetry_gaps, find_banks, in_bank, ordinate_cells, reach_network,
                            sample_land_cover, set_bank_distances, set_in_bank_roughness, single_cell_banks,
                            smooth_bank_elevations, smooth_channel_depths)
from arc.bathymetry.banks import _on_raster_ordinates
from arc.bathymetry.bed_smoothing import MAX_BED_GRADE
from arc.bathymetry.depth import power_law_geometry
from arc.config import Configs
from arc.hydraulics import DepthRoughness
from arc.io import Raster, Vector
from arc.outputs import (CrossSectionRecord, RatingCurves, RepresentativeSample,
                         representative_cross_section_dataframe, vdt_dataframe, write_ap, write_cross_sections,
                         write_curve_file, write_reach_average_curve_file, write_representative_cross_sections,
                         write_vdt)
from arc.outputs.rating_curves import BASE_ELEV, COL, COMID, ELEV, ROW, SLOPE, XS_ANGLE
from arc.rating_curve import rating_curve
from arc.xsection.low_spot import low_spot_cell
from arc.xsection.orientation import angle_offsets, narrowest_direction, stream_direction
from arc.xsection.sampling import OFF_RASTER_ELEVATION, sample_elevations
from arc.xsection.slope import (MIN_SLOPE, corrected_local_slopes, end_point_slope, fill_unresolved_reach_slopes,
                                local_average_slopes, reach_median_slope)
from arc.xsection.stream_path import along_stream_stations, stream_cells_by_reach
from arc.xsection.xsection import XSection

FLOODED_CELL = 3  # the flood raster's value at each stream cell that reached the rating curves, as legacy wrote it
_REQUIRED_MANUAL_COLUMNS = ("Row", "Col", "Ordinate_Dist", "XS1_Profile", "XS2_Profile", "LC1_Profile",
                            "LC2_Profile", "XS1_Row", "XS1_Col", "XS2_Row", "XS2_Col")
_ALONG_A_ROW = 1e-12  # a cross section whose direction's sine is smaller than this runs along a row
_METADATA_BUT_BASEFLOW = [COMID, ROW, COL, ELEV, SLOPE, XS_ANGLE, BASE_ELEV]


# --- What a run works with ------------------------------------------------------------------------------------------


class Grid(NamedTuple):
    """The input rasters, on one grid."""
    reference: Raster  # the DEM, whose grid the outputs are written on
    dem: np.ndarray  # float64, with cells without data as walls
    streams: np.ndarray  # int64, each stream cell holding its reach's ID and every other cell 0
    land_cover: np.ndarray  # uint8
    dx: float  # the cells' width and height in metres
    dy: float


@dataclass(frozen=True)
class Cells:
    """The stream cells a run works on, in order."""
    rows: np.ndarray
    cols: np.ndarray
    comids: np.ndarray  # the flow file's ID for each cell: its reach's, or a manual cross section's
    reaches: np.ndarray  # the reach each cell is on (a manual cross section's source stream)

    @property
    def count(self) -> int:
        return int(self.rows.size)

    def subset(self, keep: np.ndarray) -> Cells:
        return Cells(self.rows[keep], self.cols[keep], self.comids[keep], self.reaches[keep])


@dataclass
class Section:
    """A stream cell's cross section and what the run finds out about it."""
    xs: XSection
    row: int  # the cell it's centred on, which the low spot may have moved from the stream cell
    col: int
    direction: float  # the stream direction it was sampled in (arc.xsection.sampling)
    xs_angle: float  # legacy's XS_Angle: the cross section's own direction, from 0 up to pi
    rows: np.ndarray  # each ordinate's cell
    cols: np.ndarray
    side_one: int  # which half of the cross section, +1 or -1, legacy numbered side 1
    dem_low_point: float  # the elevation at its centre before any bathymetry
    land_cover: np.ndarray | None = None
    banks: Banks | None = None  # as found
    hydraulic_banks: Banks | None = None  # as the hydraulics use them: after the width filter with bathymetry
    bank_elevation: float = math.nan  # the smoothed bank elevation, with bathymetry
    usable: bool = True  # False once the bathymetry drops its reach
    manual: bool = False


@dataclass
class Results:
    """What a run made. The per-cell arrays and sections are in the order of cells, a section None where its
    centre had no data."""
    configs: Configs
    cells: Cells
    slopes: np.ndarray
    baseflow: np.ndarray
    max_flow: np.ndarray
    sections: list
    curves: RatingCurves | None = None
    vdt: pd.DataFrame | None = None
    cross_sections: list = field(default_factory=list)
    representative: pd.DataFrame | None = None
    bathymetry: np.ndarray | None = None
    seconds: float = 0.0


# --- 1. Inputs --------------------------------------------------------------------------------------------------------


def read_flows(configs: Configs) -> pd.DataFrame | None:
    """The flow file's baseflow and maximum-flow columns, indexed by its IDs (legacy read_flow_file). None for
    representative cross sections without baseflow bathymetry, which need no flows."""
    if configs.build_representative_cross_section and not configs.use_representative_baseflow_bathymetry:
        return None
    if not configs.flow_file or not configs.flow_file_id:
        raise ValueError("A Flow_File and its Flow_File_ID are needed for the rating curves.")
    path = str(configs.flow_file)
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    columns = [configs.flow_file_bf] if configs.use_representative_baseflow_bathymetry else \
        [c for c in (configs.flow_file_bf, configs.flow_file_qmax) if c]
    missing = [c for c in [configs.flow_file_id, *columns] if c not in df.columns]
    if missing:
        raise KeyError(f"The flow file {path} has no column {', '.join(missing)}.")
    df = df.set_index(configs.flow_file_id)[columns]
    if not df.index.is_unique:
        raise ValueError(f"The flow file {path} repeats IDs in {configs.flow_file_id}.")
    return df


def read_grid(configs: Configs) -> Grid:
    """The DEM, stream and land cover rasters, checked to be on one grid in metres or degrees."""
    rasters = {name: Raster(path) for name, path in (("DEM", configs.dem_file), ("stream", configs.stream_file),
                                                     ("land cover", configs.lu_raster_sameres))}
    reference = rasters["DEM"]
    for name, raster in rasters.items():
        if raster.is_rotated:
            raise ValueError(f"The {name} raster {raster.filepath} is rotated, which ARC can't use.")
        units = raster.axis_units()
        if not units or not units <= {"metre", "metres", "meter", "meters", "degree", "degrees"}:
            raise ValueError(f"The {name} raster {raster.filepath}'s coordinates are in "
                             f"{', '.join(sorted(units)) or 'no known units'}, not metres or degrees.")
        if not raster.same_grid(reference):
            raise ValueError(f"The {name} raster {raster.filepath} isn't on the DEM's grid: {raster.shape} cells at "
                             f"{raster.geotransform}, where the DEM has {reference.shape} at {reference.geotransform}.")

    dem = reference.read_array(np.float64)
    no_data = ~np.isfinite(dem)
    if reference.nodata_value is not None:
        no_data |= dem == reference.nodata_value
    dem[no_data] = OFF_RASTER_ELEVATION

    stream_raster = rasters["stream"]
    streams = stream_raster.read_array(np.int64)
    if stream_raster.nodata_value is not None and math.isfinite(stream_raster.nodata_value):
        streams[streams == int(stream_raster.nodata_value)] = 0
    streams[streams < 0] = 0

    land_cover = rasters["land cover"].read_array().astype(np.uint8)
    dx, dy = reference.cell_size_in_metres()
    LOG.info(f"Cell size {dx} m by {dy} m")
    return Grid(reference, dem, streams, land_cover, dx, dy)


def read_stream_layer(configs: Configs):
    """The stream network's features, or None if the run doesn't need them or there's no StrmShp_File."""
    needs = bool(configs.bathy_out_file) or configs.use_bathymetry_powerlaw or configs.use_bathymetry_powerlaw_width \
        or configs.stream_slope_method in ("reach_average", "local_average_corrected", "end_points")
    if not needs or not configs.strmshp_file:
        return None
    return Vector(configs.strmshp_file).to_geopandas()


def stream_network(configs: Configs, layer):
    """The reach network from the stream layer (legacy _build_reach_network_graph), with lengths in metres. None
    without the layer, or without its reach_id and downstream_reach_id fields when there's no bathymetry to need them
    (they then only fill in slopes that couldn't be resolved, which legacy only read them for)."""
    if layer is None:
        return None
    for column in (configs.reach_id, configs.downstream_reach_id):
        if not column or column not in layer.columns:
            if configs.bathy_out_file:
                raise ValueError(f"The stream network {configs.strmshp_file} has no field {column!r}.")
            LOG.warning(f"The stream network {configs.strmshp_file} has no field {column!r}, so slopes that can't "
                        "be resolved can't be filled in from the network.")
            return None
    return reach_network(layer[configs.reach_id].tolist(), layer[configs.downstream_reach_id].tolist(),
                         Vector.lengths_in_metres(layer))


def power_law_targets(configs: Configs, layer) -> dict[int, tuple[float | None, float | None]]:
    """Each reach's drainage-area bankfull depth and width, where configured (legacy
    build_bathymetry_geometry_dict)."""
    if not (configs.use_bathymetry_powerlaw or configs.use_bathymetry_powerlaw_width):
        return {}
    if layer is None:
        raise ValueError("The drainage-area power laws need a StrmShp_File.")
    field_name, reach_field = configs.drainage_area_field, configs.reach_id
    missing = sorted({reach_field, field_name} - set(layer.columns))
    if missing:
        raise KeyError("The stream vector dataset is missing the columns required for drainage-area bathymetry: "
                       + ", ".join(missing))
    table = layer[[reach_field, field_name]].dropna(subset=[reach_field, field_name])
    reach_ids = pd.to_numeric(table[reach_field], errors='raise').astype(np.int64)
    areas = pd.to_numeric(table[field_name], errors='raise')
    depth_pair = (configs.coefficient_depth, configs.exponent_depth) if configs.use_bathymetry_powerlaw \
        else (None, None)
    targets = {}
    for reach, area in zip(reach_ids, areas):
        reach = int(reach)
        if reach in targets:
            continue  # a reach's first feature counts, as legacy took it
        area = float(area)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(f"Drainage area for reach {reach} must be positive in field {field_name}.")
        depth, width = power_law_geometry(area, *depth_pair, configs.coefficient_width, configs.exponent_width)
        if depth is not None and not (math.isfinite(depth) and depth > 0.0):
            raise ValueError(f"Estimated depth for reach {reach} was not positive. Check coefficient_depth, "
                             "exponent_depth, and the drainage area values.")
        if width is not None and not (math.isfinite(width) and width > 0.0):
            raise ValueError(f"Estimated width for reach {reach} was not positive. Check coefficient_width, "
                             "exponent_width, and the drainage area values.")
        targets[reach] = (depth, width)
    return targets


# --- Manual cross sections ------------------------------------------------------------------------------------------


def _parse_array(value, dtype) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if value is None or (isinstance(value, float) and math.isnan(value)) or value in ("", "[]"):
        return np.array([], dtype=dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype)
    try:
        parsed = json.loads(value)
    except Exception:
        parsed = ast.literal_eval(value)
    return np.asarray(parsed, dtype=dtype)


def read_manual_sections(configs: Configs) -> dict[int, dict]:
    """The manual cross sections, by the ID they share with the flow file (legacy
    load_manual_cross_section_records)."""
    path = str(configs.manual_cross_sections_file)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t" if path.lower().endswith((".tsv", ".txt")) else ",")
    id_field = configs.reach_id if configs.build_representative_cross_section else configs.flow_file_id
    missing = sorted({id_field, *_REQUIRED_MANUAL_COLUMNS} - set(df.columns))
    if missing:
        raise KeyError("Manual cross-section file is missing required columns: " + ", ".join(missing))
    records = {}
    for _, row in df.iterrows():
        manual_id = int(row[id_field])
        record = {"row": int(row["Row"]), "col": int(row["Col"]), "xs_angle": float(row.get("XS_Angle", 0.0) or 0.0),
                  "ordinate_dist": float(row["Ordinate_Dist"])}
        for side in (1, 2):
            record[f"xs{side}_profile"] = _parse_array(row[f"XS{side}_Profile"], np.float64)
            record[f"lc{side}_profile"] = _parse_array(row[f"LC{side}_Profile"], np.uint8)
            record[f"xs{side}_row"] = _parse_array(row[f"XS{side}_Row"], np.int64)
            record[f"xs{side}_col"] = _parse_array(row[f"XS{side}_Col"], np.int64)
        if record["xs1_profile"].size == 0 or record["xs2_profile"].size == 0:
            raise ValueError(f"Manual cross section {manual_id} did not contain profile values on both sides.")
        for side in (1, 2):
            if not (record[f"xs{side}_profile"].size == record[f"lc{side}_profile"].size
                    == record[f"xs{side}_row"].size == record[f"xs{side}_col"].size):
                raise ValueError(f"Manual cross section {manual_id} has mismatched side-{side} array lengths.")
        source = row.get("Source_Stream_ID", manual_id)
        record["source_stream_id"] = manual_id if pd.isna(source) else int(source)
        records[manual_id] = record
    return records


def manual_section(record: dict) -> Section:
    """A manual cross section as a Section: side 2 on the left, side 1 on the right, the shorter padded with walls
    (legacy apply_manual_cross_section_data)."""
    half = max(record["xs1_profile"].size, record["xs2_profile"].size) - 1

    def both_sides(key, fill, dtype):
        sides = []
        for side in (2, 1):
            values = np.full(half + 1, fill, dtype=dtype)
            values[:record[f"{key.format(side)}"].size] = record[f"{key.format(side)}"]
            sides.append(values)
        return np.concatenate([sides[0][:0:-1], sides[1]])

    elevations = both_sides("xs{}_profile", OFF_RASTER_ELEVATION, np.float64)
    rows = both_sides("xs{}_row", -1, np.int64)
    cols = both_sides("xs{}_col", -1, np.int64)
    land_cover = both_sides("lc{}_profile", np.nan, np.float64)
    xs = XSection(elevations, np.empty(0), float(record["ordinate_dist"]))
    xs_angle = float(record["xs_angle"])
    return Section(xs, int(record["row"]), int(record["col"]), xs_angle + np.pi / 2, xs_angle, rows, cols, 1,
                   float(record["xs1_profile"][0]), land_cover, manual=True)


# --- 2. Stream cells and 3. slopes ---------------------------------------------------------------------------------


def stream_cells(configs: Configs, grid: Grid, flows: pd.DataFrame | None, manual: dict | None) -> Cells:
    """The cells to work on, in the stream raster's row order or the manual cross sections' (legacy _main)."""
    if configs.build_representative_cross_section:
        ids = np.unique(grid.streams)
        ids = ids[ids > 0]
    else:
        ids = np.asarray(flows.index, dtype=np.int64)
    if manual:
        shared = [int(i) for i in ids if int(i) in manual]
        if not shared:
            raise ValueError("No IDs were shared between the ARC flow file and the manual cross-section file.")
        return Cells(np.array([manual[i]["row"] for i in shared], dtype=np.int64),
                     np.array([manual[i]["col"] for i in shared], dtype=np.int64), np.array(shared, dtype=np.int64),
                     np.array([manual[i]["source_stream_id"] for i in shared], dtype=np.int64))
    rows, cols = np.nonzero(np.isin(grid.streams, ids))
    if rows.size == 0:
        raise ValueError("No stream cell has an ID in the flow file, so there is nothing to work on.")
    comids = grid.streams[rows, cols].astype(np.int64)
    return Cells(rows.astype(np.int64), cols.astype(np.int64), comids, comids.copy())


def _cells_by_reach(cells: Cells) -> dict[int, list[int]]:
    by_reach: dict[int, list[int]] = {}
    for k, reach in enumerate(cells.reaches):
        by_reach.setdefault(int(reach), []).append(k)
    return by_reach


def _local_slopes(grid: Grid, rows, cols, cells: Cells, ks, distance) -> np.ndarray:
    """The local average slope at each of cells ks, over their reach's stream cells (rows, cols). A manual cross
    section's cell off the reach's cells joins them for this."""
    have = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    extra = list(dict.fromkeys((int(cells.rows[k]), int(cells.cols[k])) for k in ks
                               if (int(cells.rows[k]), int(cells.cols[k])) not in have))
    if extra:
        rows = np.concatenate([rows, np.array([r for r, _ in extra], dtype=np.int64)])
        cols = np.concatenate([cols, np.array([c for _, c in extra], dtype=np.int64)])
        have = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    stations = along_stream_stations(rows, cols, grid.dx, grid.dy)
    values = local_average_slopes(grid.dem[rows, cols], rows, cols, stations, grid.dx, grid.dy, distance)
    return np.array([values[have[(int(cells.rows[k]), int(cells.cols[k]))]] for k in ks])


def cell_slopes(configs: Configs, grid: Grid, cells: Cells, network, layer) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's slope by the configured method, and which cells to keep: those whose reach's slope was resolved
    (legacy initialize_stream_slope_dictionaries and _get_cell_bathymetry_inputs)."""
    method = configs.stream_slope_method
    stream_reaches = stream_cells_by_reach(grid.streams)
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    distance = configs.gen_slope_dist
    slopes = np.full(cells.count, np.nan)
    keep = np.ones(cells.count, dtype=bool)
    by_reach = _cells_by_reach(cells)

    local = {}
    if method in ("local_average", "local_average_corrected"):
        for reach, ks in by_reach.items():
            local[reach] = _local_slopes(grid, *stream_reaches.get(reach, empty), cells, ks, distance)
        if method == "local_average":
            for reach, ks in by_reach.items():
                slopes[ks] = local[reach]
            return slopes, keep

    if method in ("reach_average", "local_average_corrected"):
        median, lower, upper = {}, {}, {}
        for reach, (rows, cols) in stream_reaches.items():
            stations = along_stream_stations(rows, cols, grid.dx, grid.dy)
            median[reach], lower[reach], upper[reach] = reach_median_slope(
                grid.dem[rows, cols], rows, cols, stations, grid.dx, grid.dy, distance, configs.slope_low_percentile,
                configs.slope_high_percentile)
        if network is not None:
            unresolved = fill_unresolved_reach_slopes(network, median, lower, upper)
        else:
            unresolved = {reach for reach in median if lower[reach] == upper[reach]}
            if unresolved:
                LOG.error(f"Bad streams found: {sorted(unresolved)}. Please provide a stream vector file to calculate "
                          "slopes.")
        for reach, ks in by_reach.items():
            if reach in unresolved or reach not in median:
                keep[ks] = False
            elif method == "reach_average":
                slopes[ks] = median[reach]
            else:
                slopes[ks] = corrected_local_slopes(local[reach], lower[reach], upper[reach])
        skipped = int(np.count_nonzero(~keep))
        if skipped:
            LOG.info(f"Bypassing {skipped} stream cells from {len(unresolved & set(by_reach))} streams with "
                     "unresolved slope estimates.")
        return slopes, keep

    # end_points
    if layer is None:
        raise ValueError("The 'end_points' stream slope method requires a strmshp_file.")
    id_field = configs.reach_id if configs.build_representative_cross_section else configs.flow_file_id
    if id_field not in layer.columns:
        raise KeyError(f"The stream network {configs.strmshp_file} has no field {id_field!r}.")
    dem = np.where(grid.dem >= OFF_RASTER_ELEVATION, np.nan, grid.dem)
    for reach, ks in by_reach.items():
        features = layer[layer[id_field] == reach]
        slope = math.nan
        if len(features) > 0:
            length = float(features.to_crs(features.estimate_utm_crs()).length.iloc[0])
            slope = end_point_slope(features.to_crs(grid.reference.crs).geometry.iloc[0], length, dem,
                                    grid.reference.geotransform)
        if math.isfinite(slope):
            slopes[ks] = max(slope, MIN_SLOPE)
        else:
            keep[ks] = False
            LOG.warning(f"Reach {reach}'s end-point slope couldn't be measured, so its cells are skipped.")
    return slopes, keep


def cell_flows(configs: Configs, flows: pd.DataFrame | None, cells: Cells) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's baseflow and maximum flow (legacy _build_flow_arrays), 0 where there are none."""
    baseflow = np.zeros(cells.count)
    max_flow = np.zeros(cells.count)
    if flows is None:
        return baseflow, max_flow
    missing = sorted(set(cells.comids.tolist()) - set(flows.index.tolist()))
    if missing:
        raise KeyError("The flow file has no flows for stream IDs " + ", ".join(map(str, missing[:10]))
                       + ("..." if len(missing) > 10 else ""))
    if configs.flow_file_bf:
        baseflow = flows[configs.flow_file_bf].reindex(cells.comids).to_numpy(np.float64)
    if configs.flow_file_qmax and configs.flow_file_qmax in flows.columns:
        max_flow = flows[configs.flow_file_qmax].reindex(cells.comids).to_numpy(np.float64)
    return baseflow, max_flow


def cell_targets(targets: dict, cells: Cells) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's drainage-area bankfull depth and width, NaN where there's none."""
    depth = np.full(cells.count, np.nan)
    width = np.full(cells.count, np.nan)
    if not targets:
        return depth, width
    missing = sorted({int(r) for r in cells.reaches} - set(targets))
    if missing:
        raise KeyError("The stream network has no drainage area for reach_id " + ", ".join(map(str, missing[:10]))
                       + ("..." if len(missing) > 10 else ""))
    for k, reach in enumerate(cells.reaches):
        d, w = targets[int(reach)]
        depth[k] = math.nan if d is None else d
        width[k] = math.nan if w is None else w
    return depth, width


# --- 4. Roughness ----------------------------------------------------------------------------------------------------


def _bounded(n: np.ndarray) -> np.ndarray:
    # legacy's corrections: n above 10 is 0.035, and n at or below 0 is 0.005
    n[n > 10] = 0.035
    n[n <= 0.0] = 0.005
    return n


def mannings_n_raster(configs: Configs, land_cover: np.ndarray) -> tuple[np.ndarray, float]:
    """Each cell's Manning's n from its land cover class (legacy read_manning_table), and the water class's n
    (legacy _set_in_bank_mannings_n_to_water), bounded as legacy bounded them, and with float32's precision as
    legacy's raster held them."""
    path = str(configs.lu_manning_n)
    table = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, sep='\t')
    lookup = np.zeros(256, dtype=np.float32)
    lookup[table.iloc[:, 0].astype(np.uint8).values] = table.iloc[:, 2].values
    n = _bounded(lookup[land_cover]).astype(np.float64)
    water_rows = table.loc[table.iloc[:, 0].astype(int) == configs.lc_water_value]
    if water_rows.empty:
        raise ValueError("The Manning's n table has no entry for the configured water class.")
    water_n = float(water_rows.iloc[-1, 2])
    if not math.isfinite(water_n):
        raise ValueError("Water-class Manning's n must be finite.")
    return n, float(_bounded(np.array([water_n], dtype=np.float32))[0])


# --- 5. Cross sections --------------------------------------------------------------------------------------------


def _side_one(direction: float) -> int:
    """Which half of a cross section sampled in this stream direction legacy numbered side 1: the half towards
    the higher rows, or along a row towards the higher columns."""
    xs_direction = direction - np.pi / 2
    sin, cos = math.sin(xs_direction), math.cos(xs_direction)
    return 1 if sin > _ALONG_A_ROW or (abs(sin) <= _ALONG_A_ROW and cos > 0.0) else -1


def sample_section(configs: Configs, grid: Grid, row: int, col: int, offsets: np.ndarray,
                   length: float) -> Section | None:
    """A stream cell's cross section (legacy _sample_cross_section_for_cell), with its elevations and land cover
    but not yet its roughness: centred on the low spot and turned to where the channel is narrowest. None if its
    centre has no data."""
    direction = stream_direction(grid.streams, row, col, configs.gen_dir_dist, grid.dx, grid.dy)
    if configs.low_spot_range > 0:
        row, col = low_spot_cell(grid.dem, row, col, direction, length, grid.dx, grid.dy, configs.low_spot_range)
    direction = narrowest_direction(grid.dem, row, col, direction, length, grid.dx, grid.dy, offsets)
    elevations, spacing = sample_elevations(grid.dem, row, col, direction, length, grid.dx, grid.dy)
    center = elevations.size // 2
    if not elevations[center] < OFF_RASTER_ELEVATION:
        return None
    rows, cols = ordinate_cells(row, col, direction, length, grid.dx, grid.dy)
    land_cover = sample_land_cover(grid.land_cover, row, col, direction, length, grid.dx, grid.dy) \
        if configs.findbanksbasedonlandcover else None
    return Section(XSection(elevations, np.empty(0), float(spacing)), int(row), int(col), float(direction),
                   float((direction - np.pi / 2) % np.pi), rows, cols, _side_one(direction), float(elevations[center]),
                   land_cover)


def sample_roughness(section: Section, grid: Grid, mannings_n: np.ndarray, length: float) -> None:
    """Give a section its Manning's n: sampled like its elevations, or for a manual one, each ordinate's cell's."""
    if section.manual:
        on_raster = (section.rows >= 0) & (section.rows < mannings_n.shape[0]) & (section.cols >= 0) \
            & (section.cols < mannings_n.shape[1])
        n = np.full(section.xs.elevations.size, OFF_RASTER_ELEVATION)
        n[on_raster] = mannings_n[section.rows[on_raster], section.cols[on_raster]]
    else:
        n, _ = sample_elevations(mannings_n, section.row, section.col, section.direction, length, grid.dx, grid.dy)
    section.xs.mannings_n = n


# --- 7. Bathymetry ---------------------------------------------------------------------------------------------------


def unresolved_banks(xs: XSection, target_width: float) -> Banks:
    """The banks of a channel that neither the DEM nor its reach resolved: as wide as its width prior if it has
    one, and otherwise one cell wide (legacy's one-cell fallback)."""
    if math.isfinite(target_width) and target_width > 0.0:
        banks = banks_for_width(xs, target_width)
        if banks.valid:
            return banks
    return single_cell_banks(xs)


def apply_bathymetry(configs: Configs, grid: Grid, cells: Cells, sections: list, network, slopes, baseflow,
                     target_depth, target_width) -> np.ndarray:
    """Smooth the banks, find, smooth and carve each channel, and return the filled bathymetry raster. With
    Bathy_Use_Banks, the sections of a reach given no bank elevation become unusable, as legacy dropped them. A
    channel whose banks are still unresolved after the width filter is carved at its width prior, or one cell wide
    (unresolved_banks)."""
    if network is None:
        raise ValueError("Bathymetry needs the stream network, from StrmShp_File, reach_id and downstream_reach_id.")
    use_banks = configs.bathy_use_banks
    by_reach: dict[int, list[int]] = {}
    for k, section in enumerate(sections):
        if section is not None:
            by_reach.setdefault(int(cells.reaches[k]), []).append(k)
    reaches = {reach: ReachSections(cells.rows[ks], cells.cols[ks], [sections[k].xs for k in ks],
                                    [sections[k].banks for k in ks]) for reach, ks in by_reach.items()}
    smoothed = smooth_bank_elevations(network, reaches, grid.dx, grid.dy)

    for reach, ks in list(by_reach.items()):
        result = smoothed[reach]
        if use_banks and not np.isfinite(result.bank_elevations).any():
            LOG.warning(f"Marking reach_id {reach} as unusable because it has no finite bank observations and no "
                        f"network-smoothed fallback. Removed {len(ks)} sampled records from downstream bathymetry "
                        "and rating-curve generation.")
            for k in ks:
                sections[k].usable = False
            del by_reach[reach], reaches[reach]
            continue
        for position, k in enumerate(ks):
            banks = result.banks[position]
            sections[k].hydraulic_banks = banks if banks.valid else unresolved_banks(sections[k].xs,
                                                                                     float(target_width[k]))
            sections[k].bank_elevation = float(result.bank_elevations[position])

    depths, applies = {}, {}
    for reach, ks in by_reach.items():
        depths[reach], applies[reach] = [], []
        for k in ks:
            section = sections[k]
            depth = bathymetry_depth(section.xs, section.hydraulic_banks, float(baseflow[k]), float(slopes[k]),
                                     trapezoid_height=configs.bathy_trap_h,
                                     target_depth=None if math.isnan(target_depth[k]) else float(target_depth[k]))
            depths[reach].append(depth.depth)
            applies[reach].append(depth.apply)
    channels = smooth_channel_depths(network, reaches, {reach: smoothed[reach] for reach in reaches}, depths,
                                     grid.dx, grid.dy, use_banks=use_banks,
                                     max_bed_grade=MAX_BED_GRADE if configs.bathy_bed_cap else None)

    carve = {}
    for reach, ks in by_reach.items():
        for position, k in enumerate(ks):
            if applies[reach][position]:
                carve[k] = float(channels[reach].depths[position])
    bathymetry = np.full(grid.dem.shape, np.nan, dtype=np.float32)
    for k in sorted(carve):  # in the cells' order, which decides how overlapping cross sections average
        section = sections[k]
        changed = carve_channel(section.xs, section.hydraulic_banks, carve[k], trapezoid_height=configs.bathy_trap_h,
                                bank_elevation=section.bank_elevation if use_banks else None)
        burn_into_raster(bathymetry, section.rows, section.cols, section.xs.elevations, changed)
    ground = None if use_banks else np.where(grid.dem >= OFF_RASTER_ELEVATION, np.nan, grid.dem).astype(np.float32)
    return fill_bathymetry_gaps(bathymetry, ground)


# --- 8. Rating curves ------------------------------------------------------------------------------------------------


def rating_curves(configs: Configs, grid: Grid, cells: Cells, sections: list, slopes, baseflow, max_flow,
                  roughness: DepthRoughness | None, quiet: bool = True) -> RatingCurves:
    """Each cell's rating curve and metadata (legacy calculate_hydraulic_data_for_cell), NaN where it has none."""
    increments = int(configs.vdt_database_numiterations)
    curves = RatingCurves.empty(cells.count, max(increments, 0))
    reach_average = configs.reach_average_curve_file
    for k in tqdm.tqdm(range(cells.count), disable=quiet):
        section = sections[k]
        comid = float(cells.comids[k])
        if section is None or not section.usable:
            if reach_average:
                # A cell without a usable cross section still gets metadata for the reach-average curve file
                row, col = (int(cells.rows[k]), int(cells.cols[k])) if section is None else (section.row, section.col)
                elevation = float(grid.dem[row, col])
                curves.metadata[k, _METADATA_BUT_BASEFLOW] = \
                    [comid, row, col, elevation, 0.0, np.nan if section is None else section.xs_angle, elevation]
            continue
        elevation = float(grid.dem[section.row, section.col])
        thalweg = float(section.xs.elevations[section.xs.elevations.size // 2])
        curve = rating_curve(section.xs, float(max_flow[k]), float(slopes[k]), increments,
                             baseflow=float(baseflow[k]), roughness=roughness,
                             slope_factor=configs.slope_adjustment_factor)
        if curve is None:
            if reach_average:
                curves.metadata[k, _METADATA_BUT_BASEFLOW] = \
                    [comid, section.row, section.col, elevation, slopes[k], section.xs_angle, elevation]
            continue
        if curve.valid:
            curves.increments[k] = curve.increments
            curves.metadata[k] = [comid, section.row, section.col, elevation, baseflow[k], slopes[k],
                                  section.xs_angle, thalweg]
        # The curve file's metadata, with the elevation before any bathymetry (legacy set_non_vdt_data)
        if reach_average or (configs.print_curve_file and curve.start >= 0 and curve.last > curve.start + 1):
            curves.metadata[k, _METADATA_BUT_BASEFLOW] = \
                [comid, section.row, section.col, section.dem_low_point, slopes[k], section.xs_angle, thalweg]
    return curves


# --- 9. Outputs -----------------------------------------------------------------------------------------------------


def cross_section_record(section: Section, comid: int) -> CrossSectionRecord:
    """A section's row of the cross-section file: each side from its centre out to its last ordinate on the raster,
    the sides as legacy numbered them (legacy _build_cross_section_export_record)."""
    elevations, n = section.xs.elevations, section.xs.mannings_n
    center = elevations.size // 2
    sides = []
    for step in (section.side_one, -section.side_one):
        index = center + step * np.arange(1 + _on_raster_ordinates(elevations, center, step))
        sides.append((elevations[index].astype(np.float64), n[index].astype(np.float64), int(section.rows[index[-1]]),
                      int(section.cols[index[-1]])))
    (xs1, n1, r1, c1), (xs2, n2, r2, c2) = sides
    return CrossSectionRecord(int(comid), section.row, section.col, xs1, float(section.xs.ordinate_distance), n1, xs2,
                              n2, r1, c1, r2, c2)


def write_outputs(configs: Configs, grid: Grid, results: Results, flows: pd.DataFrame | None) -> None:
    """Write the outputs the configuration names, in legacy's order."""
    curves = results.curves
    if curves is not None:
        if not curves.has_vdt_data() and configs.vdt_database_numiterations > 0:
            LOG.warning('No VDT data was generated, so no hydraulic output files will be created.')
        else:
            if configs.print_vdt_database:
                results.vdt = write_vdt(curves, configs.print_vdt_database)
            if configs.print_ap_database:
                write_ap(curves, configs.print_ap_database)
            qmax = flows[configs.flow_file_qmax] if flows is not None and configs.flow_file_qmax in flows.columns \
                else pd.Series(dtype=np.float64)
            if configs.reach_average_curve_file:
                vdt = results.vdt if results.vdt is not None else vdt_dataframe(curves)
                write_reach_average_curve_file(curves, vdt, qmax, configs.print_curve_file)
            elif configs.print_curve_file:
                write_curve_file(curves, qmax, configs.print_curve_file)
    if configs.xs_out_file:
        write_cross_sections(results.cross_sections, configs.xs_out_file)
    if configs.build_representative_cross_section and configs.representative_cross_section_file:
        write_representative_cross_sections(results.representative, configs.representative_cross_section_file)
    if results.bathymetry is not None:
        Raster.write_array(results.bathymetry, grid.reference, configs.bathy_out_file, dtype=np.float32,
                           compression=configs.compression)
    if configs.aroutflood:
        flood = np.zeros(grid.dem.shape, dtype=np.uint8)
        if curves is not None:
            for section in results.sections:
                if section is not None and section.usable:
                    flood[section.row, section.col] = FLOODED_CELL
        Raster.write_array(flood, grid.reference, configs.aroutflood, dtype=np.uint8,
                           compression=configs.compression)


# --- The whole run -------------------------------------------------------------------------------------------------


def run(configs: Configs, *, quiet: bool = False, write: bool = True) -> Results:
    """Run ARC for a configuration, write the outputs it names (unless write is False), and return what it made."""
    started = time.perf_counter()
    flows = read_flows(configs)
    grid = read_grid(configs)
    layer = read_stream_layer(configs)
    network = stream_network(configs, layer)
    targets = power_law_targets(configs, layer)
    manual = read_manual_sections(configs) if configs.manual_cross_sections_file else None

    length = float(configs.x_section_dist)
    if manual:
        needed = max(2.0 * (max(r["xs1_profile"].size, r["xs2_profile"].size) - 1) * r["ordinate_dist"]
                     for r in manual.values())
        if needed > length:
            LOG.info(f"Increasing X_Section_Dist from {length} to {needed} to accommodate the supplied manual "
                     "cross sections.")
            length = needed

    cells = stream_cells(configs, grid, flows, manual)
    slopes, keep = cell_slopes(configs, grid, cells, network, layer)
    if not keep.any():
        raise ValueError("No stream cells remain after bypassing streams with unresolved slope estimates.")
    cells, slopes = cells.subset(keep), slopes[keep]
    baseflow, max_flow = cell_flows(configs, flows, cells)
    target_depth, target_width = cell_targets(targets, cells)

    # The stream cells are water, whatever the land cover says
    land_cover = grid.land_cover.copy()
    land_cover[cells.rows, cells.cols] = configs.lc_water_value
    grid = grid._replace(land_cover=land_cover)
    mannings_n, water_n = mannings_n_raster(configs, land_cover)

    # Each cell's cross section and banks
    offsets = angle_offsets(configs.degree_manip, configs.degree_interval)
    sections: list[Section | None] = []
    for k in tqdm.tqdm(range(cells.count), disable=quiet):
        if manual:
            section = manual_section(manual[int(cells.comids[k])])
        else:
            section = sample_section(configs, grid, int(cells.rows[k]), int(cells.cols[k]), offsets, length)
        if section is not None:
            width = None if math.isnan(target_width[k]) else float(target_width[k])
            land = section.land_cover if configs.findbanksbasedonlandcover else None
            section.banks = find_banks(section.xs, target_width=width, land_cover=land,
                                       water_value=configs.lc_water_value if land is not None else None)
            section.hydraulic_banks = section.banks
        sections.append(section)

    # The water's n between each cross section's banks, put in the raster so that the cross sections crossing it get
    # it too (legacy _set_in_bank_mannings_n_to_water), and then each cross section's n
    for section in sections:
        if section is not None:
            wet = in_bank(section.xs, section.banks)
            rows, cols = section.rows[wet], section.cols[wet]
            on_raster = (rows >= 0) & (rows < mannings_n.shape[0]) & (cols >= 0) & (cols < mannings_n.shape[1])
            mannings_n[rows[on_raster], cols[on_raster]] = water_n
    for section in sections:
        if section is not None:
            sample_roughness(section, grid, mannings_n, length)
            set_in_bank_roughness(section.xs, section.banks, water_n)

    results = Results(configs, cells, slopes, baseflow, max_flow, sections)
    if configs.bathy_out_file:
        results.bathymetry = apply_bathymetry(configs, grid, cells, sections, network, slopes, baseflow,
                                              target_depth, target_width)
    usable = [(k, s) for k, s in enumerate(sections) if s is not None and s.usable]
    for _, section in usable:
        set_bank_distances(section.xs, section.hydraulic_banks)
    if configs.xs_out_file or (configs.build_representative_cross_section
                               and configs.representative_cross_section_file):
        results.cross_sections = [cross_section_record(s, cells.comids[k]) for k, s in usable]

    roughness = DepthRoughness(configs.k_decay, configs.shallow_factor, configs.deep_factor) \
        if configs.depth_varying_n else None
    if configs.build_representative_cross_section:
        samples = [RepresentativeSample(int(cells.comids[k]), s.xs, float(s.xs.elevations[s.xs.elevations.size // 2]),
                                        float(slopes[k])) for k, s in usable]
        results.representative = representative_cross_section_dataframe(
            samples, roughness=roughness, slope_factor=configs.slope_adjustment_factor)
    else:
        results.curves = rating_curves(configs, grid, cells, sections, slopes, baseflow, max_flow, roughness, quiet)

    if write:
        write_outputs(configs, grid, results, flows)
    results.seconds = time.perf_counter() - started
    LOG.info(f"Simulation Took {results.seconds:.1f} seconds")
    return results


def main(argv: list[str] | None = None) -> Results:
    """The command line: python -m arc.pipeline ARC_Input_File.yaml [--quiet]."""
    parser = argparse.ArgumentParser(description="Run ARC")
    parser.add_argument("config", help="ARC's input file, YAML or tab-separated text")
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide the progress bars")
    args = parser.parse_args(argv)
    return run(Configs.from_file(args.config), quiet=args.quiet)


if __name__ == "__main__":
    main()
