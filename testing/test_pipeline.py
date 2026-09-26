"""The whole workflow, arc.pipeline, on synthetic channels whose answers are known, and on real sites when their data
are on this machine.

The synthetic channel is a straight trapezoid along a row of 1 m cells: a 10 m wide bed, banks rising 1 m every 2 m
to 2 m high, and floodplains rising 1%, falling 0.001 per metre downstream, in two reaches. Along a row the cross
sections run along columns and every ordinate is a cell centre, so the sampled channel is exactly the trapezoid and
its rating curve is Manning's equation for a trapezoid.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr
from scipy.optimize import brentq
from shapely.geometry import LineString

from arc import pipeline
from arc.Automated_Rating_Curve_Generator import main as legacy_main
from arc.config import Configs
from arc.hydraulics import DepthRoughness, discharge
from arc.io import Raster
from arc.outputs.rating_curves import Q, T, WSE

gdal.UseExceptions()

ROWS, COLS, CENTER_ROW = 61, 240, 30
SLOPE = 0.001
BED = 100.0
N = 0.035
N_RASTER = float(np.float32(N))  # Manning's n as the n raster holds it, in float32 as legacy's did
Q_MAX = {11: 15.0, 12: 18.0}
BASEFLOW = {11: 2.0, 12: 2.5}
INCREMENTS = 10


def ground(distance: np.ndarray) -> np.ndarray:
    """The channel's height above its bed at a distance from its centre line."""
    return np.where(distance <= 5.0, 0.0, np.where(distance <= 9.0, (distance - 5.0) / 2.0,
                                                   2.0 + 0.01 * (distance - 9.0)))


def trapezoid_discharge(depth: float, slope: float = SLOPE, n: float = N_RASTER) -> float:
    area = depth * (10.0 + 2.0 * depth)
    perimeter = 10.0 + 2.0 * depth * math.sqrt(5.0)
    return area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / n


def normal_depth(q: float, slope: float = SLOPE, n: float = N_RASTER) -> float:
    return brentq(lambda y: trapezoid_discharge(y, slope, n) - q, 1e-6, 2.0, xtol=1e-14)


def _write_raster(path: Path, values: np.ndarray, geotransform, srs: osr.SpatialReference, gdal_type) -> str:
    ds = gdal.GetDriverByName("GTiff").Create(str(path), values.shape[1], values.shape[0], 1, gdal_type)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(values)
    ds = None
    return str(path)


def _projected() -> tuple[osr.SpatialReference, tuple]:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    return srs, (500000.0, 1.0, 0.0, 3800000.0, 0.0, -1.0)


def write_inputs(folder: Path, dem: np.ndarray, streams: np.ndarray, srs, geotransform, reaches: dict, *,
                 land_cover: np.ndarray | None = None, water_n: float = N, land_n: float = N) -> dict:
    """ARC's inputs for a DEM and stream raster, and an input mapping for them. reaches holds each reach's line, as
    cells (row, col) at its ends, its downstream reach (-1 for none), drainage area, baseflow and maximum flow."""
    folder.mkdir(parents=True, exist_ok=True)
    land_cover = np.ones(dem.shape, dtype=np.uint8) if land_cover is None else land_cover
    _write_raster(folder / "dem.tif", dem, geotransform, srs, gdal.GDT_Float64)
    _write_raster(folder / "streams.tif", streams.astype(np.int32), geotransform, srs, gdal.GDT_Int32)
    _write_raster(folder / "land.tif", land_cover, geotransform, srs, gdal.GDT_Byte)
    (folder / "mannings.txt").write_text(f"LC_ID\tDescription\tManning_n\n1\tGrass\t{land_n}\n80\tWater\t{water_n}\n",
                                         encoding="utf-8")
    ids = list(reaches)
    pd.DataFrame({"COMID": ids, "qbf": [reaches[r]["baseflow"] for r in ids],
                  "qmax": [reaches[r]["q_max"] for r in ids]}).to_csv(folder / "flows.csv", index=False)
    point = lambda cell: (geotransform[0] + (cell[1] + 0.5) * geotransform[1],
                          geotransform[3] + (cell[0] + 0.5) * geotransform[5])
    lines = [LineString([point(reaches[r]["start"]), point(reaches[r]["end"])]) for r in ids]
    gpd.GeoDataFrame({"LINKNO": ids, "COMID": ids, "DSLINKNO": [reaches[r]["downstream"] for r in ids],
                      "DA": [reaches[r].get("area", 50.0) for r in ids]}, geometry=lines,
                     crs=srs.ExportToWkt()).to_file(folder / "streams.gpkg")
    return {"DEM_File": str(folder / "dem.tif"), "Stream_File": str(folder / "streams.tif"),
            "LU_Raster_SameRes": str(folder / "land.tif"), "LU_Manning_n": str(folder / "mannings.txt"),
            "Flow_File": str(folder / "flows.csv"), "Flow_File_ID": "COMID", "Flow_File_BF": "qbf",
            "Flow_File_QMax": "qmax", "StrmShp_File": str(folder / "streams.gpkg"), "reach_id": "LINKNO",
            "downstream_reach_id": "DSLINKNO", "X_Section_Dist": 60, "Degree_Manip": 0, "Low_Spot_Range": 0,
            "Gen_Dir_Dist": 10, "Gen_Slope_Dist": 10, "Stream_Slope_Method": "local_average",
            "VDT_Database_NumIterations": INCREMENTS, "Print_VDT_Database": str(folder / "vdt.csv"),
            "Depth_Varying_N": False}


def write_channel(folder: Path, *, geographic: bool = False, water_n: float = N, land_n: float = N,
                  land_cover: np.ndarray | None = None, south=ground, bed=None) -> dict:
    """The synthetic channel's inputs (see the notes above), and an input mapping for them. south gives the ground
    south of the stream (rows below it) if not the same as the north's, and bed its bed elevation at each column
    distance in metres if it doesn't fall evenly."""
    folder.mkdir(parents=True, exist_ok=True)
    if geographic:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        latitude = 34.0
        # Cells of about 1 m: the degrees along each axis that the ellipsoid makes 1 m at this latitude
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        dy_deg = 1.0 / geod.inv(0.0, latitude, 0.0, latitude + 1.0)[2]
        dx_deg = 1.0 / geod.inv(0.0, latitude, 1.0, latitude)[2]
        geotransform = (-84.0, dx_deg, 0.0, latitude + 0.5 * ROWS * dy_deg, 0.0, -dy_deg)
    else:
        srs, geotransform = _projected()
    _write_raster(folder / "dem.tif", np.zeros((ROWS, COLS)), geotransform, srs, gdal.GDT_Float64)
    dx, dy = Raster(folder / "dem.tif").cell_size_in_metres()

    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    distance = np.abs(rows - CENTER_ROW) * dy
    bed_elevation = BED - SLOPE * cols * dx if bed is None else bed(cols * dx)
    dem = bed_elevation + np.where(rows > CENTER_ROW, south(distance), ground(distance))
    streams = np.zeros((ROWS, COLS), dtype=np.int32)
    streams[CENTER_ROW, 20:120] = 11
    streams[CENTER_ROW, 120:220] = 12
    reaches = {11: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 119), "downstream": 12, "area": 50.0,
                    "baseflow": BASEFLOW[11], "q_max": Q_MAX[11]},
               12: {"start": (CENTER_ROW, 120), "end": (CENTER_ROW, 219), "downstream": -1, "area": 60.0,
                    "baseflow": BASEFLOW[12], "q_max": Q_MAX[12]}}
    return write_inputs(folder, dem, streams, srs, geotransform, reaches, land_cover=land_cover, water_n=water_n,
                        land_n=land_n)


def interior(results: pipeline.Results) -> np.ndarray:
    """The cells more than a gen_slope_dist from either end of the stream, where the slope is the bed's."""
    cols = results.cells.cols
    return (cols >= 40) & (cols <= 200)


# --- The synthetic channel ----------------------------------------------------------------------------------------


def test_the_rating_curve_is_manning_s_for_the_trapezoid(tmp_path: Path) -> None:
    results = pipeline.run(Configs.from_mapping(write_channel(tmp_path)), quiet=True)
    inside = interior(results)

    assert inside.sum() == 161 and np.allclose(results.slopes[inside], SLOPE, rtol=1e-9)
    for k in np.flatnonzero(inside):
        reach = int(results.cells.comids[k])
        increments = results.curves.increments[k]
        bed = float(results.sections[k].xs.elevations[results.sections[k].xs.elevations.size // 2])
        assert increments[-1, WSE] - bed == pytest.approx(normal_depth(Q_MAX[reach]), abs=1e-9)
        for q, v, t, wse, p in increments:
            depth = wse - bed
            assert q == pytest.approx(trapezoid_discharge(depth), rel=1e-9)
            assert t == pytest.approx(10.0 + 4.0 * depth, rel=1e-12)
            assert p == pytest.approx(10.0 + 2.0 * depth * math.sqrt(5.0), rel=1e-12)
    vdt = pd.read_csv(tmp_path / "vdt.csv")
    assert len(vdt) == results.cells.count == 200
    assert (vdt[f"q_{INCREMENTS}"] == vdt["COMID"].map(Q_MAX)).all()  # rounded to 3 decimals
    assert np.allclose(vdt["XS_Angle"], math.pi / 2)  # the cross sections run along the columns


def test_depth_varying_roughness_and_the_slope_factor_reach_the_rating_curves(tmp_path: Path) -> None:
    inputs = {**write_channel(tmp_path), "Depth_Varying_N": True, "k_decay": 5.0, "shallow_factor": 2.0,
              "deep_factor": 0.9, "slope_adjustment_factor": 1.2}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    roughness = DepthRoughness(5.0, 2.0, 0.9)

    for k in np.flatnonzero(interior(results)):
        xs = results.sections[k].xs
        for q, _, _, wse, _ in results.curves.increments[k]:
            assert q == pytest.approx(discharge(xs, SLOPE, wse=wse, roughness=roughness, slope_factor=1.2), rel=1e-9)
        assert results.curves.increments[k, -1, Q] == pytest.approx(Q_MAX[int(results.cells.comids[k])], rel=1e-9)
    assert results.vdt is None  # nothing written


def test_the_water_s_roughness_goes_between_the_banks(tmp_path: Path) -> None:
    """With a rougher land cover, the channel between its banks (±9 m) still takes the water's n."""
    results = pipeline.run(Configs.from_mapping(write_channel(tmp_path, land_n=0.1)), quiet=True, write=False)
    section = results.sections[100]
    offsets = np.arange(section.xs.elevations.size) - section.xs.elevations.size // 2

    assert np.all(section.xs.mannings_n[np.abs(offsets) <= 9] == np.float32(N))
    assert np.all(section.xs.mannings_n[np.abs(offsets) >= 10] == np.float32(0.1))
    k = 100
    bed = float(section.xs.elevations[section.xs.elevations.size // 2])
    assert results.curves.increments[k, -1, WSE] - bed == pytest.approx(normal_depth(Q_MAX[12]), abs=1e-9)


def test_on_a_geographic_grid_the_cells_are_measured_in_metres(tmp_path: Path) -> None:
    """The cells are about 1 m, measured along the ellipsoid at the raster's middle latitude."""
    inputs = write_channel(tmp_path, geographic=True)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    inside = interior(results)
    _, dy = Raster(inputs["DEM_File"]).cell_size_in_metres()

    assert dy == pytest.approx(1.0, rel=1e-3)
    assert np.allclose(results.slopes[inside], SLOPE, rtol=1e-6)
    for k in np.flatnonzero(inside)[::20]:
        section = results.sections[k]
        assert section.xs.ordinate_distance == pytest.approx(dy, rel=1e-12)  # along a column
        bed = float(section.xs.elevations[section.xs.elevations.size // 2])
        # The trapezoid's corners fall a little between the ordinates, 0.9999 m apart
        assert results.curves.increments[k, -1, WSE] - bed == pytest.approx(
            normal_depth(Q_MAX[int(results.cells.comids[k])]), rel=1e-4)


@pytest.mark.parametrize("use_banks", [False, True])
def test_the_bathymetry_carves_a_channel_for_the_baseflow(tmp_path: Path, use_banks: bool) -> None:
    """Without bank elevations the channel is carved below the stream cell, which the gap fill keeps below the DEM.
    With them it is carved below the smoothed bank elevation, here the banks' top, 2 m above the bed."""
    inputs = {**write_channel(tmp_path), "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Use_Banks": use_banks,
              "XS_Out_File": str(tmp_path / "xs.txt")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    bathymetry = Raster(tmp_path / "bathy.tif").read_array()
    dem = Raster(tmp_path / "dem.tif").read_array()
    carved = np.isfinite(bathymetry)

    assert np.array_equal(np.isfinite(results.bathymetry), carved)
    assert carved[CENTER_ROW, 40:200].all()
    assert not carved[:CENTER_ROW - 10].any() and not carved[CENTER_ROW + 11:].any()  # only between the banks
    if not use_banks:
        assert np.all(bathymetry[carved] <= dem[carved] + 1e-6)
        assert np.all(bathymetry[CENTER_ROW, 40:200] < dem[CENTER_ROW, 40:200])
    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        assert section.usable
        bed = section.xs.elevations[section.xs.elevations.size // 2]
        reference = section.bank_elevation if use_banks else section.dem_low_point
        assert bed < reference
        # Each rating curve is on the carved cross section
        assert results.curves.increments[k, -1, Q] == pytest.approx(Q_MAX[int(results.cells.comids[k])], rel=1e-9)
    xs = pd.read_csv(tmp_path / "xs.txt", sep="\t")
    assert len(xs) == results.cells.count
    assert list(xs.columns) == ['COMID', 'Row', 'Col', 'XS1_Profile', 'Ordinate_Dist', 'Manning_N_Raster1',
                                'XS2_Profile', 'Manning_N_Raster2', 'r1', 'c1', 'r2', 'c2']


def test_the_drainage_area_depth_is_carved_as_it_is(tmp_path: Path) -> None:
    """A power-law target depth applies whatever the baseflow, and the bed smoothing then evens it out."""
    inputs = {**write_channel(tmp_path), "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Bed_Cap": False,
              "drainage_area_field": "DA", "coefficient_depth": 0.1, "exponent_depth": 0.5}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        depth = section.dem_low_point - section.xs.elevations[section.xs.elevations.size // 2]
        target = 0.1 * (50.0 if results.cells.comids[k] == 11 else 60.0) ** 0.5
        assert depth == pytest.approx(target, abs=0.01)


def test_representative_cross_sections(tmp_path: Path) -> None:
    inputs = {**write_channel(tmp_path), "Build_Representative_Cross_Section": True,
              "Representative_Cross_Section_File": str(tmp_path / "representative.csv")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    df = pd.read_csv(tmp_path / "representative.csv")

    assert results.curves is None and not (tmp_path / "vdt.csv").exists()
    assert sorted(df["COMID"].unique()) == [11, 12]
    assert df.groupby("COMID").size().tolist() == [250, 250]
    stage = df[(df["COMID"] == 12) & (df["Depth_Stage_Index"] == 10)].iloc[0]
    # 1 m deep in the trapezoid; the slopes at the reach's ends are the bed's too
    assert stage["Mean_Cross_Sectional_Area"] == pytest.approx(12.0, rel=1e-6)
    assert stage["Mean_Discharge"] == pytest.approx(trapezoid_discharge(1.0), rel=1e-4)


def test_the_command_line_writes_the_outputs(tmp_path: Path) -> None:
    import yaml

    inputs = write_channel(tmp_path)
    config = tmp_path / "ARC_Input_File.yaml"
    config.write_text(yaml.safe_dump({**inputs, "AROutFLOOD": str(tmp_path / "flood.tif")}), encoding="utf-8")

    results = pipeline.main([str(config), "--quiet"])

    assert (tmp_path / "vdt.csv").exists()
    flood = Raster(tmp_path / "flood.tif").read_array()
    assert flood.sum() == pipeline.FLOODED_CELL * results.cells.count
    assert np.all(flood[CENTER_ROW, 20:220] == pipeline.FLOODED_CELL)


def test_inputs_on_different_grids_are_refused(tmp_path: Path) -> None:
    inputs = write_channel(tmp_path)
    dem = Raster(inputs["DEM_File"])
    shifted = list(dem.geotransform)
    shifted[0] += 5.0
    ds = gdal.Open(inputs["LU_Raster_SameRes"], gdal.GA_Update)
    ds.SetGeoTransform(shifted)
    ds = None

    with pytest.raises(ValueError, match="isn't on the DEM's grid"):
        pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)


def test_cells_without_dem_data_are_walls(tmp_path: Path) -> None:
    """A strip of nodata beyond the right bank holds the water in, like the raster's edge."""
    inputs = write_channel(tmp_path)
    ds = gdal.Open(inputs["DEM_File"], gdal.GA_Update)
    band = ds.GetRasterBand(1)
    values = band.ReadAsArray()
    values[CENTER_ROW + 12:CENTER_ROW + 15] = -9999.0
    band.WriteArray(values)
    band.SetNoDataValue(-9999.0)
    ds = None

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    section = results.sections[100]
    strip = (section.rows >= CENTER_ROW + 12) & (section.rows < CENTER_ROW + 15)

    assert strip.sum() == 3 and np.all(section.xs.elevations[strip] == 9999.0)
    assert results.curves.increments[100, -1, Q] == pytest.approx(Q_MAX[12], rel=1e-9)
    # Water 3 m deep stops at the strip on its side, 12 m out, and on the other reaches the section's end, 30 m out
    from arc.hydraulics import top_widths
    center = section.xs.elevations.size // 2
    sides = top_widths(section.xs.elevations, 1.0, section.xs.elevations[center] + 3.0)
    assert sorted(sides) == pytest.approx([11.0, 30.0], abs=0.001)


def test_stream_cells_are_water_for_the_land_cover_banks(tmp_path: Path) -> None:
    """Legacy made every stream cell water in the land cover. With water in a band 5 cells either side of the stream
    but not at the stream cells themselves, the land cover gives banks only because of that, 5.5 m out."""
    land = np.ones((ROWS, COLS), dtype=np.uint8)
    land[CENTER_ROW - 5:CENTER_ROW + 6] = 80
    land[CENTER_ROW] = 1
    inputs = {**write_channel(tmp_path, land_cover=land), "FindBanksBasedOnLandCover": True}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        banks = results.sections[k].banks
        assert banks.method == "land_cover" and (banks.left, banks.right) == pytest.approx((5.5, 5.5))


def write_oblique_channel(folder: Path, degrees: float = 30.0, roughness: float = 0.0, **kwargs) -> dict:
    """The synthetic trapezoid along a line turned `degrees` from the rows, as one reach of 150 columns, with random
    bumps up to `roughness` metres high."""
    shape = (140, 180)
    theta = math.radians(degrees)
    first_row, first_col = 25, 20
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    along = (cols - first_col) * math.cos(theta) + (rows - first_row) * math.sin(theta)
    across = (rows - first_row) * math.cos(theta) - (cols - first_col) * math.sin(theta)
    dem = BED - SLOPE * along + ground(np.abs(across))
    dem += np.random.default_rng(12).uniform(0.0, roughness, shape)
    streams = np.zeros(shape, dtype=np.int32)
    for col in range(first_col, first_col + 150):
        streams[first_row + int(round((col - first_col) * math.tan(theta))), col] = 21
    end = (first_row + int(round(149 * math.tan(theta))), first_col + 149)
    reaches = {21: {"start": (first_row, first_col), "end": end, "downstream": -1, "baseflow": 2.0, "q_max": 15.0}}
    srs, geotransform = _projected()
    return write_inputs(folder, dem, streams, srs, geotransform, reaches, **kwargs)


def test_the_water_s_roughness_between_banks_reaches_other_cross_sections(tmp_path: Path) -> None:
    """Legacy put the water's n in the roughness raster at every cell between each cross section's banks, so every
    cross section crossing those cells takes it too. On an oblique channel the ordinates just beyond a cross
    section's banks are sampled partly from cells of its neighbours' channels, and are smoother than the land. The
    ordinates between its own banks are the water's exactly."""
    from arc.bathymetry import in_bank

    inputs = write_oblique_channel(tmp_path, land_n=0.1)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    water = float(np.float32(N))
    blended = 0
    for k in range(20, results.cells.count - 20):
        section = results.sections[k]
        xs = section.xs
        inside = in_bank(xs, section.banks)
        assert section.banks.valid and np.all(xs.mannings_n[inside] == water)
        offsets = np.abs(np.arange(xs.elevations.size) - xs.elevations.size // 2) * xs.ordinate_distance
        near = ~inside & (offsets <= max(section.banks.left, section.banks.right) + 1.5 * xs.ordinate_distance)
        blended += int(np.count_nonzero(xs.mannings_n[near] < 0.1 - 1e-9))
        assert np.all(xs.mannings_n[near] >= water - 1e-12)
    assert blended > 50


def test_the_cross_section_file_numbers_the_sides_as_legacy_did(tmp_path: Path) -> None:
    """With banks twice as steep south of the stream, legacy's side 1 is the south side, the higher rows."""
    steep = lambda d: np.where(d <= 5.0, 0.0, np.where(d <= 7.0, d - 5.0, 2.0 + 0.01 * (d - 7.0)))
    inputs = {**write_channel(tmp_path, south=steep), "XS_Out_File": str(tmp_path / "xs.txt")}
    legacy_main("", {**legacy_inputs(inputs, tmp_path), "XS_Out_File": str(tmp_path / "legacy_xs.txt")}, quiet=True)
    pipeline.run(Configs.from_mapping(inputs), quiet=True)
    parse = lambda text: np.array(text.strip("[]").split(), dtype=float)
    new = pd.read_csv(tmp_path / "xs.txt", sep="\t")
    old = pd.read_csv(tmp_path / "legacy_xs.txt", sep="\t")

    merged = old.merge(new, on=["COMID", "Row", "Col"], suffixes=("_legacy", "_new"))
    assert len(merged) == 200
    for _, row in merged.iterrows():
        for side in (1, 2):
            legacy_profile, profile = parse(row[f"XS{side}_Profile_legacy"]), parse(row[f"XS{side}_Profile_new"])
            count = min(legacy_profile.size, profile.size)  # legacy stopped one ordinate short of the raster's edge
            assert count >= 29 and np.allclose(legacy_profile[:count], profile[:count], atol=1e-4)
        xs1 = parse(row["XS1_Profile_new"])
        assert xs1[6] - xs1[5] == pytest.approx(1.0)  # the steep side
        assert row["r1_new"] == ROWS - 1 and row["c1_new"] == row["Col"]


def test_without_bank_elevations_the_bathymetry_never_rises_above_the_dem(tmp_path: Path) -> None:
    """On rough ground along an oblique channel the cross sections' elevations are interpolated, so carving them can
    leave a cell above its own ground; legacy's step that drops those cells before the gap fill (and the fill's own
    limit) catch it."""
    inputs = {**write_oblique_channel(tmp_path, roughness=0.3), "BATHY_Out_File": str(tmp_path / "bathy.tif")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    dem = Raster(inputs["DEM_File"]).read_array()
    carved = np.isfinite(results.bathymetry)

    assert carved.sum() > 1000
    assert np.all(results.bathymetry[carved] <= dem[carved] + 1e-6)


def test_a_cross_section_s_outlying_width_is_rebuilt_at_the_reach_median(tmp_path: Path) -> None:
    """Where the channel widens to a 20 m bed for three columns, those cross sections' banks are outliers of their
    reach's widths, so the bathymetry and the hydraulics use banks at the reach's median width instead."""
    wide_cols = range(95, 98)

    inputs = write_channel(tmp_path)
    dem = Raster(inputs["DEM_File"]).read_array()
    rows = np.arange(ROWS)
    for col in wide_cols:
        distance = np.abs(rows - CENTER_ROW) * 1.0
        dem[:, col] = BED - SLOPE * col + np.where(distance <= 10.0, 0.0, np.where(
            distance <= 14.0, (distance - 10.0) / 2.0, 2.0 + 0.01 * (distance - 14.0)))
    srs, geotransform = _projected()
    _write_raster(Path(inputs["DEM_File"]), dem, geotransform, srs, gdal.GDT_Float64)
    inputs = {**inputs, "BATHY_Out_File": str(tmp_path / "bathy.tif")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(np.isin(results.cells.cols, wide_cols)):
        section = results.sections[k]
        assert section.banks.top_width == pytest.approx(28.0, abs=1e-6)  # its own banks, at its water's edges
        assert section.hydraulic_banks.method == "target_width"
        assert section.hydraulic_banks.top_width == pytest.approx(18.0, abs=1e-6)
        assert (section.xs.left_bank_distance, section.xs.right_bank_distance) == pytest.approx((9.0, 9.0))


def write_valley(folder: Path, side, *, q_max: float = 2.0, **power_laws) -> dict:
    """A stream along a row of 10 m cells, falling 0.001 per metre, whose ground rises side(distance) metres either
    side, and power laws of the drainage area with these coefficients and exponents of 0: a constant width and
    depth."""
    srs, _ = _projected()
    geotransform = (500000.0, 10.0, 0.0, 3800000.0, 0.0, -10.0)
    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    dem = BED - SLOPE * cols * 10.0 + side(np.abs(rows - CENTER_ROW) * 10.0)
    streams = np.zeros((ROWS, COLS), dtype=np.int32)
    streams[CENTER_ROW, 20:120] = 11
    streams[CENTER_ROW, 120:220] = 12
    reaches = {11: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 119), "downstream": 12, "baseflow": 0.5,
                    "q_max": q_max},
               12: {"start": (CENTER_ROW, 120), "end": (CENTER_ROW, 219), "downstream": -1, "baseflow": 0.5,
                    "q_max": q_max}}
    inputs = {**write_inputs(folder, dem, streams, srs, geotransform, reaches), "X_Section_Dist": 400,
              "BATHY_Out_File": str(folder / "bathy.tif"), "Bathy_Bed_Cap": False, "drainage_area_field": "DA"}
    for name, value in power_laws.items():
        inputs.update({f"coefficient_{name}": value, f"exponent_{name}": 0.0})
    return inputs


def test_a_creek_narrower_than_a_cell_is_carved_and_rated_at_its_own_width(tmp_path: Path) -> None:
    """A stream cell 1 m below its neighbours, with a width prior of 4 m in its 10 m cells and a depth prior of 1.5 m,
    and bank elevations. The DEM can't resolve the creek, so it's a single-cell channel, 4 m wide: the raster gets its
    bed at the stream cell and nothing beside it, and the rating curve, within the channel, is Manning's for the
    4 m trapezoid, where the ordinates alone would have made it a V 20 m across."""
    inputs = write_valley(tmp_path, lambda d: np.where(d == 0.0, 0.0, 1.0 + 0.01 * (d - 10.0)), width=4.0,
                          depth=1.5)
    results = pipeline.run(Configs.from_mapping({**inputs, "Bathy_Use_Banks": True}), quiet=True, write=False)
    carved = np.isfinite(results.bathymetry)

    assert carved[CENTER_ROW, 40:200].all() and carved[:CENTER_ROW].sum() == carved[CENTER_ROW + 1:].sum() == 0
    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        center = section.xs.elevations.size // 2
        banks = section.hydraulic_banks
        assert banks.single_cell and (banks.left, banks.right) == (2.0, 2.0)
        assert banks.left_elevation == section.dem_low_point + 1.0  # the ordinate beside the stream cell
        bed = float(section.xs.elevations[center])
        depth = section.bank_elevation - bed
        assert depth == pytest.approx(1.5, abs=0.05)  # as the bed smoothing left it
        assert results.bathymetry[section.row, section.col] == pytest.approx(bed, abs=1e-4)
        assert section.xs.elevations[[center - 1, center + 1]].tolist() == [section.dem_low_point + 1.0] * 2
        for q, _, t, wse, p in results.curves.increments[k]:
            y = wse - bed
            side = 0.8 * y / depth  # each side slopes over 0.2 of the 4 m width, whatever the depth
            area = y * (2.4 + side)
            perimeter = 2.4 + 2 * math.hypot(side, y)
            assert y <= depth
            assert t == pytest.approx(2.4 + 2 * side, rel=1e-9)
            assert p == pytest.approx(perimeter, rel=1e-9)
            assert q == pytest.approx(area * (area / perimeter) ** (2 / 3) * math.sqrt(SLOPE) / N_RASTER, rel=1e-9)


@pytest.mark.parametrize("width", [30.0, None])
def test_a_channel_nothing_resolves_is_carved_at_its_width_prior_or_one_cell_wide(tmp_path: Path,
                                                                                    width: float | None) -> None:
    """A V rising 0.5 m a cell: its width-to-depth ratio never rises, and its flat water is 2 m out, too narrow to
    resolve, so no cross section in the reach has valid banks. A width prior wider than two cells doesn't make it a
    single cell either. The channel is carved at the prior's width, or without one, one cell wide."""
    inputs = write_valley(tmp_path, lambda d: 0.05 * d, **({} if width is None else {"width": width}))
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        assert not section.banks.valid
        banks = section.hydraulic_banks
        if width is None:
            assert banks.single_cell and banks.top_width == 10.0
        else:
            assert banks.method == "target_width" and banks.top_width == width
        offsets = (np.arange(section.xs.elevations.size) - section.xs.elevations.size // 2) * 10.0
        carved = section.xs.elevations < section.dem_low_point + 0.05 * np.abs(offsets) - 1e-9
        assert np.flatnonzero(carved).tolist() == np.flatnonzero(np.abs(offsets) < banks.top_width / 2).tolist()


def write_channel_and_plateau(folder: Path) -> dict:
    """The synthetic channel, and south of it a flat plateau with a reach of its own that joins nothing."""
    channel = write_channel(folder / "channel")
    dem = np.vstack([Raster(channel["DEM_File"]).read_array(), np.full((ROWS, COLS), BED + 10.0)])
    streams = np.vstack([Raster(channel["Stream_File"]).read_array(), np.zeros((ROWS, COLS), dtype=np.int32)])
    streams[ROWS + CENTER_ROW, 20:100] = 31
    reaches = {11: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 119), "downstream": 12, "baseflow": BASEFLOW[11],
                    "q_max": Q_MAX[11]},
               12: {"start": (CENTER_ROW, 120), "end": (CENTER_ROW, 219), "downstream": -1, "baseflow": BASEFLOW[12],
                    "q_max": Q_MAX[12]},
               31: {"start": (ROWS + CENTER_ROW, 20), "end": (ROWS + CENTER_ROW, 99), "downstream": -1,
                    "baseflow": 2.0, "q_max": 15.0}}
    srs, geotransform = _projected()
    return write_inputs(folder, dem, streams, srs, geotransform, reaches)


@pytest.mark.parametrize("use_banks", [True, False])
def test_a_reach_without_a_bank_elevation_is_dropped_with_bank_elevations(tmp_path: Path, use_banks: bool) -> None:
    """On the plateau no bank stands above the stream, so the network gives its reach no bank elevation. Legacy then
    dropped the reach from the bathymetry and the rating curves; here only with Bathy_Use_Banks, which needs it."""
    inputs = {**write_channel_and_plateau(tmp_path), "BATHY_Out_File": str(tmp_path / "bathy.tif"),
              "Bathy_Use_Banks": use_banks, "AROutFLOOD": str(tmp_path / "flood.tif")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    plateau = results.cells.comids == 31
    flood = Raster(tmp_path / "flood.tif").read_array()

    assert all(results.sections[k].usable for k in np.flatnonzero(~plateau))
    assert all(results.sections[k].usable != use_banks for k in np.flatnonzero(plateau))
    assert np.isnan(results.curves.increments[plateau]).all() == use_banks
    assert np.isnan(results.bathymetry[ROWS:]).all() == use_banks
    assert (flood[ROWS + CENTER_ROW, 20:100] == 0).all() == use_banks
    assert (flood[CENTER_ROW, 20:220] == pipeline.FLOODED_CELL).all()


@pytest.mark.parametrize("method", ["local_average", "reach_average", "local_average_corrected", "end_points"])
def test_each_cell_gets_its_reach_s_slope_by_the_method(tmp_path: Path, method: str) -> None:
    """On a bed that rises and falls a little as it falls, each method's slopes, as arc.xsection.slope gives them."""
    from arc.xsection.slope import corrected_local_slopes, local_average_slopes, reach_median_slope
    from arc.xsection.stream_path import along_stream_stations

    bumpy = lambda x: BED - SLOPE * x + 0.05 * np.sin(x / 7.0)
    inputs = {**write_channel(tmp_path, bed=bumpy), "Stream_Slope_Method": method, "Slope_Low_Percentile": 20,
              "Slope_High_Percentile": 60}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    dem = Raster(inputs["DEM_File"]).read_array()

    for reach in (11, 12):
        cells = results.cells.comids == reach
        rows, cols = results.cells.rows[cells], results.cells.cols[cells]
        stations = along_stream_stations(rows, cols, 1.0, 1.0)
        z = dem[rows, cols]
        local = local_average_slopes(z, rows, cols, stations, 1.0, 1.0, 10)
        median, lower, upper = reach_median_slope(z, rows, cols, stations, 1.0, 1.0, 10, 20, 60)
        expected = {"local_average": local, "reach_average": np.full(rows.size, median),
                    "local_average_corrected": corrected_local_slopes(local, lower, upper),
                    "end_points": np.full(rows.size, max(round(abs(z[-1] - z[0]) / 99.0, 8), 1e-4))}[method]
        np.testing.assert_allclose(results.slopes[cells], expected, rtol=1e-12)
    distinct = len(set(np.round(results.slopes, 12)))
    assert distinct == 2 if method in ("reach_average", "end_points") else distinct > 50


def write_manual_sections(folder: Path, inputs: dict, cols: list[int]) -> dict:
    """Manual cross sections, in legacy's file format, that are the synthetic channel's own along columns, one per
    ID from 1001, each on its reach's stream; and inputs using them."""
    dem = Raster(inputs["DEM_File"]).read_array()
    records = []
    for k, col in enumerate(cols):
        south, north = np.arange(CENTER_ROW, ROWS), np.arange(CENTER_ROW, -1, -1)
        records.append({"COMID": 1001 + k, "Row": CENTER_ROW, "Col": col, "Ordinate_Dist": 1.0, "XS_Angle": math.pi / 2,
                        "XS1_Profile": json_list(dem[south, col]), "XS2_Profile": json_list(dem[north, col]),
                        "LC1_Profile": json_list(np.ones(south.size, dtype=int)),
                        "LC2_Profile": json_list(np.ones(north.size, dtype=int)),
                        "XS1_Row": json_list(south), "XS1_Col": json_list(np.full(south.size, col)),
                        "XS2_Row": json_list(north), "XS2_Col": json_list(np.full(north.size, col)),
                        "Source_Stream_ID": 11 if col < 120 else 12})
    pd.DataFrame(records).to_csv(folder / "manual.tsv", sep="\t", index=False)
    pd.DataFrame({"COMID": [1001 + k for k in range(len(cols))], "qbf": 2.0,
                  "qmax": [Q_MAX[11 if c < 120 else 12] for c in cols]}).to_csv(folder / "manual_flows.csv", index=False)
    return {**inputs, "Manual_Cross_Sections_File": str(folder / "manual.tsv"),
            "Flow_File": str(folder / "manual_flows.csv"), "Print_VDT_Database": str(folder / "manual_vdt.csv")}


def json_list(values: np.ndarray) -> str:
    import json

    return json.dumps(np.asarray(values).tolist())


def test_manual_cross_sections_give_the_rating_curves_their_sampled_twins_give(tmp_path: Path) -> None:
    inputs = write_channel(tmp_path)
    cols = [45, 80, 150, 190]
    manual_inputs = write_manual_sections(tmp_path, inputs, cols)

    manual = pipeline.run(Configs.from_mapping(manual_inputs), quiet=True)
    sampled = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    assert manual.cells.comids.tolist() == [1001, 1002, 1003, 1004]
    assert manual.cells.reaches.tolist() == [11, 11, 12, 12]
    for k, col in enumerate(cols):
        twin = int(np.flatnonzero(sampled.cells.cols == col)[0])
        assert manual.slopes[k] == sampled.slopes[twin]
        np.testing.assert_allclose(manual.curves.increments[k], sampled.curves.increments[twin], rtol=1e-12)
    vdt = pd.read_csv(tmp_path / "manual_vdt.csv")
    assert vdt["COMID"].tolist() == [1001, 1002, 1003, 1004]
    legacy = legacy_inputs(manual_inputs, tmp_path)
    legacy_main("", legacy, quiet=True)
    old = pd.read_csv(tmp_path / "legacy_vdt.csv")
    merged = old.merge(vdt, on=["COMID", "Row", "Col"], suffixes=("_legacy", "_new"))
    assert len(merged) == 4
    assert np.allclose(merged[f"q_{INCREMENTS}_legacy"], merged[f"q_{INCREMENTS}_new"], rtol=0.01)
    assert np.allclose(merged[f"wse_{INCREMENTS}_legacy"], merged[f"wse_{INCREMENTS}_new"], atol=0.02)


# --- Against legacy ARC, on the synthetic channel -----------------------------------------------------------------


def legacy_inputs(inputs: dict, folder: Path) -> dict:
    """The same run for legacy ARC: its constant roughness is both factors 1."""
    legacy = {key: value for key, value in inputs.items() if key != "Depth_Varying_N"}
    legacy.update({"shallow_factor": 1.0, "deep_factor": 1.0, "Print_VDT_Database": str(folder / "legacy_vdt.csv")})
    return legacy


def test_the_rating_curves_match_legacy_s_on_the_synthetic_channel(tmp_path: Path) -> None:
    """Legacy found the maximum flow's water surface to within a centimetre and stepped the depth in millimetres,
    so its rating curves are within that of the exact ones here, where its geometry is exact too."""
    inputs = write_channel(tmp_path)
    legacy_main("", legacy_inputs(inputs, tmp_path), quiet=True)
    pipeline.run(Configs.from_mapping(inputs), quiet=True)
    legacy = pd.read_csv(tmp_path / "legacy_vdt.csv")
    new = pd.read_csv(tmp_path / "vdt.csv")

    merged = legacy.merge(new, on=["COMID", "Row", "Col"], suffixes=("_legacy", "_new"))
    assert len(merged) == len(legacy) == len(new) == 200
    for column in ("Elev", "QBaseflow"):
        assert (merged[f"{column}_legacy"] == merged[f"{column}_new"]).all()
    assert np.allclose(merged["XS_Angle_legacy"], merged["XS_Angle_new"])
    inside = (merged["Col"] >= 40) & (merged["Col"] <= 200)
    # Legacy read the DEM as float32, which puts its slopes within 0.2% of the bed's
    assert np.allclose(merged.loc[inside, "Slope_legacy"], merged.loc[inside, "Slope_new"], rtol=2e-3)
    for i in range(1, INCREMENTS + 1):
        # Legacy's depth steps were rounded to the millimetre, so the i-th is up to i/2 mm from the exact one
        assert np.allclose(merged[f"wse_{i}_legacy"], merged[f"wse_{i}_new"], atol=0.0006 * i + 0.01)
        assert np.allclose(merged[f"t_{i}_legacy"], merged[f"t_{i}_new"], atol=4 * (0.0006 * i + 0.01))
    assert np.allclose(merged[f"q_{INCREMENTS}_legacy"], merged[f"q_{INCREMENTS}_new"], rtol=0.01)


def test_the_workflow_is_faster_than_legacy_s(tmp_path: Path) -> None:
    inputs = write_channel(tmp_path)
    inputs = {**inputs, "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Degree_Manip": 30, "Degree_Interval": 5,
              "XS_Out_File": str(tmp_path / "xs.txt")}
    legacy = {**legacy_inputs(inputs, tmp_path), "XS_Out_File": str(tmp_path / "legacy_xs.txt"),
              "BATHY_Out_File": str(tmp_path / "legacy_bathy.tif")}
    configs = Configs.from_mapping(inputs)

    def seconds(function):
        start = time.perf_counter()
        function()
        return time.perf_counter() - start

    seconds(lambda: pipeline.run(configs, quiet=True))  # compiled code loads from its cache the first time
    seconds(lambda: legacy_main("", legacy, quiet=True))
    new = min(seconds(lambda: pipeline.run(configs, quiet=True)) for _ in range(3))
    old = min(seconds(lambda: legacy_main("", legacy, quiet=True)) for _ in range(3))
    assert new < old, (new, old)


# --- Real sites ---------------------------------------------------------------------------------------------------

SITES = Path(r"C:\Users\lrr43\Documents\masters\fim_sites_nencarta")
REAL_SITES = ["Salt_Creek_at_Wood_Dale_(2010)", "Blue_River_nr_Stanley_(2012)"]


def real_inputs(site: str, folder: Path) -> dict:
    import yaml

    path = SITES / site / "ARC_InputFiles" / "GEOGLOWS_ARC_Input_fabdem_Bathy.yaml"
    if not path.exists():
        pytest.skip(f"{site}'s data aren't on this machine")
    inputs = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("FSOutBATHY", "BathyWaterMask", "Comid_Flow_File"):
        inputs.pop(key, None)
    inputs.update({"Print_VDT_Database": str(folder / "vdt.parquet"), "BATHY_Out_File": str(folder / "bathy.tif"),
                   "AROutBATHY": str(folder / "bathy.tif"), "XS_Out_File": str(folder / "xs.txt"),
                   "Print_AP_Database": str(folder / "ap.parquet")})
    return inputs


@pytest.mark.parametrize("site", REAL_SITES)
def test_a_real_site_s_rating_curves_are_physically_consistent(tmp_path: Path, site: str) -> None:
    """Whatever the cross sections, banks and bathymetry, each written rating curve is Manning's discharge on its
    own cross section, rises with the water, and reaches the maximum flow unless the water spills there."""
    inputs = real_inputs(site, tmp_path)
    configs = Configs.from_mapping(inputs)
    results = pipeline.run(configs, quiet=True)
    roughness = DepthRoughness(configs.k_decay, configs.shallow_factor, configs.deep_factor)
    vdt = pd.read_parquet(tmp_path / "vdt.parquet")
    ap = pd.read_parquet(tmp_path / "ap.parquet")
    n = configs.vdt_database_numiterations

    assert len(vdt) > 0.8 * results.cells.count
    reached = 0
    for k in range(results.cells.count):
        increments = results.curves.increments[k]
        if np.isnan(increments[-1, Q]):
            continue
        section = results.sections[k]
        q, wse = increments[:, Q], increments[:, WSE]
        assert np.all(np.diff(q) >= 0.0) and np.all(np.diff(wse) >= 0.0)
        for i in range(1, n):  # the first increment's discharge can be the baseflow's
            assert q[i] == pytest.approx(discharge(section.xs, results.slopes[k], wse=wse[i], roughness=roughness,
                                                   slope_factor=configs.slope_adjustment_factor), rel=1e-9)
        assert 0.5 * results.max_flow[k] <= q[-1] <= 1.5 * results.max_flow[k]
        reached += abs(q[-1] / results.max_flow[k] - 1.0) < 1e-6
        assert np.all(increments[:, T] > 0.0)
    assert reached > 0.5 * len(vdt)
    # The AP database's area is discharge over velocity, which is the cross section's area
    merged = vdt.merge(ap, on=["COMID", "Row", "Col"], suffixes=("", "_ap"))
    assert np.allclose(merged[f"q_{n}"], merged[f"q_{n}_ap"])
    assert np.all(merged[[f"a_{i}" for i in range(1, n + 1)]].to_numpy() > 0.0)
    # The written tables and rasters line up with the DEM
    dem = Raster(configs.dem_file)
    bathymetry = Raster(tmp_path / "bathy.tif")
    assert bathymetry.same_grid(dem)
    assert vdt["Row"].between(0, dem.nrows - 1).all() and vdt["Col"].between(0, dem.ncols - 1).all()
    xs = pd.read_csv(tmp_path / "xs.txt", sep="\t")
    assert len(xs) == sum(1 for s in results.sections if s is not None and s.usable)


@pytest.mark.parametrize("site", REAL_SITES)
def test_a_real_site_agrees_with_legacy_where_the_methods_don_t_differ(tmp_path: Path, site: str) -> None:
    """The cells, their DEM elevations and baseflows are legacy's, and both carry about the maximum flow at the top
    of most rating curves. Everything else legitimately differs (see arc.pipeline and the modules' notes)."""
    inputs = real_inputs(site, tmp_path)
    legacy = {**inputs, "Print_VDT_Database": str(tmp_path / "legacy_vdt.parquet"),
              "BATHY_Out_File": str(tmp_path / "legacy_bathy.tif"), "AROutBATHY": str(tmp_path / "legacy_bathy.tif"),
              "XS_Out_File": None, "Print_AP_Database": None}
    legacy = {key: value for key, value in legacy.items() if value is not None}
    legacy_main("", legacy, quiet=True)
    pipeline.run(Configs.from_mapping(inputs), quiet=True)
    old = pd.read_parquet(tmp_path / "legacy_vdt.parquet")
    new = pd.read_parquet(tmp_path / "vdt.parquet")
    n = len([c for c in new.columns if c.startswith("q_")])

    merged = old.merge(new, on=["COMID", "Row", "Col"], suffixes=("_legacy", "_new"))
    assert len(merged) > 0.9 * len(old)
    assert (merged["Elev_legacy"] == merged["Elev_new"]).all()
    assert (merged["QBaseflow_legacy"] == merged["QBaseflow_new"]).all()
    close = np.abs(merged[f"q_{n}_legacy"] / merged[f"q_{n}_new"] - 1.0) < 0.02
    assert close.mean() > 0.7
