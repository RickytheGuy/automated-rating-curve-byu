from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString

from arc.Automated_Rating_Curve_Generator import (_measure_reach_geometry_length, convert_cell_size,
                                                  read_and_pad_and_maybe_make_shared, write_output_raster)
from arc.io import Raster, Vector

gdal.UseExceptions()


def write_raster(path: Path, array: np.ndarray, geotransform, epsg: int, nodata=None) -> Path:
    driver = gdal.GetDriverByName("GTiff")
    types = {np.dtype(np.float32): gdal.GDT_Float32, np.dtype(np.int32): gdal.GDT_Int32,
             np.dtype(np.uint8): gdal.GDT_Byte, np.dtype(np.float64): gdal.GDT_Float64}
    ds = driver.Create(str(path), array.shape[1], array.shape[0], 1, types[array.dtype])
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array)
    ds = None
    return path


GEOGRAPHIC = (-84.5, 0.000277777777, 0.0, 34.2, 0.0, -0.000277777777)  # 1 arcsecond cells near Atlanta
PROJECTED = (500000.0, 10.0, 0.0, 3800000.0, 0.0, -10.0)


# --- Rasters --------------------------------------------------------------------------------------------------------


def test_a_raster_s_grid_and_values(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    raster = Raster(write_raster(tmp_path / "dem.tif", values, PROJECTED, 32616, nodata=-9999.0))

    assert raster.shape == (3, 4) and (raster.nrows, raster.ncols) == (3, 4)
    assert raster.geotransform == PROJECTED
    assert raster.resolution == (10.0, 10.0)
    assert raster.bbox == (500000.0, 3799970.0, 500040.0, 3800000.0)
    assert raster.nodata_value == -9999.0
    assert raster.dtype == np.float32
    assert raster.is_north_up and not raster.is_rotated
    assert raster.is_in_meters and not raster.is_in_degrees
    assert np.array_equal(raster.read_array(), values)
    assert raster.read_array(np.float64).dtype == np.float64


def test_a_missing_raster_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Raster(tmp_path / "missing.tif")


def test_the_file_isn_t_kept_open(tmp_path: Path) -> None:
    """So an output can be written over, or deleted, while a Raster of it is around."""
    path = write_raster(tmp_path / "dem.tif", np.ones((2, 2), np.float32), PROJECTED, 32616)
    raster = Raster(path)
    raster.read_array()
    Raster.write_array(np.zeros((2, 2), np.float32), raster, path)
    path.unlink()
    assert not path.exists()


def test_the_bounding_box_in_longitude_and_latitude_covers_a_projected_raster(tmp_path: Path) -> None:
    raster = Raster(write_raster(tmp_path / "dem.tif", np.ones((50, 60), np.float32), PROJECTED, 32616))
    west, south, east, north = raster.epsg_4326_bbox
    geographic = Raster(write_raster(tmp_path / "geo.tif", np.ones((2, 2), np.float32), GEOGRAPHIC, 4326))

    assert geographic.epsg_4326_bbox == geographic.bbox
    assert -87.001 < west < east < -86.9 and 34.3 < south < north < 34.4
    assert raster.is_in_meters and geographic.is_in_degrees


@pytest.mark.parametrize("geotransform, epsg", [(GEOGRAPHIC, 4326), (PROJECTED, 32616),
                                                 ((-120.0, 0.001, 0.0, 45.0, 0.0, -0.0005), 4269)])
def test_cell_sizes_in_metres_match_legacy(tmp_path: Path, geotransform, epsg: int) -> None:
    raster = Raster(write_raster(tmp_path / "dem.tif", np.ones((1234, 567), np.float32), geotransform, epsg))
    gt = raster.geotransform
    legacy = convert_cell_size(gt[1], gt[5], gt[3] - raster.nrows * abs(gt[5]), gt[3], raster.projection)

    assert raster.cell_size_in_metres() == (legacy[0], legacy[1])


def test_a_raster_projected_in_feet_has_no_cell_size_in_metres(tmp_path: Path) -> None:
    raster = Raster(write_raster(tmp_path / "dem.tif", np.ones((2, 2), np.float32), PROJECTED, 2240))  # US feet

    with pytest.raises(ValueError, match="not metres"):
        raster.cell_size_in_metres()


def test_the_same_grid(tmp_path: Path) -> None:
    dem = Raster(write_raster(tmp_path / "dem.tif", np.ones((3, 4), np.float32), PROJECTED, 32616))
    streams = Raster(write_raster(tmp_path / "streams.tif", np.ones((3, 4), np.int32), PROJECTED, 32616))
    shifted = Raster(write_raster(tmp_path / "shifted.tif", np.ones((3, 4), np.int32),
                                  (PROJECTED[0] + 5.0,) + PROJECTED[1:], 32616))
    smaller = Raster(write_raster(tmp_path / "smaller.tif", np.ones((3, 3), np.int32), PROJECTED, 32616))

    assert dem.same_grid(streams)
    assert not dem.same_grid(shifted) and not dem.same_grid(smaller)


def test_bounds_intersect() -> None:
    assert Raster.bounds_intersect((0, 0, 2, 2), (1, 1, 3, 3))
    assert not Raster.bounds_intersect((0, 0, 1, 1), (1, 0, 2, 1))  # touching edges don't overlap


def test_writing_a_raster_matches_legacy_s_file(tmp_path: Path) -> None:
    """The same values and grid as legacy write_output_raster, compressed with ZSTD in 512 × 512 tiles without a
    predictor, where legacy used LZW in strips with horizontal differencing."""
    reference = Raster(write_raster(tmp_path / "dem.tif", np.ones((40, 30), np.float32), PROJECTED, 32616))
    values = np.random.default_rng(1).normal(100, 5, (40, 30)).astype(np.float32)
    values[3:7, 4:9] = np.nan
    written = Raster.write_array(values, reference, tmp_path / "out" / "bathymetry.tif", dtype=np.float32)
    write_output_raster(str(tmp_path / "legacy.tif"), values, 30, 40, reference.geotransform, reference.projection,
                        "GTiff", gdal.GDT_Float32)
    legacy = gdal.Open(str(tmp_path / "legacy.tif"))
    new = gdal.Open(str(written.filepath))

    assert np.array_equal(new.ReadAsArray(), legacy.ReadAsArray(), equal_nan=True)
    assert new.GetGeoTransform() == legacy.GetGeoTransform()
    assert new.GetProjection() == legacy.GetProjection()
    structure = new.GetMetadata("IMAGE_STRUCTURE")
    assert structure.get("COMPRESSION") == "ZSTD" and "PREDICTOR" not in structure
    assert legacy.GetMetadata("IMAGE_STRUCTURE").get("PREDICTOR") == "2"
    assert new.GetRasterBand(1).GetBlockSize() == [512, 512]
    assert new.GetRasterBand(1).GetNoDataValue() is None
    new = legacy = None


@pytest.mark.parametrize("compression", ["LZW", "DEFLATE", "LZMA"])
def test_any_compressed_raster_is_tiled_without_a_predictor(tmp_path: Path, compression: str) -> None:
    reference = Raster(write_raster(tmp_path / "dem.tif", np.ones((40, 30), np.float32), PROJECTED, 32616))
    values = np.full((40, 30), np.nan, np.float32)
    values[10:12, 5:20] = 97.25

    written = Raster.write_array(values, reference, tmp_path / "bathymetry.tif", compression=compression.lower())

    ds = gdal.Open(str(written.filepath))
    structure = ds.GetMetadata("IMAGE_STRUCTURE")
    assert structure.get("COMPRESSION") == compression and "PREDICTOR" not in structure
    assert ds.GetRasterBand(1).GetBlockSize() == [512, 512]
    ds = None
    assert np.array_equal(written.read_array(), values, equal_nan=True)


def test_writing_a_raster_with_nodata_and_other_compressions(tmp_path: Path) -> None:
    reference = Raster(write_raster(tmp_path / "dem.tif", np.ones((4, 5), np.float32), PROJECTED, 32616))
    flood = np.zeros((4, 5), np.uint8)
    flood[1, 2] = 3
    written = Raster.write_array(flood, reference, tmp_path / "flood.tif", compression="NONE", nodata_value=255)

    assert written.dtype == np.uint8 and written.nodata_value == 255
    assert np.array_equal(written.read_array(), flood)
    ds = gdal.Open(str(written.filepath))
    assert ds.GetRasterBand(1).GetBlockSize()[0] == 5  # uncompressed rasters stay in strips, unpadded
    ds = None
    with pytest.raises(ValueError, match="reference raster"):
        Raster.write_array(np.zeros((3, 3)), reference, tmp_path / "wrong.tif")


def test_reading_is_at_least_as_fast_as_legacy(tmp_path: Path) -> None:
    """Legacy read each raster into a padded array; ARC no longer pads."""
    path = write_raster(tmp_path / "dem.tif", np.random.default_rng(2).random((1500, 1500), dtype=np.float32),
                        PROJECTED, 32616)
    raster = Raster(path)

    def best(function, repeats=5):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            times.append(time.perf_counter() - start)
        return min(times)

    new = best(lambda: raster.read_array(np.float32))
    legacy = best(lambda: read_and_pad_and_maybe_make_shared(str(path), 1, 10, np.float32, "_DEM"))
    assert new <= legacy * 1.1, (new, legacy)


# --- Vectors --------------------------------------------------------------------------------------------------------


def stream_lines(crs) -> gpd.GeoDataFrame:
    lines = [LineString([(-87.0, 34.0), (-86.99, 34.005), (-86.98, 34.0)]),
             MultiLineString([[(-86.98, 34.0), (-86.97, 34.0)], [(-86.97, 34.0), (-86.97, 34.02)]]),
             LineString([(-86.97, 34.02), (-86.95, 34.03)])]
    return gpd.GeoDataFrame({"LINKNO": [1, 2, 3], "DSLINKNO": [2, 3, -1], "area": [1.0, 2.0, 3.5]},
                            geometry=lines, crs=crs)


def test_a_vector_s_metadata_and_features(tmp_path: Path) -> None:
    path = tmp_path / "streams.gpkg"
    stream_lines("EPSG:4326").to_file(path)
    vector = Vector(path)

    assert vector.crs == CRS.from_epsg(4326)
    assert vector.bbox == pytest.approx((-87.0, 34.0, -86.95, 34.03))
    assert vector.epsg_4326_bbox == vector.bbox
    assert vector.columns == ("LINKNO", "DSLINKNO", "area")
    assert len(vector) == 3
    assert vector.to_geopandas()["LINKNO"].tolist() == [1, 2, 3]
    assert list(vector.read_table(["LINKNO", "DSLINKNO"]).columns) == ["LINKNO", "DSLINKNO"]
    assert vector.to_geopandas(bbox_epsg_4326=(-86.965, 34.015, -86.9, 34.04))["LINKNO"].tolist() == [3]


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:32616", "EPSG:2240"])
def test_lengths_in_metres_match_legacy(crs: str) -> None:
    lines = stream_lines("EPSG:4326")
    if crs != "EPSG:4326":
        lines = lines.to_crs(crs)
    legacy = [_measure_reach_geometry_length(geometry, lines.crs) for geometry in lines.geometry]

    assert Vector.lengths_in_metres(lines).tolist() == legacy
    assert Vector.lengths_in_metres(lines)[0] == pytest.approx(2155.0, rel=1e-3)  # along the ellipsoid


def test_a_missing_or_empty_geometry_has_no_length() -> None:
    lines = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[None, LineString()], crs="EPSG:4326")

    assert np.isnan(Vector.lengths_in_metres(lines)).all()


def test_saving_and_deleting_any_kind_of_vector_file(tmp_path: Path) -> None:
    lines = stream_lines("EPSG:4326")
    shapefile = Vector.save_any_geom(lines, tmp_path / "streams.shp")
    parquet = Vector.save_any_geom(lines, tmp_path / "streams.parquet")

    assert len(shapefile) == 3 and len(parquet) == 3
    assert parquet.crs == CRS.from_epsg(4326)
    assert parquet.read_table(["LINKNO"])["LINKNO"].tolist() == [1, 2, 3]
    Vector.save_any_geom(lines.iloc[:1], tmp_path / "streams.shp")  # replaces the file
    assert len(Vector(tmp_path / "streams.shp")) == 1
    Vector.delete_any_geom(tmp_path / "streams.shp")
    assert not list(tmp_path.glob("streams.s*")) and not list(tmp_path.glob("streams.dbf"))
    Vector.delete_any_geom(tmp_path / "streams.parquet")
    Vector.delete_any_geom(tmp_path / "never_there.gpkg")
    assert not (tmp_path / "streams.parquet").exists()
