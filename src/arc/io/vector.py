"""A vector file on disk (shapefile, GeoPackage, GeoJSON, GeoParquet...), read and written with GDAL through pyogrio
and geopandas.

Modelled on nencarta's core Vector, without its metadata cache: a Vector reads its metadata from the file when it's
made, and its features whenever they're asked for.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS
from shapely.geometry import box

_PARQUET = (".parquet", ".geoparquet", ".pq")


class Vector:
    """A layer of features in a vector file."""

    def __init__(self, filepath: os.PathLike, layer: str | int | None = None, use_threads: bool = True):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Cannot find the vector file {self.filepath}.")
        self.layer = layer
        self.use_threads = use_threads
        if self.filepath.suffix.lower() in _PARQUET:
            self._projection, self._bbox, self._columns, self._feature_count = self._parquet_info()
            return
        info = pyogrio.read_info(self.filepath, layer=layer, force_total_bounds=True)
        if info["crs"] is not None:
            self._projection = info["crs"]
        else:
            warnings.warn(f"Could not determine the CRS of {self.filepath}. Assuming EPSG:4326.", stacklevel=2)
            self._projection = "EPSG:4326"
        self._bbox = tuple(float(v) for v in info["total_bounds"])
        self._columns = tuple(str(name) for name in info["fields"])
        self._feature_count = int(info["features"])

    def __repr__(self) -> str:
        return f"Vector({str(self.filepath)!r})"

    def _parquet_info(self):
        """A GeoParquet file's CRS, bounds, columns and row count, from its own metadata with pyarrow, which doesn't
        need GDAL's Parquet driver."""
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(self.filepath)
        geo = json.loads((parquet.metadata.metadata or {}).get(b"geo", b"{}").decode("utf-8"))
        geometry_column = geo.get("primary_column", "geometry")
        column = geo.get("columns", {}).get(geometry_column, {})
        crs = column.get("crs", "missing")
        if crs is None or crs == "missing":
            # GeoParquet leaves the CRS out for OGC:CRS84, longitude and latitude
            projection = "EPSG:4326"
        elif isinstance(crs, dict) and "id" in crs:
            projection = ":".join(map(str, crs["id"].values()))
        else:
            projection = CRS.from_user_input(crs).to_wkt()
        bbox = column.get("bbox")
        if bbox is None:
            bbox = gpd.read_parquet(self.filepath, columns=[geometry_column]).total_bounds
        columns = tuple(name for name in parquet.schema_arrow.names if name != geometry_column)
        return projection, tuple(float(v) for v in bbox), columns, int(parquet.metadata.num_rows)

    # --- Metadata ----------------------------------------------------------------------------------------------

    @property
    def projection(self) -> str:
        return self._projection

    @cached_property
    def crs(self) -> CRS:
        return CRS.from_user_input(self._projection)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._bbox

    @cached_property
    def epsg_4326_bbox(self) -> tuple[float, float, float, float]:
        if self.crs == CRS.from_epsg(4326):
            return self._bbox
        return tuple(float(v) for v in gpd.GeoSeries([box(*self._bbox)], crs=self.crs).to_crs(4326).total_bounds)

    @property
    def columns(self) -> tuple[str, ...]:
        """The attribute fields, without the geometry."""
        return self._columns

    def __len__(self) -> int:
        return self._feature_count

    # --- Features ----------------------------------------------------------------------------------------------

    def to_geopandas(self, bbox_epsg_4326: tuple[float, float, float, float] | None = None,
                     columns: list[str] | None = None) -> gpd.GeoDataFrame:
        """The features, with their geometry, optionally only those in a box given in longitude and latitude."""
        bbox = None
        if bbox_epsg_4326 is not None:
            bbox = tuple(bbox_epsg_4326)
            if self.crs != CRS.from_epsg(4326):
                bbox = tuple(gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(self.crs).total_bounds)
        if self.filepath.suffix.lower() in _PARQUET:
            try:
                return gpd.read_parquet(self.filepath, bbox=bbox, columns=columns, use_threads=self.use_threads)
            except ValueError as e:
                if "Specifying 'bbox' not supported for this Parquet file" not in str(e):
                    raise
                gdf = gpd.read_parquet(self.filepath, columns=columns, use_threads=self.use_threads)
                minx, miny, maxx, maxy = bbox
                return gdf.cx[minx:maxx, miny:maxy]
        return gpd.read_file(self.filepath, layer=self.layer, use_arrow=True, bbox=bbox, columns=columns)

    def read_table(self, columns: list[str] | None = None) -> pd.DataFrame:
        """The features' attributes, without their geometry."""
        if self.filepath.suffix.lower() in _PARQUET:
            return pd.read_parquet(self.filepath, columns=columns)
        return pyogrio.read_dataframe(self.filepath, layer=self.layer, columns=columns, read_geometry=False,
                                      use_arrow=True)

    @staticmethod
    def lengths_in_metres(gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Each feature's length in metres (legacy _measure_reach_geometry_length): along the ellipsoid for a
        geographic CRS, or converted from the CRS's linear unit for a projected one. NaN for a missing or empty
        geometry, and the length in the CRS's own units if its units can't be told."""
        lengths = np.full(len(gdf), math.nan)
        try:
            crs = CRS.from_user_input(gdf.crs)
        except Exception:
            crs = None
        geod = crs.get_geod() if crs is not None and crs.is_geographic else None
        to_metres = None
        if crs is not None and not crs.is_geographic:
            try:
                to_metres = float(crs.axis_info[0].unit_conversion_factor)
            except Exception:
                to_metres = None
        for k, geometry in enumerate(gdf.geometry):
            if geometry is None or geometry.is_empty:
                continue
            native = float(geometry.length)
            if not math.isfinite(native) or native <= 0.0:
                continue
            length = native
            if geod is not None:
                try:
                    geodesic = abs(float(geod.geometry_length(geometry)))
                    if math.isfinite(geodesic) and geodesic > 0.0:
                        length = geodesic
                except Exception:
                    pass
            elif to_metres is not None and math.isfinite(native * to_metres) and native * to_metres > 0.0:
                length = native * to_metres
            lengths[k] = length
        return lengths

    # --- Writing -----------------------------------------------------------------------------------------------

    @classmethod
    def save_any_geom(cls, gdf: gpd.GeoDataFrame, path: os.PathLike, **kwargs) -> Vector:
        """Write features to a file of the kind its extension says, replacing any there, and return it."""
        cls.delete_any_geom(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if Path(path).suffix.lower() in _PARQUET:
            gdf.to_parquet(path, **kwargs)
        else:
            gdf.to_file(path, **kwargs)
        return cls(path)

    @classmethod
    def delete_any_geom(cls, path: os.PathLike) -> None:
        """Delete a vector file, with a shapefile's sidecar files."""
        path = Path(path)
        if not path.exists():
            return
        if path.suffix.lower() == ".shp":
            for sidecar in path.parent.glob(f"{path.stem}.*"):
                sidecar.unlink()
            return
        path.unlink()
