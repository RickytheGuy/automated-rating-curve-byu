from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arc.bathymetry import carve_channel, find_banks
from arc.hydraulic_data import HydraulicData, build_representative_cross_section_dataframe
from arc.hydraulics import DepthRoughness, hydraulic_geometry
from arc.outputs import (REPRESENTATIVE_CROSS_SECTION_COLUMNS, XS_EXPORT_COLUMNS, CrossSectionRecord, RatingCurves,
                         RepresentativeSample, ap_dataframe, curve_file_dataframe, format_array,
                         reach_average_curve_file_dataframe, representative_cross_section_dataframe, vdt_dataframe,
                         write_ap, write_cross_sections, write_curve_file, write_reach_average_curve_file,
                         write_representative_cross_sections, write_table, write_vdt)
from arc.xsection.xsection import XSection

INCREMENTS = 6


def legacy_writer(tmp_path: Path, output_array: np.ndarray, **files) -> HydraulicData:
    params = {"s_output_ap_database": str(tmp_path / files.get("ap", "legacy_ap.csv")),
              "s_output_vdt_database": str(tmp_path / files.get("vdt", "legacy_vdt.csv")),
              "s_output_curve_file": str(tmp_path / files.get("curve", "legacy_curve.csv")),
              "i_number_of_increments": INCREMENTS, "b_reach_average_curve_file": files.get("reach_average", False),
              "s_xs_output_file": str(tmp_path / "legacy_xs.txt"), "b_build_representative_cross_section": False,
              "s_representative_cross_section_file": "", "b_modified_dem": False}
    writer = HydraulicData(params)
    writer.associate_with_output_data(output_array)
    return writer


def random_curves(rng, cells: int = 60) -> RatingCurves:
    """Rating curves shaped like ARC's, with cells lacking curves, repeated rows, negative values, partial rows."""
    curves = RatingCurves.empty(cells, INCREMENTS)
    for k in range(cells):
        comid = 1000 + k // 12
        base = 100.0 + rng.uniform(0, 30)
        curves.metadata[k] = [comid, rng.integers(0, 500), rng.integers(0, 500), base + rng.uniform(0, 2),
                              rng.uniform(0.5, 20), rng.uniform(1e-4, 0.02), rng.uniform(0, math.pi), base]
        q_top = rng.uniform(5, 500)
        depth = np.linspace(0.3, 1.0, INCREMENTS) * rng.uniform(1, 6)
        q = q_top * (depth / depth[-1]) ** rng.uniform(1.3, 2.2)
        top = rng.uniform(5, 30) * depth ** rng.uniform(0.2, 0.6)
        area = top * depth * rng.uniform(0.5, 0.8)
        curves.increments[k] = np.column_stack([q, q / area, top, base + depth, top + 2 * depth])
    curves.metadata[3] = np.nan  # no rating curve
    curves.increments[3] = np.nan
    curves.increments[5, 2, 0] = np.nan  # an incomplete one
    curves.increments[7, 1, 1] = -0.5  # a negative velocity
    curves.metadata[9] = curves.metadata[8]  # a repeat of the cell before
    curves.increments[9] = curves.increments[8]
    return curves


def assert_same_file(new: Path, legacy: Path) -> None:
    assert new.read_bytes() == legacy.read_bytes()


def test_the_vdt_database_is_legacy_s(tmp_path: Path) -> None:
    curves = random_curves(np.random.default_rng(1))
    legacy = legacy_writer(tmp_path, curves.output_array())

    expected = legacy.save_vdt()
    written = write_vdt(curves, tmp_path / "vdt.csv")

    pd.testing.assert_frame_equal(written, expected)
    assert_same_file(tmp_path / "vdt.csv", tmp_path / "legacy_vdt.csv")
    assert len(written) == 60 - 4  # without the cell with no curve, the incomplete one, the negative one and the repeat
    assert list(written.columns[:7]) == ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle']


def test_the_vdt_database_as_parquet_is_legacy_s(tmp_path: Path) -> None:
    curves = random_curves(np.random.default_rng(2))
    legacy_writer(tmp_path, curves.output_array(), vdt="legacy_vdt.parquet").save_vdt()

    write_vdt(curves, tmp_path / "vdt.parquet")

    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "vdt.parquet"),
                                  pd.read_parquet(tmp_path / "legacy_vdt.parquet"))


def test_parquet_tables_take_a_dictionary_only_where_values_repeat(tmp_path: Path) -> None:
    """Brotli, and dictionary encoding only on the columns where under 30% of the values differ."""
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"COMID": np.repeat([7, 8, 9], 100), "Row": np.arange(300),
                       "v_1": rng.choice([0.5, 0.75, 1.125], 300), "wse_1": np.round(rng.uniform(100, 200, 300), 3)})

    write_table(df, tmp_path / "table.parquet")

    row_group = pq.ParquetFile(tmp_path / "table.parquet").metadata.row_group(0)
    columns = {row_group.column(i).path_in_schema: row_group.column(i) for i in range(row_group.num_columns)}
    assert {name for name, column in columns.items() if 'RLE_DICTIONARY' in column.encodings} == {"COMID", "v_1"}
    assert {column.compression for column in columns.values()} == {'BROTLI'}
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "table.parquet"), df)


def test_the_ap_database_is_legacy_s(tmp_path: Path) -> None:
    curves = random_curves(np.random.default_rng(3))
    curves.increments[11, 2, 1] = 0.0  # no velocity, so no area
    legacy_writer(tmp_path, curves.output_array()).save_ap()

    written = write_ap(curves, tmp_path / "ap.csv")

    assert_same_file(tmp_path / "ap.csv", tmp_path / "legacy_ap.csv")
    assert (written["a_3"] == 0).sum() >= 1
    pd.testing.assert_frame_equal(ap_dataframe(curves), written)


def test_the_curve_file_is_legacy_s(tmp_path: Path) -> None:
    curves = random_curves(np.random.default_rng(4))
    qmax = pd.Series({1000 + k: 100.0 + k for k in range(5)})
    flows = {comid: {"qmax": value} for comid, value in qmax.items()}
    legacy_writer(tmp_path, curves.output_array()).save_curve_file(flows, "qmax")

    written = write_curve_file(curves, qmax, tmp_path / "curve.csv")

    assert_same_file(tmp_path / "curve.csv", tmp_path / "legacy_curve.csv")
    assert len(written) > 40
    assert list(written.columns) == ['COMID', 'Row', 'Col', 'BaseElev', 'DEM_Elev', 'QMax', 'Slope', 'XS_Angle',
                                     'depth_a', 'depth_b', 'tw_a', 'tw_b', 'vel_a', 'vel_b']
    pd.testing.assert_frame_equal(curve_file_dataframe(curves, qmax), written)


def test_the_reach_average_curve_file_is_legacy_s(tmp_path: Path) -> None:
    curves = random_curves(np.random.default_rng(5))
    # A cell with no rating curve that still has metadata, as the reach-average curve file gives it
    curves.metadata[3] = [1000, 7, 8, 101.0, np.nan, 0.001, 1.0, 101.0]
    qmax = pd.Series({1000 + k: 100.0 + k for k in range(5)})
    flows = {comid: {"qmax": value} for comid, value in qmax.items()}
    legacy = legacy_writer(tmp_path, curves.output_array(), reach_average=True)
    vdt = legacy.save_vdt()
    legacy.save_reach_average_curve_file(vdt, flows, "qmax")

    written = write_reach_average_curve_file(curves, vdt_dataframe(curves), qmax, tmp_path / "curve.csv")

    assert_same_file(tmp_path / "curve.csv", tmp_path / "legacy_curve.csv")
    assert 1000 in written["COMID"].values and (written["Row"] == 7).any()
    pd.testing.assert_frame_equal(reach_average_curve_file_dataframe(curves, vdt, qmax), written)


# --- The cross-section file ------------------------------------------------------------------------------------


def random_records(rng, count: int = 40) -> list[CrossSectionRecord]:
    records = []
    for k in range(count):
        side1, side2 = int(rng.integers(1, 90)), int(rng.integers(1, 90))
        profile1 = 150.0 + np.cumsum(rng.uniform(-0.2, 1.5, side1))
        profile2 = np.concatenate([[profile1[0]], 150.0 + np.cumsum(rng.uniform(-0.2, 1.5, side2 - 1))]) \
            if side2 > 1 else profile1[:1].copy()
        n1 = rng.choice([0.03, 0.035, 0.12], side1).astype(np.float32).astype(np.float64)
        n2 = rng.choice([0.03, 0.035, 0.12], side2).astype(np.float32).astype(np.float64)
        records.append(CrossSectionRecord(1000 + k, int(rng.integers(0, 900)), int(rng.integers(0, 900)), profile1,
                                          float(rng.uniform(20, 35)), n1, profile2, n2, int(rng.integers(0, 900)),
                                          int(rng.integers(0, 900)), int(rng.integers(0, 900)),
                                          int(rng.integers(0, 900))))
    return records


def legacy_record(record: CrossSectionRecord) -> dict:
    """The dict legacy _build_cross_section_export_record made, with the extra keys the file leaves out."""
    return {'COMID': record.comid, 'Row': record.row, 'Col': record.col, 'XS1_Profile': record.xs1_profile,
            'Ordinate_Dist': record.ordinate_dist, 'Manning_N_Raster1': record.manning_n_raster1,
            'XS2_Profile': record.xs2_profile, 'Manning_N_Raster2': record.manning_n_raster2, 'r1': record.r1,
            'c1': record.c1, 'r2': record.r2, 'c2': record.c2, 'Bank_Index1': 3, 'Bank_Index2': 4, 'Slope': 0.001,
            'Thalweg': float(record.xs1_profile[0]), 'Inflect_D2W_Dy2': None}


def test_the_cross_section_file_is_legacy_s(tmp_path: Path) -> None:
    records = random_records(np.random.default_rng(6))
    legacy = legacy_writer(tmp_path, np.empty((0, 8 + 5 * INCREMENTS)))
    legacy.add_cross_section_data([legacy_record(r) for r in records] + [None])
    legacy.save_cross_section_file()

    write_cross_sections(records + [None], tmp_path / "xs.txt")

    assert_same_file(tmp_path / "xs.txt", tmp_path / "legacy_xs.txt")


def test_an_empty_cross_section_file_is_legacy_s(tmp_path: Path) -> None:
    legacy = legacy_writer(tmp_path, np.empty((0, 8 + 5 * INCREMENTS)))
    legacy.add_cross_section_data([])
    legacy.save_cross_section_file()

    write_cross_sections([], tmp_path / "xs.txt")

    assert_same_file(tmp_path / "xs.txt", tmp_path / "legacy_xs.txt")


ARRAY_COLUMNS = ('XS1_Profile', 'Manning_N_Raster1', 'XS2_Profile', 'Manning_N_Raster2')


def test_the_cross_section_file_can_be_parquet(tmp_path: Path) -> None:
    """In Parquet the profiles and n are lists of float64, the records' arrays exactly, which the text file prints."""
    records = random_records(np.random.default_rng(6))
    text = write_cross_sections(records + [None], tmp_path / "xs.txt")

    returned = write_cross_sections(records + [None], tmp_path / "xs.parquet")

    df = pd.read_parquet(tmp_path / "xs.parquet")
    assert list(df.columns) == list(returned.columns) == XS_EXPORT_COLUMNS
    assert len(df) == len(returned) == len(records)
    for column, field in zip(XS_EXPORT_COLUMNS, CrossSectionRecord._fields):
        expected = [getattr(record, field) for record in records]
        if column in ARRAY_COLUMNS:
            for value, array in zip(df[column], expected):
                assert value.dtype == np.float64 and np.array_equal(value, array)
            assert [format_array(value) for value in df[column]] == list(text[column])
        else:
            assert df[column].dtype == (np.float64 if column == 'Ordinate_Dist' else np.int64)
            assert df[column].tolist() == expected


def test_an_empty_cross_section_file_can_be_parquet(tmp_path: Path) -> None:
    write_cross_sections([], tmp_path / "xs.parquet")

    schema = pq.read_schema(tmp_path / "xs.parquet")
    assert schema.names == XS_EXPORT_COLUMNS
    for column in ARRAY_COLUMNS:
        assert pa.types.is_list(schema.field(column).type) and schema.field(column).type.value_type == pa.float64()
    assert pd.read_parquet(tmp_path / "xs.parquet").empty


def test_parquet_cross_sections_split_the_elevations_bytes_and_keep_a_dictionary_of_n(tmp_path: Path) -> None:
    """The layout that makes the file small. pyarrow silently ignores a list column named without its
    '.list.element' path, which would undo it."""
    write_cross_sections(random_records(np.random.default_rng(9)), tmp_path / "xs.parquet")

    row_group = pq.ParquetFile(tmp_path / "xs.parquet").metadata.row_group(0)
    columns = {row_group.column(i).path_in_schema: row_group.column(i) for i in range(row_group.num_columns)}
    for side in (1, 2):
        assert 'BYTE_STREAM_SPLIT' in columns[f'XS{side}_Profile.list.element'].encodings
        assert 'RLE_DICTIONARY' in columns[f'Manning_N_Raster{side}.list.element'].encodings
    assert {column.compression for column in columns.values()} == {'ZSTD'}


def _array2string(values):
    return np.array2string(values, precision=6, max_line_width=np.inf, threshold=np.inf, floatmode='fixed')


@pytest.mark.parametrize("values", [
    np.array([170.472009, 171.74819, 99999.9]), np.array([-3.5, 12.25, 0.0, -0.0]), np.array([]),
    np.array([0.035, 0.12], dtype=np.float32), np.array([np.nan, 1.5, np.inf, -np.inf]), np.array([np.nan, np.nan]),
    np.array([0.00001, 250.0]), np.array([1.0, 1e9]), np.array([0.5, 600.0]), np.array([1e7], dtype=np.float32),
    np.array([3, 4]), np.array([[1.5, 2.5]])])
def test_arrays_print_as_numpy_prints_them(values: np.ndarray) -> None:
    assert format_array(values) == _array2string(values)


def test_arrays_print_as_numpy_prints_them_at_random() -> None:
    rng = np.random.default_rng(7)
    for _ in range(5000):
        size = int(rng.integers(1, 60))
        values = rng.uniform(-50, 3000, size) if rng.random() < 0.7 else 10 ** rng.uniform(-6, 9, size)
        if rng.random() < 0.3:
            values = np.round(values, int(rng.integers(0, 7)))
        if rng.random() < 0.2:
            values = values.astype(np.float32)
        assert format_array(values) == _array2string(values)


def test_the_cross_section_file_is_written_faster_than_legacy(tmp_path: Path) -> None:
    records = random_records(np.random.default_rng(8), count=400)
    legacy = legacy_writer(tmp_path, np.empty((0, 8 + 5 * INCREMENTS)))
    legacy.add_cross_section_data([legacy_record(r) for r in records])

    def seconds(function):
        start = time.perf_counter()
        function()
        return time.perf_counter() - start

    new = min(seconds(lambda: write_cross_sections(records, tmp_path / "xs.txt")) for _ in range(3))
    old = min(seconds(legacy.save_cross_section_file) for _ in range(3))
    assert new < old / 3


# --- Representative cross sections ------------------------------------------------------------------------------


def flat_bed_sample(rng, comid: int) -> tuple[RepresentativeSample, dict]:
    """A flat-bed channel between near-vertical walls 30 m high, where legacy's and the new hydraulics agree, as a
    sample and as legacy's export record."""
    spacing = float(rng.uniform(5, 15))
    half = int(rng.integers(3, 10))
    bed = 100.0 + float(np.round(rng.uniform(-2, 2), 3))
    elevations = np.full(2 * half + 41, bed + 30.0)
    center = elevations.size // 2
    elevations[center - half:center + half + 1] = bed
    n = np.full(elevations.size, float(np.float32(0.035)))
    xs = XSection(elevations, n, spacing)
    slope = float(rng.uniform(0.0005, 0.005))
    record = {'COMID': comid, 'XS1_Profile': elevations[center:].copy(), 'XS2_Profile': elevations[center::-1].copy(),
              'Manning_N_Raster1': n[center:].copy(), 'Manning_N_Raster2': n[center::-1].copy(),
              'Ordinate_Dist': spacing, 'Slope': slope, 'Thalweg': bed, 'Bank_Index1': -1, 'Bank_Index2': -1}
    return RepresentativeSample(comid, xs, bed, slope), record


def test_representative_cross_sections_match_legacy_s_where_the_hydraulics_agree() -> None:
    """Legacy rounded its hydraulics to 3 decimals, so the means agree to about that."""
    rng = np.random.default_rng(9)
    samples, records = zip(*(flat_bed_sample(rng, 5000 + k // 6) for k in range(18)))
    roughness = DepthRoughness(4.0, 1.5, 0.8)

    new = representative_cross_section_dataframe(samples, roughness=roughness, slope_factor=1.2)
    legacy = build_representative_cross_section_dataframe(list(records), 4.0, 1.5, 0.8, 1.2)

    assert list(new.columns) == list(legacy.columns)
    assert new.shape == legacy.shape == (3 * 250, 22)
    for column in new.columns:
        if new[column].dtype.kind in "iu":
            assert (new[column] == legacy[column]).all(), column
        else:
            assert np.allclose(new[column], legacy[column], rtol=2e-3, atol=2e-3), column


def test_representative_cross_sections_use_the_channel_s_velocity_where_there_are_banks() -> None:
    rng = np.random.default_rng(10)
    sample, _ = flat_bed_sample(rng, 7)
    sample.xs.left_bank_distance = sample.xs.right_bank_distance = sample.xs.ordinate_distance

    banked = representative_cross_section_dataframe([sample])
    sample.xs.left_bank_distance = sample.xs.right_bank_distance = -1.0
    unbanked = representative_cross_section_dataframe([sample])

    assert (banked["Mean_Discharge"] >= 0).all()
    assert not np.allclose(banked["Mean_Velocity"], unbanked["Mean_Velocity"])
    assert np.allclose(unbanked["Mean_Velocity"], unbanked["Mean_Discharge"] / unbanked["Mean_Cross_Sectional_Area"])


def test_representative_cross_sections_use_a_carved_channel_s_own_shape() -> None:
    """A 4 m channel carved into 10 m cells: at each stage the area is its profile's, not the ordinates'."""
    elevations = 100.0 + np.array([6.0, 4.0, 2.0, 0.0, 2.0, 4.0, 6.0])
    xs = XSection(elevations.copy(), np.full(7, 0.035), 10.0)
    carve_channel(xs, find_banks(xs, target_width=4.0), 1.0, trapezoid_height=0.2, bank_elevation=101.0)
    sample = RepresentativeSample(8, xs, float(xs.elevations[3]), 0.001)

    df = representative_cross_section_dataframe([sample])

    for _, stage in df[df["Depth_Stage_Meters"] <= 1.0 + 1e-9].iterrows():
        assert stage["Mean_Cross_Sectional_Area"] == pytest.approx(
            hydraulic_geometry(xs, depth=stage["Depth_Stage_Meters"]).area, rel=1e-9)
        assert stage["Mean_Top_Width"] <= 4.0 + 1e-9  # within the channel, narrower than a cell


def test_representative_cross_sections_are_written_rounded_to_6_decimals(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    samples = [flat_bed_sample(rng, 3)[0] for _ in range(3)]
    df = representative_cross_section_dataframe(samples)

    write_representative_cross_sections(df, tmp_path / "representative.csv")
    back = pd.read_csv(tmp_path / "representative.csv")

    assert np.allclose(back["Mean_Discharge"], df["Mean_Discharge"].round(6))
    write_representative_cross_sections(None, tmp_path / "empty.csv")
    assert pd.read_csv(tmp_path / "empty.csv").empty


def test_representative_cross_sections_can_be_parquet(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    samples = [flat_bed_sample(rng, 3)[0] for _ in range(3)]
    df = representative_cross_section_dataframe(samples)

    write_representative_cross_sections(df, tmp_path / "representative.csv")
    written = write_representative_cross_sections(df, tmp_path / "representative.parquet")

    back = pd.read_parquet(tmp_path / "representative.parquet")
    pd.testing.assert_frame_equal(back, written)
    pd.testing.assert_frame_equal(back, pd.read_csv(tmp_path / "representative.csv"), check_exact=False, rtol=1e-12)
    write_representative_cross_sections(None, tmp_path / "empty.parquet")
    empty = pd.read_parquet(tmp_path / "empty.parquet")
    assert empty.empty and list(empty.columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
    assert empty["COMID"].dtype == np.int64 and empty["Mean_Discharge"].dtype == np.float64


def test_a_reach_stops_at_the_first_stage_whose_hydraulics_are_not_finite() -> None:
    """Water reaching a NaN in one cross section ends its reach: the stages before stand, and say where it ended."""
    rng = np.random.default_rng(12)
    samples = [flat_bed_sample(rng, 4)[0] for _ in range(3)] + [flat_bed_sample(rng, 5)[0]]
    elevations, bed = samples[1].xs.elevations, samples[1].thalweg
    center = elevations.size // 2
    wall = center + int(np.argmax(elevations[center:] > bed))
    elevations[wall:] = bed + 1.25  # a right overbank 1.25 m up, to a NaN at its end
    elevations[-1] = np.nan

    df = representative_cross_section_dataframe(samples)

    ended = df[df["COMID"] == 4]
    assert ended["Depth_Stage_Index"].tolist() == list(range(1, 13))  # at 1.3 m the water reaches the NaN
    assert (ended["Reach_Inflect_Terrace_Depth"] == ended["Depth_Stage_Meters"].iloc[-1]).all()
    assert (df["COMID"] == 5).sum() == 250


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 1000, 4099])
def test_representative_means_sum_as_numpy_does(count: int) -> None:
    """The compiled stages' sums are numpy's to the bit, so their means and deviations are numpy's too."""
    from arc.outputs.representative import _sum

    rng = np.random.default_rng(count)
    values = rng.standard_normal(count + 3) * 10.0 ** rng.integers(-3, 6, count + 3)

    assert _sum(values, 3, count) == np.sum(values[3:])
