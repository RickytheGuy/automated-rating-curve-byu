from __future__ import annotations

import logging
from pathlib import Path

import pytest

from arc.config import Configs

BASE_INPUTS = {
    "DEM_File": "dem.tif",
    "Stream_File": "stream.tif",
    "LU_Raster_SameRes": "land.tif",
    "LU_Manning_n": "mannings.txt",
    "Flow_File": "flows.csv",
    "Flow_File_ID": "COMID",
    "Flow_File_QMax": "qmax",
    "StrmShp_File": "stream_network.gpkg",
    "reach_id": "COMID",
    "downstream_reach_id": "DSCOMID",
}
DEPTH_POWERLAW = {"drainage_area_field": "DA", "coefficient_depth": 0.5, "exponent_depth": 0.25}
WIDTH_POWERLAW = {"drainage_area_field": "DA", "coefficient_width": 3.0, "exponent_width": 0.4}
REPRESENTATIVE = {
    "Build_Representative_Cross_Section": True,
    "Representative_Cross_Section_File": "representative.csv",
}


def test_text_yaml_and_mapping_inputs_build_equal_configs(tmp_path: Path) -> None:
    """All three input styles parse to the same, correctly typed values."""
    text_file = tmp_path / "ARC_Input_File.txt"
    text_file.write_text(
        "# ARC Inputs\n"
        "DEM_File\tinputs/my dem.tif\n"
        "Stream_File\tinputs/stream.tif\n"
        "\n"
        "X_Section_Dist\t40\n"
        "Low_Spot_Range\t2\n"
        "k_decay\t4.5\n"
        "Bathy_Use_Banks\tTrue\n"
        "FindBanksBasedOnLandCover\tfalse\n"
        "Stream_Slope_Method\treach_average\n",
        encoding="utf-8",
    )
    yaml_file = tmp_path / "ARC_Input_File.yaml"
    yaml_file.write_text(
        "DEM_File: inputs/my dem.tif\n"
        "Stream_File: inputs/stream.tif\n"
        "X_Section_Dist: 40\n"
        "Low_Spot_Range: 2\n"
        "k_decay: 4.5\n"
        "Bathy_Use_Banks: true\n"
        "FindBanksBasedOnLandCover: false\n"
        "Stream_Slope_Method: reach_average\n",
        encoding="utf-8",
    )
    mapping = {
        "dem_file": "inputs/my dem.tif",
        "stream_file": "inputs/stream.tif",
        "x_section_dist": 40,
        "low_spot_range": 2,
        "k_decay": 4.5,
        "bathy_use_banks": True,
        "findbanksbasedonlandcover": False,
        "stream_slope_method": "reach_average",
    }

    configs = Configs.from_file(text_file)

    assert configs == Configs.from_file(yaml_file) == Configs.from_mapping(mapping)
    assert configs.dem_file == "inputs/my dem.tif"
    assert configs.x_section_dist == 40.0 and isinstance(configs.x_section_dist, float)
    assert configs.low_spot_range == 2 and isinstance(configs.low_spot_range, int)
    assert configs.bathy_use_banks is True
    assert configs.findbanksbasedonlandcover is False
    assert configs.flow_file is None
    assert configs.gen_dir_dist == 10


def test_unrecognized_key_is_rejected_with_a_suggestion() -> None:
    with pytest.raises(ValueError, match=r"'x_section_distance' \(did you mean 'x_section_dist'\?\)"):
        Configs.from_mapping({**BASE_INPUTS, "X_Section_Distance": 40})


def test_obsolete_roughness_key_names_its_replacement() -> None:
    with pytest.raises(ValueError, match="alpha_low has been replaced by shallow_factor and deep_factor"):
        Configs.from_mapping({**BASE_INPUTS, "alpha_low": 1.0})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"StrmShp_File": "", "Stream_Slope_Method": "end_points"}, "'end_points' stream slope method requires a strmshp_file", id="end-points-needs-stream-vector"),
        pytest.param({"Stream_Slope_Method": "steepest"}, "stream_slope_method must be one of", id="unknown-slope-method"),
        pytest.param({"Build_Representative_Cross_Section": True}, "requires a representative_cross_section_file", id="representative-needs-output-file"),
        pytest.param({"coefficient_depth": 0.5, "exponent_depth": 0.25}, "drainage_area_field is required", id="power-law-needs-drainage-area"),
        pytest.param({"drainage_area_field": "DA", "coefficient_depth": 0.5}, "requires coefficient_depth and exponent_depth together", id="unpaired-depth-power-law"),
        pytest.param({**WIDTH_POWERLAW, "StrmShp_File": ""}, "StrmShp_File is required", id="power-law-needs-stream-vector"),
        pytest.param({"Flow_File_BF": "baseflow", "AROutBATHY": "bathy.tif", "reach_id": ""}, "requires both reach_id and downstream_reach_id", id="bathymetry-needs-reach-ids"),
        pytest.param({"X_Section_Dist": "wide"}, "x_section_dist must be a number", id="non-numeric-float"),
        pytest.param({"Low_Spot_Range": "2.5"}, "low_spot_range must be an integer", id="non-integer-int"),
        pytest.param({"k_decay": 0}, "k_decay must be finite and positive", id="k-decay"),
        pytest.param({"shallow_factor": 0.5}, "shallow_factor must be finite and >= 1", id="shallow-factor"),
        pytest.param({"deep_factor": 1.5}, r"deep_factor must be finite and in \(0, 1\]", id="deep-factor"),
        pytest.param({"slope_adjustment_factor": "nan"}, "slope_adjustment_factor must be finite and positive", id="slope-adjustment-factor"),
    ],
)
def test_invalid_options_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Configs.from_mapping({**BASE_INPUTS, **overrides})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"Flow_File_BF": "baseflow", "AROutBATHY": "bathy.tif"}, (False, False, False, "bathy.tif"), id="baseflow"),
        pytest.param({"AROutBATHY": "bathy.tif"}, (False, False, False, None), id="no-source-disables-bathymetry"),
        pytest.param({**DEPTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, False, "bathy.tif"), id="depth-power-law"),
        pytest.param({**WIDTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, True, "bathy.tif"), id="width-power-law"),
        pytest.param({**REPRESENTATIVE, "Flow_File_BF": "baseflow", **DEPTH_POWERLAW, **WIDTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (True, False, True, "bathy.tif"), id="representative-baseflow-beats-power-law"),
        pytest.param({**REPRESENTATIVE, "Flow_File_ID": "", "Flow_File_BF": "baseflow", **DEPTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, False, "bathy.tif"), id="representative-partial-baseflow-falls-back"),
        pytest.param({**REPRESENTATIVE, "AROutBATHY": "bathy.tif"}, (False, False, False, None), id="representative-without-source"),
        pytest.param({"Flow_File_BF": "baseflow", "AROutBATHY": "arc.tif", "BATHY_Out_File": "other.tif"}, (False, False, False, "arc.tif"), id="aroutbathy-beats-bathy-out-file"),
    ],
)
def test_bathymetry_options_are_derived_like_read_main_input_file(overrides: dict, expected: tuple) -> None:
    """Expected values are what read_main_input_file produces for the same inputs."""
    configs = Configs.from_mapping({**BASE_INPUTS, **overrides})

    assert (
        configs.use_representative_baseflow_bathymetry,
        configs.use_bathymetry_powerlaw,
        configs.use_bathymetry_powerlaw_width,
        configs.bathy_out_file,
    ) == expected


def test_bathymetry_fallbacks_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="arc._log"):
        Configs.from_mapping({**BASE_INPUTS, **REPRESENTATIVE, "AROutBATHY": "bathy.tif"})

    assert "Missing: Flow_File_BF. Falling back to drainage-area power-law bathymetry." in caplog.text
    assert "disabling bathymetry estimation" in caplog.text


def test_reach_average_curve_file_needs_a_curve_file() -> None:
    inputs = {**BASE_INPUTS, "Reach_Average_Curve_File": "True"}

    assert Configs.from_mapping(inputs).reach_average_curve_file is False
    assert Configs.from_mapping({**inputs, "Print_Curve_File": "curve.csv"}).reach_average_curve_file is True
