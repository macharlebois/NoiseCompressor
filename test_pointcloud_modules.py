"""This script tests the functions in the pointcloud modules.

This script requires that the following packages be installed
within the Python environment you are running this script in:
numpy (1.24.2), pandas (1.5.3), plyfile (0.7.4), pytest (8.3.5)

"""


import numpy as np
import pandas as pd
import polars as pl
import plyfile
import pytest
from unittest.mock import patch

from pointcloud.fraternity_indexer import arc_separator, fraternity_index
from pointcloud.locator import convert2spherical, convert2cartesian
from pointcloud import reader, writer
from pointcloud.skeletonizer import generate_skeleton
from pointcloud.utility import error_message, search_file, select_threshold, set_parameters
from pointcloud.validator import validate_data


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "X": [0.25110114, 0.21761797, 0.25442076, 0.24544311, 0.27970350, 0.24725962, 0.18574882, 0.23995221, 0.25598133, 0.26434636],
        "Y": [9.08468342, 8.98589420, 9.02813435, 9.00447178, 9.05148792, 9.06576633, 9.06445789, 9.02040195, 8.98791981, 9.00155354],
        "Z": [1.13968015, 1.10745335, 1.15730727, 1.15117717, 1.10726714, 1.11770403, 1.16977441, 1.15703380, 1.12377214, 1.14748609],
        "azimuth": [-0.644666, -0.837074, -0.112268, 0.580682, -0.856804, -0.919381, 0.559727, -0.121555, -1.090198, 0.160279],
        "polar": [-11.000, -11.000, -5.000, 3.000, -9.000, -11.000, 3.000, -5.000, -9.000, -5.000],
        "frame_num": [1830, 1852, 1800, 1737, 1846, 4189, 1737, 1803, 1915, 4090],
        "dist_lidar": [3.172, 3.790, 2.858, 4.108, 3.580, 6.192, 4.118, 2.870, 6.290, 5.582],
        "relative_z": [1.122437, 1.060402, 1.160290, 1.104125, 1.110250, 1.100461, 1.152532, 1.109982, 1.169523, 1.193236],
    })


@pytest.fixture
def sample_data2():
    return pd.DataFrame({
        "X": [1.0, 2.0, 3.0],
        "Y": [4.0, 5.0, 6.0],
        "Z": [7.0, 8.0, 9.0],
        "relative_z": [0.5, 1.5, 2.5]
    })


# ---------------------------TESTS FRATERNITY INDEXER---------------------------

def test_fraternity_index(sample_data):
    if isinstance(sample_data, pd.DataFrame):
        sample_data = pl.from_pandas(sample_data)
    result = fraternity_index(sample_data.clone())
    if isinstance(result, pl.DataFrame):
        result = result.to_pandas()

    assert "1st_neighbor_dist" in result.columns
    assert not result["1st_neighbor_dist"].isna().any()


def test_arc_separator(sample_data):
    result = arc_separator(sample_data.copy())

    assert "1st_neighbor_dist" in result.columns
    assert len(result) == len(sample_data)


# ---------------------------TESTS LOCATOR---------------------------

def test_convert2cartesian(sample_data):
    dist = 10
    result = convert2cartesian(sample_data.copy(), dist)

    assert "X" in result.columns
    assert "Y" in result.columns
    assert "Z" in result.columns

    expected_x = dist * np.sin(np.deg2rad(sample_data["polar"])) * np.cos(np.deg2rad(sample_data["azimuth"]))
    expected_y = dist * np.sin(np.deg2rad(sample_data["polar"])) * np.sin(np.deg2rad(sample_data["azimuth"]))
    expected_z = dist * np.cos(np.deg2rad(sample_data["polar"]))

    np.testing.assert_almost_equal(result["X"].values, expected_x.values, decimal=3)
    np.testing.assert_almost_equal(result["Y"].values, expected_y.values, decimal=3)
    np.testing.assert_almost_equal(result["Z"].values, expected_z.values, decimal=3)


def test_convert2spherical(sample_data):
    sample_data["X2"] = sample_data["X"] + 1
    sample_data["Y2"] = sample_data["Y"] + 1
    sample_data["Z2"] = sample_data["Z"] + 1

    result = convert2spherical(sample_data, dimension1=["X", "Y", "Z"], dimension2=["X2", "Y2", "Z2"])

    expected_r = 1.73205
    expected_polar = 54.73561
    expected_azimuth = 45.00000

    np.testing.assert_almost_equal(result["r"].values, expected_r, decimal=3)
    np.testing.assert_almost_equal(result["polar"].values, expected_polar, decimal=3)
    np.testing.assert_almost_equal(result["azimuth"].values, expected_azimuth, decimal=3)


# ---------------------------TESTS READER---------------------------

def test_read_csv(tmp_path):
    file = tmp_path / "test.csv"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df.to_csv(file, index=False)
    result = reader.read_csv(file)
    pd.testing.assert_frame_equal(df, result)


def test_read_ply(tmp_path):
    file = str(tmp_path / "test.ply")
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    writer.write_ply(file, df)
    result = reader.read_ply(file)
    pd.testing.assert_frame_equal(df, result)


def test_read_xlsx(tmp_path):
    file = tmp_path / "test.xlsx"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df.to_excel(file, index=False)
    result = reader.read_xlsx(file)
    pd.testing.assert_frame_equal(df, result)


# ---------------------------TESTS SKELETONIZER---------------------------

def test_generate_skeleton(sample_data):
    if isinstance(sample_data, pd.DataFrame):
        sample_data = pl.from_pandas(sample_data)
    param_sk = dict(
        voxel_size=0.01,
        search_radius=0.1,
        max_relocation_dist=0.21
    )
    result = generate_skeleton(sample_data, param_sk)

    assert isinstance(result, pl.DataFrame)
    assert not result.is_empty()
    assert {"oldX", "oldY", "oldZ", "old_relative_z", "X", "Y", "Z", "relative_z"}.issubset(result.columns)


# ---------------------------TESTS UTILITY---------------------------

def test_error_message():
    with patch("easygui.buttonbox") as mock_buttonbox:
        mock_buttonbox.return_value = "OK"
        error_message("Test error")
        mock_buttonbox.assert_called_once()


def test_search_file():
    with patch("easygui.buttonbox", return_value="Yes"):
        assert search_file("optimization") == "Yes"
    with patch("easygui.buttonbox", return_value="No"):
        assert search_file("skeleton") == "No"


def test_select_threshold():
    with patch("easygui.buttonbox", return_value="Skeleton index"):
        assert select_threshold("compressor") == "Skeleton index"
    with patch("easygui.buttonbox", return_value="Fraternity index"):
        assert select_threshold("optimizer") == "Fraternity index"


def test_set_parameters():
    with patch("easygui.multenterbox", return_value=["0.01", "0.1", "0.21"]):
        assert set_parameters("skeleton") == [0.01, 0.1, 0.21]
    with patch("easygui.multenterbox", return_value=["0.00", "0.98", "0.00", "0.171"]):
        assert set_parameters("compression", "Fraternity index") == [0.00, 0.98, 0.00, 0.171]


# ---------------------------TESTS VALIDATOR---------------------------

def test_validate_data_skeleton(sample_data2):
    df_validated, df_xyz = validate_data(sample_data2, usage="skeleton")

    assert set(df_validated.columns) >= {"X", "Y", "Z", "relative_z", "line_id"}
    assert set(df_xyz.columns) == {"line_id", "X", "Y", "Z", "relative_z"}

    assert df_validated["line_id"].is_unique


def test_validate_data_skeleton_missing_column():
    data = pd.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0]})

    with pytest.raises(ValueError, match="The 'relative_z' column could not be found in the point cloud."):
        validate_data(data, usage="skeleton")


def test_validate_data_skeleton_wrong_dtype():
    data = pd.DataFrame({
        "X": [1.0, 2.0, 3.0],
        "Y": ["a", "b", "c"],
        "Z": [7.0, 8.0, 9.0],
        "relative_z": [0.5, 1.5, 2.5]
    })

    with pytest.raises(ValueError, match="The 'Y' column is not of type float"):
        validate_data(data, usage="skeleton")


def test_validate_data_point_indexation(sample_data2):
    df_validated = validate_data(sample_data2, usage="point_indexation")
    assert set(df_validated.columns) >= {"X", "Y", "Z", "relative_z"}


def test_validate_data_point_indexation_fraternity(sample_data2):
    data = sample_data2.copy()
    data["dist_lidar"] = [1.1, 2.2, 3.3]
    data["polar"] = [10, 20, 30]
    data["frame_num"] = [100, 101, 102]

    df_validated = validate_data(data, usage="point_indexation", threshold="Fraternity index")
    assert set(df_validated.columns) >= {"X", "Y", "Z", "relative_z", "dist_lidar", "polar", "frame_num"}


def test_validate_data_compressor(sample_data2):
    df_validated = validate_data(sample_data2, usage="compressor")
    assert set(df_validated.columns) >= {"X", "Y", "Z", "relative_z"}


def test_validate_data_compressor_fraternity(sample_data2):
    data = sample_data2.copy()
    data["1st_neighbor_dist"] = [0.1, 0.2, 0.3]

    df_validated = validate_data(data, usage="compressor", threshold="Fraternity index")
    assert set(df_validated.columns) >= {"X", "Y", "Z", "relative_z", "1st_neighbor_dist"}


# ---------------------------TESTS WRITER---------------------------

def test_pd2array():
    df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0]})
    array = writer.pd2array(df)
    assert array.dtype.names == ("A", "B")
    np.testing.assert_array_equal(array["A"], df["A"].values)


def test_write_csv(tmp_path):
    file = tmp_path / "test_output.csv"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    writer.write_csv(file, df)
    result = pd.read_csv(file)
    pd.testing.assert_frame_equal(df, result)


def test_write_ply(tmp_path):
    file = str(tmp_path / "test_output.ply")
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    writer.write_ply(file, df)
    result = plyfile.PlyData.read(file)
    result2 = pd.DataFrame(result.elements[0].data)
    pd.testing.assert_frame_equal(df, result2)


def test_write_xlsx(tmp_path):
    file = tmp_path / "test_output.xlsx"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    writer.write_xlsx(file, df)
    result = pd.read_excel(file)
    pd.testing.assert_frame_equal(df, result)
