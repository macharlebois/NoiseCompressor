"""This script tests the functions in the pointcloud modules.

This script requires that the following packages be installed
within the Python environment you are running this script in:
pandas (1.5.3), plyfile (0.7.4), pytest (8.3.5)

"""
import io
import pandas as pd
from plyfile import PlyData
import pytest
import urllib.request

from step1_OPTIMIZER import main_optimizer


@pytest.fixture
def sample_data():
    threshold = "Skeleton index"
    param_limits = pd.DataFrame({"Parameter": ["m1", "m2", "b", "SI_threshold"],
                                 "Min": [0, -1, 0, 0],
                                 "Max": [0, 1, 0, 1],
                                 "Step": [0.01, 0.01, 0.001, 0.001]
                                 })

    stem_info = pd.DataFrame({"stem_id": [10001], "true_dbh": [0.01]})

    ref_file = "https://github.com/macharlebois/Sample_Data/raw/refs/heads/main/noise_compressor/pytest/ref4testing.ply"
    with urllib.request.urlopen(ref_file) as response:
        ply_data = PlyData.read(io.BytesIO(response.read()))
    vertex_data = ply_data["vertex"].data
    ref_data = pd.DataFrame(vertex_data)
    ref_data["stem_id"] = ref_data["stem_id"].astype(int)

    skeleton_file = "https://github.com/macharlebois/Sample_Data/raw/refs/heads/main/noise_compressor/pytest/skl4testing.csv"
    skeleton_data = pd.read_csv(skeleton_file)

    return stem_info, ref_data, skeleton_data, param_limits, threshold


def test_main_optimizer(sample_data):
    relation_tab, param_tab = main_optimizer(sample_data[0], sample_data[1], sample_data[2], sample_data[3], sample_data[4])
    best = ["hof1"]
    result = param_tab[param_tab["individual"].isin(best)]

    assert not relation_tab.empty
    assert {"stem_id", "estimated_dbh", "rel_dbh_diff", "true_dbh", "dbh_diff", "individual"}.issubset(relation_tab.columns)
    assert not param_tab.empty
    assert round(result["m1"].values[0], 3) == 0.000
    assert round(result["m2"].values[0], 3) == -0.44
    assert round(result["b"].values[0], 3) == 0.000
    assert round(result["SI_threshold"].values[0], 3) == 0.005
