"""This module generates a skeleton from a point cloud.

The skeleton creates a relocation vector which determine the maximal
relocation distance (r) and direction for each point during compression.

This script requires that the following packages be installed within the
Python environment you are running this script in:
datetime, numpy (1.24.2), pandas (1.5.3), scipy (1.15.1), tqdm (4.66.4)

This script uses the following local modules: 'reader', 'writer', 'utility',
'validator' from the local 'pointcloud' package.

This file can be imported as a module and contains the following functions:
* generate_skeleton - generates a skeleton from a point cloud
* compressor4smalldata - for small dataset (< 50 000 points)
* compressor4bigdata - for larger dataset (50 000+ points)
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pointcloud.utility as utility
from scipy.spatial import cKDTree
from tqdm import tqdm


def generate_skeleton(cloud, p):
    """
    This function generates a skeleton from a point cloud using
    the compressor4smalldata or compressor4bigdata.

    Parameters
    ----------
    cloud : pandas.DataFrame
        Point cloud data
    p : dict
        Dictionary containing the skeleton parameters

    Returns
    -------
    skeleton : pandas.DataFrame
        Skeleton data
    """
    if cloud.shape[0] < 50000:
        skeleton = compressor4smalldata(
            cloud.copy(),
            voxel_size=p["voxel_size"],
            search_radius=p["search_radius"],
            max_relocation_dist=p["max_relocation_dist"],
        )
    else:
        skeleton = compressor4bigdata(
            cloud.copy(),
            voxel_size=p["voxel_size"],
            search_radius=p["search_radius"],
            max_relocation_dist=p["max_relocation_dist"],
        )
    return skeleton


def compressor4smalldata(data, voxel_size, search_radius, max_relocation_dist):
    """
    This function generates a skeleton for small dataset (< 50 000 points)

    Parameters
    ----------
    data : pandas.DataFrame
        Point cloud data
    voxel_size : float
        Voxel size (in cubic meters) for voxel creation
    search_radius : float
        Search radius (in meters) for densest surrounding regions
    max_relocation_dist : float
        Maximum relocation distance (in meters) for each point

    Returns
    -------
    skeleton : pandas.DataFrame
        Skeleton data
    """
    # Creation of the voxels
    data["binx"] = np.round(data["X"] / voxel_size) * voxel_size
    data["biny"] = np.round(data["Y"] / voxel_size) * voxel_size
    data["binz"] = np.round(data["Z"] / voxel_size) * voxel_size
    # Identification of the voxels
    n = 10**3
    data["VoxelID"] = (
        (data["binx"] * n).round().astype(int).astype(str)
        + "_"
        + (data["biny"] * n).round().astype(int).astype(str)
        + "_"
        + (data["binz"] * n).round().astype(int).astype(str)
    )

    # Calculation of the central and average position of points within each voxel (centroid)
    voxdata = data.groupby(["VoxelID"]).aggregate(
        {
            "binx": "max",
            "biny": "max",
            "binz": "max",
            "X": "mean",
            "Y": "mean",
            "Z": "mean",
            "VoxelID": "size",
        }
    )
    voxdata = voxdata.rename(
        columns={"VoxelID": "Count", "X": "Xmean", "Y": "Ymean", "Z": "Zmean"}
    )
    voxdata.reset_index(inplace=True)

    # Calculation of the voxel relocation
    voxdata_xyz = np.array(
        voxdata[["binx", "biny", "binz"]]
    )  # Here, XYZ-mean could also be used to define the voxel position
    tree = cKDTree(voxdata_xyz)
    indices = tree.query_ball_point(
        voxdata_xyz, r=search_radius, workers=-1, return_sorted=False, p=2
    )
    voxdata["indices"] = indices
    voxdata["neighbors_count"] = voxdata["indices"].apply(len)
    indices_series = pd.Series(voxdata["indices"])
    count_values = voxdata["Count"].values
    xyz_values = voxdata[
        ["binx", "biny", "binz"]
    ].values  # Here, XYZ-mean could also be used to define the voxel position
    weighted_mean = np.array(
        [
            (xyz_values[indices] * count_values[indices][:, None]).sum(axis=0)
            / count_values[indices].sum()
            for indices in indices_series
        ]
    )
    weighted_mean_df = pd.DataFrame(
        weighted_mean, columns=["X_weighted_mean", "Y_weighted_mean", "Z_weighted_mean"]
    )
    voxdata[["X_weighted_mean", "Y_weighted_mean", "Z_weighted_mean"]] = (
        weighted_mean_df
    )

    voxdata["diffx"], voxdata["diffy"], voxdata["diffz"] = (
        voxdata["X_weighted_mean"] - voxdata["binx"],
        voxdata["Y_weighted_mean"] - voxdata["biny"],
        voxdata["Z_weighted_mean"] - voxdata["binz"],
    )
    voxdata["relocation_dist"] = np.sqrt(
        voxdata["diffx"] ** 2 + voxdata["diffy"] ** 2 + voxdata["diffz"] ** 2
    )
    voxdata_prob = voxdata[voxdata["relocation_dist"] > max_relocation_dist].copy()
    voxdata_prob["polar"] = np.arccos(
        voxdata_prob["diffz"] / voxdata_prob["relocation_dist"]
    ) * (180 / np.pi)
    voxdata_prob["azimuth"] = (
        np.arctan2(voxdata_prob["diffy"], voxdata_prob["diffx"]) * (180 / np.pi)
    ) % 360
    voxdata_prob["diffx"] = (
        max_relocation_dist
        * np.sin(np.deg2rad(voxdata_prob["polar"]))
        * np.cos(np.deg2rad(voxdata_prob["azimuth"]))
    )
    voxdata_prob["diffy"] = (
        max_relocation_dist
        * np.sin(np.deg2rad(voxdata_prob["polar"]))
        * np.sin(np.deg2rad(voxdata_prob["azimuth"]))
    )
    voxdata_prob["diffz"] = max_relocation_dist * np.cos(
        np.deg2rad(voxdata_prob["polar"])
    )
    cols_to_merge = ["VoxelID", "diffx", "diffy", "diffz"]
    voxdata = voxdata.merge(
        voxdata_prob[cols_to_merge], how="left", on="VoxelID", suffixes=("", "_A")
    )
    cols_to_update = ["diffx", "diffy", "diffz"]
    for col in cols_to_update:
        mask = ~voxdata[
            col + "_A"
        ].isnull()  # Mask to select only non-NaN values in '_A' columns
        voxdata[col] = voxdata[col].mask(mask, voxdata[col + "_A"])

    cols_to_merge = ["VoxelID", "diffx", "diffy", "diffz", "Count"]
    finaltab = data.merge(voxdata[cols_to_merge], how="left", on="VoxelID")
    finaltab["newx"], finaltab["newy"], finaltab["newz"], finaltab["new_relative_z"] = (
        finaltab["X"] + finaltab["diffx"],
        finaltab["Y"] + finaltab["diffy"],
        finaltab["Z"] + finaltab["diffz"],
        finaltab["relative_z"] + finaltab["diffz"],
    )
    finaltab = finaltab.rename(
        columns={"X": "oldX", "Y": "oldY", "Z": "oldZ", "relative_z": "old_relative_z"}
    )
    finaltab = finaltab.rename(
        columns={"newx": "X", "newy": "Y", "newz": "Z", "new_relative_z": "relative_z"}
    )
    finaltab.drop(
        ["binx", "biny", "binz", "VoxelID", "diffx", "diffy", "diffz"],
        axis=1,
        inplace=True,
    )

    return finaltab


def compressor4bigdata(data, voxel_size, search_radius, max_relocation_dist):
    """
    This function generates a skeleton for large dataset (50 000+ points).

    Parameters
    ----------
    data : pandas.DataFrame
        Point cloud data
    voxel_size : float
        Voxel size (in cubic meters) for voxel creation
    search_radius : float
        Search radius (in meters) for densest surrounding regions
    max_relocation_dist : float
        Maximum relocation distance (in meters) for each point

    Returns
    -------
    skeleton : pandas.DataFrame
        Skeleton data
    """
    # Creation of the voxels
    data["binx"] = np.round(data["X"] / voxel_size) * voxel_size
    data["biny"] = np.round(data["Y"] / voxel_size) * voxel_size
    data["binz"] = np.round(data["Z"] / voxel_size) * voxel_size
    # Identification of the voxels
    n = 10**3  # Helps with rounding
    data["VoxelID"] = (
        (data["binx"] * n).round().astype(int).astype(str)
        + "_"
        + (data["biny"] * n).round().astype(int).astype(str)
        + "_"
        + (data["binz"] * n).round().astype(int).astype(str)
    )

    # Calculation of the central and average position of points within each voxel (centroid)
    voxdata = data.groupby(["VoxelID"]).aggregate(
        {
            "binx": "max",
            "biny": "max",
            "binz": "max",
            "X": "mean",
            "Y": "mean",
            "Z": "mean",
            "VoxelID": "size",
        }
    )
    voxdata = voxdata.rename(
        columns={"VoxelID": "Count", "X": "Xmean", "Y": "Ymean", "Z": "Zmean"}
    )
    voxdata.reset_index(inplace=True)

    # Calculation of the voxel relocation
    voxdata_xyz = np.array(
        voxdata[["binx", "biny", "binz"]]
    )  # Here, XYZ-mean could also be used to define the voxel position
    count_values = voxdata["Count"].values
    tree = cKDTree(voxdata_xyz)

    block_size = 3000  # 3000 voxels block processing
    num_blocks = voxdata_xyz.shape[0] // block_size
    leftover = voxdata_xyz.shape[0] % block_size
    blocks = np.split(voxdata_xyz[:-leftover], num_blocks)
    if leftover > 0:
        last_block = voxdata_xyz[-leftover:]
        blocks.append(last_block)
    cumul_weightmean = np.empty((0, 3))
    i = 0
    utility.timer = datetime.now()
    n = len(blocks)
    pbar = tqdm(desc="Generating skeleton", total=n)
    while i < n:
        for block in blocks:
            indices = tree.query_ball_point(block, r=search_radius, workers=7, p=2)
            weighted_mean = np.array(
                [
                    (voxdata_xyz[indice] * count_values[indice][:, None]).sum(axis=0)
                    / count_values[indice].sum()
                    for indice in indices
                ]
            )
            cumul_weightmean = np.concatenate((cumul_weightmean, weighted_mean))
            i = i + 1
            pbar.update(1)

    utility.calculate_time("queryball by block", reset_timer=True)

    weighted_mean_df = pd.DataFrame(
        cumul_weightmean,
        columns=["X_weighted_mean", "Y_weighted_mean", "Z_weighted_mean"],
    )
    voxdata[["X_weighted_mean", "Y_weighted_mean", "Z_weighted_mean"]] = (
        weighted_mean_df
    )

    voxdata["diffx"], voxdata["diffy"], voxdata["diffz"] = (
        voxdata["X_weighted_mean"] - voxdata["binx"],
        voxdata["Y_weighted_mean"] - voxdata["biny"],
        voxdata["Z_weighted_mean"] - voxdata["binz"],
    )
    voxdata["relocation_dist"] = np.sqrt(
        voxdata["diffx"] ** 2 + voxdata["diffy"] ** 2 + voxdata["diffz"] ** 2
    )
    voxdata_prob = voxdata[voxdata["relocation_dist"] > max_relocation_dist].copy()
    voxdata_prob["polar"] = np.arccos(
        voxdata_prob["diffz"] / voxdata_prob["relocation_dist"]
    ) * (180 / np.pi)
    voxdata_prob["azimuth"] = (
        np.arctan2(voxdata_prob["diffy"], voxdata_prob["diffx"]) * (180 / np.pi)
    ) % 360
    voxdata_prob["diffx"] = (
        max_relocation_dist
        * np.sin(np.deg2rad(voxdata_prob["polar"]))
        * np.cos(np.deg2rad(voxdata_prob["azimuth"]))
    )
    voxdata_prob["diffy"] = (
        max_relocation_dist
        * np.sin(np.deg2rad(voxdata_prob["polar"]))
        * np.sin(np.deg2rad(voxdata_prob["azimuth"]))
    )
    voxdata_prob["diffz"] = max_relocation_dist * np.cos(
        np.deg2rad(voxdata_prob["polar"])
    )
    cols_to_merge = ["VoxelID", "diffx", "diffy", "diffz"]
    voxdata = voxdata.merge(
        voxdata_prob[cols_to_merge], how="left", on="VoxelID", suffixes=("", "_A")
    )
    cols_to_update = ["diffx", "diffy", "diffz"]
    for col in cols_to_update:
        mask = ~voxdata[
            col + "_A"
        ].isnull()  # Mask to select only non-NaN values in '_A' columns
        voxdata[col] = voxdata[col].mask(mask, voxdata[col + "_A"])

    cols_to_merge = ["VoxelID", "diffx", "diffy", "diffz", "Count"]
    finaltab = data.merge(voxdata[cols_to_merge], how="left", on="VoxelID")
    finaltab["newx"], finaltab["newy"], finaltab["newz"], finaltab["new_relative_z"] = (
        finaltab["X"] + finaltab["diffx"],
        finaltab["Y"] + finaltab["diffy"],
        finaltab["Z"] + finaltab["diffz"],
        finaltab["relative_z"] + finaltab["diffz"],
    )
    finaltab = finaltab.rename(
        columns={"X": "oldX", "Y": "oldY", "Z": "oldZ", "relative_z": "old_relative_z"}
    )
    finaltab = finaltab.rename(
        columns={"newx": "X", "newy": "Y", "newz": "Z", "new_relative_z": "relative_z"}
    )
    finaltab.drop(
        ["binx", "biny", "binz", "VoxelID", "diffx", "diffy", "diffz"],
        axis=1,
        inplace=True,
    )

    return finaltab
