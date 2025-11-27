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
* process_block - process a block of points for big data skeletonization
"""

import gc
import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from tqdm import tqdm
from joblib import Parallel, delayed


def generate_skeleton(cloud, p):
    """
    This function generates a skeleton from a point cloud using
    the compressor4smalldata or compressor4bigdata.

    Parameters
    ----------
    cloud : polars.DataFrame
        Point cloud data
    p : dict
        Dictionary containing the skeleton parameters

    Returns
    -------
    skeleton : polars.DataFrame
        Skeleton data
    """
    if cloud.shape[0] < 50000:
        skeleton = compressor4smalldata(
            cloud.clone(),
            voxel_size=p["voxel_size"],
            search_radius=p["search_radius"],
            max_relocation_dist=p["max_relocation_dist"],
        )
    else:
        skeleton = compressor4bigdata(
            cloud.clone(),
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
    data : polars.DataFrame
        Point cloud data
    voxel_size : float
        Voxel size (in cubic meters) for voxel creation
    search_radius : float
        Search radius (in meters) for densest surrounding regions
    max_relocation_dist : float
        Maximum relocation distance (in meters) for each point

    Returns
    -------
    skeleton : polars.DataFrame
        Skeleton data
    """
    # Creation of the voxels
    data = data.with_columns([
        ((pl.col("X") / voxel_size).round() * voxel_size).alias("binx"),
        ((pl.col("Y") / voxel_size).round() * voxel_size).alias("biny"),
        ((pl.col("Z") / voxel_size).round() * voxel_size).alias("binz"),
    ])
    # Identification of the voxels
    n = 10 ** 3  # Helps with rounding
    data = data.with_columns([
        (
                (pl.col("binx") * n).round().cast(pl.Int32).cast(pl.Utf8) + "_"
                + (pl.col("biny") * n).round().cast(pl.Int32).cast(pl.Utf8) + "_"
                + (pl.col("binz") * n).round().cast(pl.Int32).cast(pl.Utf8)
        ).alias("VoxelID")
    ])

    #  Calculation of the central and average position of points within each voxel (centroid)
    voxdata = data.group_by("VoxelID").agg([
        pl.col("binx").max(),
        pl.col("biny").max(),
        pl.col("binz").max(),
        pl.col("X").mean().alias("Xmean"),
        pl.col("Y").mean().alias("Ymean"),
        pl.col("Z").mean().alias("Zmean"),
        pl.len().alias("Count")
    ])

    # Calculation of the voxel relocation
    voxdata_xyz = voxdata.select(["binx", "biny", "binz"]).to_numpy()

    tree = cKDTree(voxdata_xyz)
    indices = tree.query_ball_point(
        voxdata_xyz, r=search_radius, workers=-1, return_sorted=False, p=2
    )
    voxdata = voxdata.with_columns(indices=indices)
    voxdata = voxdata.with_columns(
        pl.col("indices")
        .map_elements(lambda x: len(x), return_dtype=pl.Int64)
        .alias("neighbors_count"),
    )
    count_values = voxdata["Count"].to_numpy()

    # Calculation of the weighted mean for each voxel
    weighted_mean = np.array([
        (voxdata_xyz[idx] * count_values[idx, None]).sum(axis=0) / count_values[idx].sum()
        for idx in indices
    ])
    weighted_mean_df = pl.DataFrame(weighted_mean, schema=["X_weighted_mean", "Y_weighted_mean", "Z_weighted_mean"])

    voxdata = voxdata.with_columns([
        weighted_mean_df["X_weighted_mean"],
        weighted_mean_df["Y_weighted_mean"],
        weighted_mean_df["Z_weighted_mean"]
    ])

    # Relocation calculations
    voxdata = voxdata.with_columns([
        (pl.col("X_weighted_mean") - pl.col("binx")).alias("diffx"),
        (pl.col("Y_weighted_mean") - pl.col("biny")).alias("diffy"),
        (pl.col("Z_weighted_mean") - pl.col("binz")).alias("diffz"),
    ])
    voxdata = voxdata.with_columns(
        (pl.col("diffx") ** 2 + pl.col("diffy") ** 2 + pl.col("diffz") ** 2).sqrt().alias("relocation_dist")
    )

    # Relocation distance limitation
    voxdata_prob = voxdata.filter(pl.col("relocation_dist") > max_relocation_dist).with_columns([
        np.degrees(np.arccos(pl.col("diffz") / pl.col("relocation_dist"))).alias("polar"),
        (np.degrees(np.arctan2(pl.col("diffy"), pl.col("diffx"))) % 360).alias("azimuth")
    ]).with_columns([
        (max_relocation_dist * np.sin(np.radians(pl.col("polar"))) * np.cos(np.radians(pl.col("azimuth")))).alias(
            "diffx"),
        (max_relocation_dist * np.sin(np.radians(pl.col("polar"))) * np.sin(np.radians(pl.col("azimuth")))).alias(
            "diffy"),
        (max_relocation_dist * np.cos(np.radians(pl.col("polar")))).alias("diffz")
    ])

    # Merging recalculated relocation distance
    voxdata = voxdata.join(
        voxdata_prob.select(["VoxelID", "diffx", "diffy", "diffz"]),
        on="VoxelID",
        how="left",
        suffix="_A"
    )

    for axis in ["x", "y", "z"]:
        voxdata = voxdata.with_columns([
            pl.when(pl.col(f"diff{axis}_A").is_not_null())
            .then(pl.col(f"diff{axis}_A"))
            .otherwise(pl.col(f"diff{axis}"))
            .alias(f"diff{axis}")
        ])

    # Applying relocation to original points
    finaltab = data.join(
        voxdata.select(["VoxelID", "diffx", "diffy", "diffz", "Count"]),
        on="VoxelID",
        how="left"
    ).with_columns([
        (pl.col("X") + pl.col("diffx")).alias("newx"),
        (pl.col("Y") + pl.col("diffy")).alias("newy"),
        (pl.col("Z") + pl.col("diffz")).alias("newz"),
        (pl.col("relative_z") + pl.col("diffz")).alias("new_relative_z")
    ])

    finaltab = finaltab.rename({
        "X": "oldX", "Y": "oldY", "Z": "oldZ", "relative_z": "old_relative_z",
        "newx": "X", "newy": "Y", "newz": "Z", "new_relative_z": "relative_z"
    })
    return finaltab.drop(["binx", "biny", "binz", "VoxelID", "diffx", "diffy", "diffz"])


def compressor4bigdata(data, voxel_size, search_radius, max_relocation_dist):
    """
    This function generates a skeleton for large dataset (50 000+ points).

    Parameters
    ----------
    data : polars.DataFrame
        Point cloud data
    voxel_size : float
        Voxel size (in cubic meters) for voxel creation
    search_radius : float
        Search radius (in meters) for densest surrounding regions
    max_relocation_dist : float
        Maximum relocation distance (in meters) for each point

    Returns
    -------
    skeleton : polars.DataFrame
        Skeleton data
    """
    # Creation of the voxels
    data = data.with_columns([
        ((pl.col("X") / voxel_size).round() * voxel_size).alias("binx"),
        ((pl.col("Y") / voxel_size).round() * voxel_size).alias("biny"),
        ((pl.col("Z") / voxel_size).round() * voxel_size).alias("binz"),
    ])
    # Identification of the voxels
    n = 10 ** 3  # Helps with rounding
    data = data.with_columns([
        (
                (pl.col("binx") * n).round().cast(pl.Int32).cast(pl.Utf8) + "_"
                + (pl.col("biny") * n).round().cast(pl.Int32).cast(pl.Utf8) + "_"
                + (pl.col("binz") * n).round().cast(pl.Int32).cast(pl.Utf8)
        ).alias("VoxelID")
    ])

    # Calculation of the central and average position of points within each voxel (centroid)
    voxdata = data.group_by("VoxelID").agg([
        pl.col("binx").max(),
        pl.col("biny").max(),
        pl.col("binz").max(),
        pl.col("X").mean().alias("Xmean"),
        pl.col("Y").mean().alias("Ymean"),
        pl.col("Z").mean().alias("Zmean"),
        pl.len().alias("Count")
    ])

    # Block preparation for large dataset processing
    voxdata_xyz = voxdata.select(["binx", "biny", "binz"]).to_numpy()
    count_values = voxdata["Count"].to_numpy()
    ####

    weighted_xyz = voxdata_xyz * count_values[:, None]

    ###
    tree = cKDTree(voxdata_xyz)
    block_size = 30000  # 30 000 voxels per block
    num_points = voxdata_xyz.shape[0]
    num_blocks = num_points // block_size
    blocks = np.array_split(voxdata_xyz, num_blocks + 1 if num_points % block_size != 0 else num_blocks)

    results = Parallel(n_jobs=8, prefer="threads", batch_size=2)(
        delayed(process_block)(blk, tree, search_radius, weighted_xyz, count_values)
        for blk in tqdm(blocks)
    )
    cumul_weightmean = results
    cumul_weightmean = np.vstack(cumul_weightmean)
    weighted_mean_df = pl.DataFrame(cumul_weightmean, schema=["X_weighted_mean", "Y_weighted_mean", "Z_weighted_mean"])


    voxdata = voxdata.with_columns([
        weighted_mean_df["X_weighted_mean"],
        weighted_mean_df["Y_weighted_mean"],
        weighted_mean_df["Z_weighted_mean"]
    ])

    # Relocation calculations
    voxdata = voxdata.with_columns([
        (pl.col("X_weighted_mean") - pl.col("binx")).alias("diffx"),
        (pl.col("Y_weighted_mean") - pl.col("biny")).alias("diffy"),
        (pl.col("Z_weighted_mean") - pl.col("binz")).alias("diffz"),
    ])
    voxdata = voxdata.with_columns(
        (pl.col("diffx") ** 2 + pl.col("diffy") ** 2 + pl.col("diffz") ** 2).sqrt().alias("relocation_dist")
    )

    # Relocation distance limitation
    voxdata_prob = voxdata.filter(pl.col("relocation_dist") > max_relocation_dist).with_columns([
        np.degrees(np.arccos(pl.col("diffz") / pl.col("relocation_dist")) * (180 / np.pi)).alias("polar"),
        (np.degrees(np.arctan2(pl.col("diffy"), pl.col("diffx")) * (180 / np.pi)) % 360).alias("azimuth")
    ]).with_columns([
        (max_relocation_dist * np.sin(np.radians(pl.col("polar"))) * np.cos(np.radians(pl.col("azimuth")))).alias(
            "diffx"),
        (max_relocation_dist * np.sin(np.radians(pl.col("polar"))) * np.sin(np.radians(pl.col("azimuth")))).alias(
            "diffy"),
        (max_relocation_dist * np.cos(np.radians(pl.col("polar")))).alias("diffz")
    ])

    del cumul_weightmean, weighted_xyz
    gc.collect()
    # Merging recalculated relocation distance
    voxdata = voxdata.join(
        voxdata_prob.select(["VoxelID", "diffx", "diffy", "diffz"]),
        on="VoxelID", how="left", suffix="_A"
    )

    for axis in ["x", "y", "z"]:
        voxdata = voxdata.with_columns([
            pl.when(pl.col(f"diff{axis}_A").is_not_null())
            .then(pl.col(f"diff{axis}_A"))
            .otherwise(pl.col(f"diff{axis}"))
            .alias(f"diff{axis}")
        ])

    # Applying relocation to original points
    finaltab = data.join(
        voxdata.select(["VoxelID", "diffx", "diffy", "diffz", "Count"]),
        on="VoxelID", how="left"
    ).with_columns([
        (pl.col("X") + pl.col("diffx")).alias("newx"),
        (pl.col("Y") + pl.col("diffy")).alias("newy"),
        (pl.col("Z") + pl.col("diffz")).alias("newz"),
        (pl.col("relative_z") + pl.col("diffz")).alias("new_relative_z")
    ])

    finaltab = finaltab.rename({
        "X": "oldX", "Y": "oldY", "Z": "oldZ", "relative_z": "old_relative_z",
        "newx": "X", "newy": "Y", "newz": "Z", "new_relative_z": "relative_z"
    })

    return finaltab.drop(["binx", "biny", "binz", "VoxelID", "diffx", "diffy", "diffz"])


def process_block(block, tree, search_radius, weighted_xyz, count_values):
    indices = tree.query_ball_point(block, r=search_radius, p=2)
    block_weighted_mean = np.array([
       (weighted_xyz[idx]).sum(axis=0) / count_values[idx].sum()
       if len(idx) > 0 else np.zeros(weighted_xyz.shape[1])
       for idx in indices
    ])

    return block_weighted_mean

