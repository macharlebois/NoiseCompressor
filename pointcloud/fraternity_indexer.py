"""This module adds fraternity index to a point cloud.

The fraternity index improves the denoising of point clouds by
distinguishing small diameter stems (whose noise distribution
is more complex).

This script accepts pandas dataframe and uses each point's
frame number, polar angle, cartesian coordinates (XYZ) and
distance (in meters) to lidar.

This script requires that the following packages be installed
within the Python environment you are running this script in:
numpy (1.24.3), polars (0.19.4), spicy (1.15.1)

This file can also be imported as a module and contains the
following functions:
* arc_separator - groups points by frame number and polar angle
* fraternity_index - calculates each point's fraternity index
"""

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree


def arc_separator(data):
    """
    This function groups points from a point cloud (polars.DataFrame)
    by frame number and polar angle to calculate their fraternity index
    (added to the dataframe as a new column).

    Returns
    -------
    polars.DataFrame
        The dataframe with the fraternity index added.
    """
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    data2 = data.group_by("frame_num", "polar").map_groups(fraternity_index)
    data2 = data2.to_pandas()
    return data2


def fraternity_index(group):
    """
    This function is called by the arc_separator function and returns
    the dataframe (group) with the fraternity index added as a new column.

    The fraternity index estimates the expected distance between a
    point and its closest neighbor from the group (polars dataframe
    of a single frame number and polar angle) based on its distance
    to the lidar when recorded.

    Parameters
    ----------
    group : polars.DataFrame
        A group of points from the point cloud that shares the same
        frame number and polar angle.
    """
    nb_points = group.height
    if nb_points == 1:
        # If there is only one point, we cannot calculate the fraternity index
        result_group = group.with_columns(pl.lit(5).alias("1st_neighbor_dist"))
    elif nb_points == 2:
        points = group.select(["X", "Y", "Z"]).to_numpy()
        distance = np.linalg.norm(points[0, :] - points[1, :])
        result_group = group.with_columns(pl.lit(distance).alias("1st_neighbor_dist"))
    else:
        points = group.select(["X", "Y", "Z"]).to_numpy()
        tree = cKDTree(points)
        dists, inds = tree.query(points, k=2)

        result_group = group.with_columns([
            pl.Series("1st_neighbor_dist", dists[:, 1] / group["dist_lidar"])
        ])
    result_group = result_group.with_columns(pl.col("1st_neighbor_dist").cast(pl.Float64))

    return result_group

