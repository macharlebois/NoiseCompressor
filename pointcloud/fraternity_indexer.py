"""This module adds fraternity index to a point cloud.

The fraternity index improves the denoising of point clouds by
distinguishing small diameter stems (whose noise distribution
is more complex).

This script accepts pandas dataframe and uses each point's
frame number, polar angle, cartesian coordinates (XYZ) and
distance (in meters) to lidar.

This script requires that the following packages be installed
within the Python environment you are running this script in:
spicy (1.15.1), tqdm (4.66.4)

This file can also be imported as a module and contains the
following functions:
* arc_separator - groups points by frame number and polar angle
* fraternity_index - calculates each point's fraternity index
"""

from scipy.spatial import cKDTree
from tqdm import tqdm


def arc_separator(data):
    """
    This function groups points from a point cloud (pandas.DataFrame)
    by frame number and polar angle to calculate their fraternity index
    (added to the dataframe as a new column).

    Returns
    -------
    pandas.DataFrame
        The dataframe with the fraternity index added.
    """
    tqdm.pandas()
    data2 = data.groupby(["frame_num", "polar"], group_keys=False).progress_apply(
        fraternity_index
    )
    print()

    return data2


def fraternity_index(group):
    """
    This function is called by the arc_separator function and returns
    the dataframe (group) with the fraternity index added as a new column.

    The fraternity index estimates the expected distance between a
    point and its closest neighbor from the group (pandas dataframe
    of a single frame number and polar angle) based on its distance
    to the lidar when recorded.

    Parameters
    ----------
    group : pandas.DataFrame
        A group of points from the point cloud that shares the same
        frame number and polar angle.
    """
    points = group[["X", "Y", "Z"]].values
    tree = cKDTree(points)
    dists, inds = tree.query(points, k=2)
    group["dist"] = dists[
        :, 1
    ]  # gets the distance to the first neighbor that is not the point itself
    group["ind"] = inds[:, 1]
    group["1st_neighbor_dist"] = group["dist"] / group["dist_lidar"]

    return group
