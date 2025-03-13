"""This module calculates three-dimensional coordinates.

It requires that the following packages be installed within
the Python environment you are running this script in: numpy (1.24.2)

This file can be imported as a module and contains the following functions:
* convert2cartesian - converts spherical coordinates (r, polar, azimuth) to cartesian coordinates (XYZ)
* convert2spherical - converts cartesian coordinates (XYZ) to spherical coordinates (r, polar, azimuth)
"""

import numpy as np


def convert2cartesian(
    data, dist, nm_x="X", nm_y="Y", nm_z="Z", nm_polar="polar", nm_azimuth="azimuth"
):
    """
    This function converts spherical coordinates (r, polar, azimuth) to cartesian coordinates (XYZ)
    and adds them to the data.

    Parameters:
        data (DataFrame): The data to be transformed.
        dist (float): The distance from the origin.
        nm_x (str): X coordinate.
        nm_y (str): Y coordinate.
        nm_z (str): Z coordinate.
        nm_polar (str): Polar coordinate.
        nm_azimuth (str): Azimuth coordinate.
    Returns:
        DataFrame: The data with the cartesian coordinates.
    """
    data.loc[:, nm_x] = (
        dist * np.sin(np.deg2rad(data[nm_polar])) * np.cos(np.deg2rad(data[nm_azimuth]))
    )
    data.loc[:, nm_y] = (
        dist * np.sin(np.deg2rad(data[nm_polar])) * np.sin(np.deg2rad(data[nm_azimuth]))
    )
    data.loc[:, nm_z] = dist * np.cos(np.deg2rad(data[nm_polar]))

    return data


def convert2spherical(data, dimension1=None, dimension2=None):
    """
    This function converts cartesian coordinates (XYZ) to spherical coordinates (r, polar, azimuth)
    and adds them to the data.

    Parameters:
        data (DataFrame): The data to be transformed.
        dimension1 (list): The first set of cartesian coordinates.
        dimension2 (list): The second set of cartesian coordinates.
    Returns:
        DataFrame: The data with the spherical coordinates.
    """
    if dimension1 is None:
        dimension1 = ["X", "Y", "Z"]
    if dimension2 is None:
        dimension2 = ["X", "Y", "Z"]
    data = data.copy()

    dx = data[dimension2[0]].values - data[dimension1[0]].values
    dy = data[dimension2[1]].values - data[dimension1[1]].values
    dz = data[dimension2[2]].values - data[dimension1[2]].values

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r_safe = np.where(r == 0, 1e-10, r)  # Avoid division by zero

    data["r"] = r
    data["polar"] = np.arccos(dz / r_safe) * (180 / np.pi)
    data["azimuth"] = (np.arctan2(dy, dx) * (180 / np.pi)) % 360

    return data
