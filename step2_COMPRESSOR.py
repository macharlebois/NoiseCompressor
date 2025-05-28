"""This module processes noise in a point cloud by the mean of compression.

The compressor performs the final compression of the point cloud based on the
relocation vector of each point, the parameters supplied by the optimizer and
the chosen threshold.

This script requires that the following packages be installed within the Python
environment you are running this script in:
easygui (0.98.3), numpy (1.24.2), os, pandas (1.5.3)

This script uses the following local modules: 'reader', 'writer', 'utility', 'validator',
'fraternity_indexer', 'skeletonizer' from the local 'pointcloud' package.

This file can be imported as a module and contains the following function:
* compress_cloud - compresses a point cloud according to a set of optimized parameters
"""

import easygui
import numpy as np
import os
import pandas as pd
from pointcloud import (
    reader as reader,
    writer as writer,
    locator as locator,
    validator as valid,
    fraternity_indexer as fraternity_indexer,
    skeletonizer as skeletonizer,
)
from pointcloud.utility import (
    error_message,
    select_threshold,
    search_file,
    set_parameters,
)


def compress_cloud(cloud, skeleton, param, threshold):
    """This function compresses a point cloud according to a set of optimized parameters.
    Parameters:
        cloud (DataFrame): The point cloud to compress.
        skeleton (DataFrame): The skeleton associated with the point cloud.
        param (dict): The optimized parameters for the compression.
        threshold (str): The threshold used for compression.
    Returns:
        DataFrame: The compressed point cloud.
    """
    if skeleton is None:
        error_message = "The skeleton is mandatory for compression"
        raise KeyError(error_message)

    skeleton.rename(
        columns={
            "line_id": "line_id_skeleton",
            "X": "X_skeleton",
            "Y": "Y_skeleton",
            "Z": "Z_skeleton",
        },
        inplace=True,
    )
    cols_to_merge = ["line_id_skeleton", "X_skeleton", "Y_skeleton", "Z_skeleton"]

    # 1-SKELETON ASSOCIATION
    # Associating the position of each point with its position in the skeleton
    cloud2 = cloud.merge(
        skeleton[cols_to_merge], left_on=["line_id"], right_on=["line_id_skeleton"]
    )

    # 2-MAXIMAL RELOCATION (r) CALCULATION
    # Calculating spherical coordinates and distance (r) between the point
    # and its clone in the skeleton
    cloud3 = locator.convert2spherical(
        cloud2,
        dimension1=["X", "Y", "Z"],
        dimension2=["X_skeleton", "Y_skeleton", "Z_skeleton"],
    )

    # 3-POINT RELOCATION DISTANCE
    # Calculating the relocation distance of each point:
    #   -according to the equation parameters defined by the optimizer
    #   -based on the distance (r) between the point and its clone in the skeleton
    #   -based on the selected threshold

    if threshold == "Skeleton index":
        cloud3["relocation_dist"] = (
            param["m1"] * cloud3["r"] ** 2 + param["m2"] * cloud3["r"] + param["b"]
        )
        # Condition for relocation not to exceed distance 'r'
        cloud3["relocation_dist"] = np.where(
            cloud3["relocation_dist"] > cloud3["r"],
            cloud3["r"],
            cloud3["relocation_dist"],
        )
        # Retain only points below the SI_threshold
        cloud3 = cloud3[cloud3["r"] <= param["SI_threshold"]]
    else:
        mask = cloud3["1st_neighbor_dist"] > param["FI_threshold"]
        cloud3.loc[mask, "relocation_dist"] = cloud3.loc[mask, "r"]
        cloud3.loc[~mask, "relocation_dist"] = (
            param["m1"] * cloud3.loc[~mask, "r"] ** 2
            + param["m2"] * cloud3.loc[~mask, "r"]
            + param["b"]
        )
        # Condition for relocation not to exceed distance 'r'
        cloud3["relocation_dist"] = np.where(
            cloud3["relocation_dist"] > cloud3["r"],
            cloud3["r"],
            cloud3["relocation_dist"],
        )

    # 4-POINT RELOCATION POSITION
    # Calculating x, y and z relocation for each point
    cloud4 = locator.convert2cartesian(
        cloud3.copy(),
        cloud3["relocation_dist"],
        nm_x="diff_X",
        nm_y="diff_Y",
        nm_z="diff_Z",
        nm_polar="polar",
        nm_azimuth="azimuth",
    )
    # Calculating each point's final coordinates according to its relocation
    cloud4[["X", "Y", "Z"]] += np.array(
        [cloud4["diff_X"], cloud4["diff_Y"], cloud4["diff_Z"]]
    ).T

    return cloud4


if __name__ == "__main__":

    # 1-SELECT THRESHOLD
    threshold = select_threshold(utilisation="compressor")
    if threshold is None:
        print("##############################################")
        print("Process canceled by user.")
        exit()

    # 2-CHOOSE WORKING DIRECTORY
    # User must indicate where the cloud and skeleton files are located.
    work_directory = easygui.diropenbox(
        title="WORKING DIRECTORY",
        msg="Select the directory containing the original point cloud file",
    )
    if work_directory is None:
        print("##############################################")
        print("Process canceled by user.")
        exit()
    os.chdir(work_directory)

    # 3-IMPORT CLOUD FILE
    cloud_file = easygui.fileopenbox(
        title="SELECT INPUT FILE",
        msg="Select the original point cloud file (*.ply) you wish to compress",
    )
    if cloud_file is None:
        print("##############################################")
        print("Process canceled by user.")
        exit()
    if cloud_file.endswith(".ply"):
        orig_cloud = reader.read_ply(cloud_file)
        orig_cloud = valid.validate_columns(orig_cloud, threshold)
        if orig_cloud is None:
            print("##############################################")
            print("Process canceled by user.")
            exit()
    else:
        errmsg = "This file format is not supported. Please use a *.ply file."
        error_message(errmsg)
        exit()

    # USER'S ENTRY VALIDATION
    # Generate a skeleton (or use an existing one)
    generate_skeleton = search_file("skeleton")
    if generate_skeleton is None:
        print("##############################################")
        print("Process canceled by user. Please try again.")
        exit()
    elif generate_skeleton == "Yes":
        # Set skeletonization parameters
        param_sk = dict()
        try:
            (
                param_sk["voxel_size"],
                param_sk["search_radius"],
                param_sk["max_relocation_dist"],
            ) = set_parameters(usage="skeleton")
            print("##############################################")
            print("The skeleton parameters have been defined as :")
            print("voxel_size =", param_sk["voxel_size"])
            print("search_radius =", param_sk["search_radius"])
            print("max_relocation_dist =", param_sk["max_relocation_dist"])
        except ValueError:
            print("##############################################")
            print(
                "The skeletonization process could not be completed "
                "without the required parameters."
            )
            print("Please try again.")
            exit()
        # Skeleton file save as *.csv
        default_skfile = os.path.basename(cloud_file).split(".")[0] + "_skeleton.csv"
        skeleton_file = easygui.filesavebox(
            title="SAVE_SKELETON.CSV",
            msg="Save point cloud skeleton file as *.csv",
            default=default_skfile,
            filetypes="*.csv",
        )
    else:
        skeleton_file = easygui.fileopenbox(
            title="SKELETON_FILE.CSV",
            msg="Select the skeleton file (*.csv) to use for compression",
        )

    # Set compression parameters
    # The value of each parameter is defined according to the result of the optimizer
    # and must be manually adjusted by the user when asked by this script.
    # The optimized parameter values can be found on the plot generated by the optimizer.
    param_comp = dict()
    if threshold == "Skeleton index":
        try:
            (
                param_comp["m1"],
                param_comp["m2"],
                param_comp["b"],
                param_comp["SI_threshold"],
            ) = set_parameters(usage="compression", threshold=threshold)
            print("##############################################")
            print("The compression parameters have been defined as :")
            print("m1 =", param_comp["m1"])
            print("m2 =", param_comp["m2"])
            print("b =", param_comp["b"])
            print("SI_threshold =", param_comp["SI_threshold"])
        except KeyError:
            print("##############################################")
            print(
                "The compression process could not be completed without "
                "the required parameters."
            )
            print("Please try again.")
            exit()
    else:
        try:
            (
                param_comp["m1"],
                param_comp["m2"],
                param_comp["b"],
                param_comp["FI_threshold"],
            ) = set_parameters(usage="compression", threshold=threshold)
            print("##############################################")
            print("The compression parameters have been defined as :")
            print("m1 =", param_comp["m1"])
            print("m2 =", param_comp["m2"])
            print("b =", param_comp["b"])
            print("FI_threshold =", param_comp["FI_threshold"])
        except KeyError:
            print("##############################################")
            print(
                "The compression process could not be completed without "
                "the required parameters."
            )
            print("Please try again.")
            exit()

    # Output file save as *.ply
    default_compressed_file = (
            os.path.basename(cloud_file).split(".")[0] + "_compressed.ply"
    )
    compressed_file = easygui.filesavebox(
        title="COMPRESSED_FILE.PLY",
        msg="Save the compressed file as *.ply",
        default=default_compressed_file,
        filetypes="*.ply",
    )

    # Cloud data validation and identification depending on the compression threshold
    if threshold == "Skeleton index":
        cloud_data = valid.validate_data(orig_cloud, "compressor", threshold)
    else:
        if "1st_neighbor_dist" not in orig_cloud.columns:
            cloud_data1 = valid.validate_data(
                orig_cloud, "point_indexation", threshold
            )
            cloud_data2 = fraternity_indexer.arc_separator(cloud_data1)
            cloud_data3 = cloud_data2.replace(
                [np.inf, -np.inf], 5
            )  # Points with no "brother" are given a FI of 5
            cloud_data = valid.validate_data(cloud_data3, "compressor", threshold)
        else:
            cloud_data = valid.validate_data(orig_cloud, "compressor", threshold)

    print("##############################################")
    print("Working on point indexation...")
    print(
        "Point indexation successfully processed with the %s threshold." % threshold
    )

    # 4-SKELETONIZATION
    if generate_skeleton == "No":
        skeleton_data = reader.read_csv(skeleton_file, ",")
        # VALIDATION to ensure that all skeleton lines match the cloud lines
        data, cloud_data = valid.validate_data(cloud_data, "skeleton")
        # ADD missing columns to validated cloud_data
        if threshold == "Fraternity index":
            cloud_data = pd.merge(
                cloud_data,
                data[["line_id", "1st_neighbor_dist"]],
                on="line_id",
                how="left",
            )
    elif generate_skeleton == "Yes":
        # VALIDATION to ensure that all skeleton lines match the cloud lines
        data, cloud_data = valid.validate_data(cloud_data, "skeleton")
        # ADD missing columns to validated cloud_data
        if threshold == "Fraternity index":
            cloud_data = pd.merge(
                cloud_data,
                data[["line_id", "1st_neighbor_dist"]],
                on="line_id",
                how="left",
            )
        # GENERATE SKELETON
        print("##############################################")
        print("Generating skeleton...")
        print("This may take a few minutes. Please wait.")
        skeleton = skeletonizer.generate_skeleton(cloud_data, param_sk)

        # ADD missing columns to the skeleton file
        cols_to_add = [col for col in data.columns if col not in skeleton.columns or col == "line_id"]
        skeleton_data = pd.merge(data[cols_to_add], skeleton, on="line_id", how="left")
        writer.write_csv(skeleton_file, skeleton_data)
        print(" ")
        print("Skeletonization completed successfully.")
    else:
        print("##############################################")
        print("Process canceled by user.")
        exit()

    # 5-COMPRESS POINT CLOUD
    print("##############################################")
    print("Point cloud compression in progress...")
    result = compress_cloud(cloud_data, skeleton_data, param_comp, threshold)

    # ADD missing columns to the compressed file
    cols_to_add = [col for col in data.columns if col not in result.columns or col == "line_id"]
    result_data = pd.merge(result, data[cols_to_add], on="line_id", how="left")

    # Save compressed cloud
    writer.write_ply(compressed_file, result_data)

    print("##############################################")
    print(
        "The point cloud was successfully compress with the %s threshold." % threshold
    )
