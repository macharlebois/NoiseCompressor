"""This module determines the optimal compression parameters.

The optimization process uses a genetic algorithm (GA) which
evaluates the impact of different parameter combinations on
the accuracy of DBH (diameter at breast height) measurements,
comparing LiDAR-derived values with field measurements.

As the algorithm tests each parameter combination, it calculates
their fitness score based on the relative DBH difference which
enable a comparative evaluation of each parameter combination's
performance.

This script requires that the following packages be installed within
the Python environment you are running this script in:
array, datetime, deap (1.4.1), easygui (0.98.3), functools, matplotlib (3.7.1),
multiprocessing, numpy (1.24.2), os, pandas (1.5.3), random

This script also uses the following local modules:
'reader', 'writer', 'utility', 'validator', 'fraternity_indexer',
'skeletonizer' from the local 'pointcloud' package, as well as
functions from the 'step2_compressor' local module.

This file can be imported as a module and contains the following functions:
* generate_param1 - generates a random parameter within the specified limits
* generate_individual - generates an individual with specified parameters
* evaluate_individual - evaluates an individual based on the parameters
* estimate_dbh - estimates the dbh of a stem
* function2optimize - function to optimize
* custom_mutation - custom mutation function
* main_optimizer - main optimizer function
"""

import array
from datetime import datetime
from deap import creator, base, tools, algorithms
import easygui
from functools import partial
import gc
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import polars as pl
from pointcloud.utility import (
    error_message,
    search_file,
    select_threshold,
    colored,
)
from pointcloud import (
    writer as writer,
    reader as reader,
    validator as valid,
    skeletonizer as skeletonizer,
    fraternity_indexer as fraternity_indexer,
    locator as locator,
)
import random
import step2_COMPRESSOR as Compressor


def generate_param1(min, max, step):
    """Generates a random parameter within the specified limits.
    Parameters:
        min (float): The minimum value of the parameter.
        max (float): The maximum value of the parameter.
        step (float): The step value of the parameter.
    Returns:
        float: A random parameter within the specified limits.
    """
    return round(random.uniform(min, max) / step) * step


def generate_individual(param_limits):
    """Generates an individual with specified parameters.
    Parameters:
        param_limits (DataFrame): The list and limits of the parameters to generate the individual.
    Returns:
        creator.Individual: An individual with specified parameters.
    """
    ind = creator.Individual()
    for index, row in param_limits.iterrows():
        parameter = generate_param1(row["Min"], row["Max"], row["Step"])
        ind.append(parameter)
    return ind


def evaluate_individual(
    individual, param_list, ref_data, skeleton_data, true_dbh, threshold
):
    """Evaluates an individual based on the parameters provided.
    Parameters:
        individual (creator.Individual): The individual to evaluate.
        param_list (list): The list of parameters to evaluate.
        ref_data (DataFrame): The data of the reference point cloud.
        skeleton_data (DataFrame): The data of the skeleton.
        true_dbh (DataFrame): The data of the true dbh.
        threshold (str): The threshold to use for the optimization.
    Returns:
        tuple: A tuple containing the dbh error of the individual.
    """
    param = dict(zip(param_list, individual))
    dbh_error_sum, table_relation = function2optimize(
        ref_data, skeleton_data, true_dbh, param, threshold
    )

    return (dbh_error_sum,)


def estimate_dbh(data, stem_id, iterations=20, dbh_height=1.20):
    """Estimates the observed width of a stem, as a proxy for its DBH.
    Parameters:
        data (DataFrame): The data of the stem.
        iterations (int): The number of iterations to perform.
        dbh_height (float): The height at which to evaluate the dbh.
    Returns:
        tuple: A tuple containing the mean X and Y distances, and the mean dbh."""
    data2 = pd.DataFrame(columns=["Iteration", "X_diff", "Y_diff"])

    for i in range(iterations):
        # Data filtering by layer (thickness = 5cm)
        layer = data[
            (data["Z"] >= dbh_height - 0.025) & (data["Z"] <= dbh_height + 0.025)
        ]
        if layer.empty:
            continue

        # Calculating the X distance between the 99th percentile and the 1st percentile
        # Here, we use the 1st and 99th percentiles to avoid outliers.
        x_max = np.percentile(layer["X"], 99)
        x_min = np.percentile(layer["X"], 1)
        x_difference = x_max - x_min

        # Calculating the Y distance between the 99th percentile and the 1st percentile
        # Again, we use the 1st and 99th percentiles to avoid outliers.
        y_max = np.percentile(layer["Y"], 99)
        y_min = np.percentile(layer["Y"], 1)
        y_difference = y_max - y_min

        new_row = pd.DataFrame(
            {"Iteration": [i + 1], "X_diff": [x_difference], "Y_diff": [y_difference]}
        )
        if data2.empty:
            data2 = new_row.copy()
        else:
            data2 = pd.concat([data2, new_row], ignore_index=True)

        # Adding 1.5cm to the dbh evaluation height for every iteration
        # With 20 iterations, the dbh is calculated over a height of 30 cm,
        # between 1.20 and 1.50 meters from the ground.
        dbh_height += 0.015

    try:
        mean_X = np.round(data2["X_diff"].mean(), 3)
        mean_Y = np.round(data2["Y_diff"].mean(), 3)
        mean_dbh = np.round((mean_X + mean_Y) / 2, 3)

    except ValueError:
        print(f"ERROR WARNING: DBH estimation failed for stem '{stem_id}' "
              f"due to invalid data (NO POINTS FOUND AT DBH HEIGHT)")
        mean_X = 0
        mean_Y = 0
        mean_dbh = 0

    return mean_X, mean_Y, mean_dbh


def function2optimize(ref_data, skeleton_data, true_dbh, param, threshold):
    """This function is used to optimize the compression parameters.
    Parameters:
        ref_data (DataFrame): The data of the reference point cloud.
        skeleton_data (DataFrame): The data of the skeleton.
        true_dbh (DataFrame): The data of the true dbh.
        param (dict): The parameters to optimize.
        threshold (str): The threshold to use for the optimization.
    Returns:
        tuple: A tuple containing the dbh error sum and the dbh error table of the individual.
    """
    if isinstance(ref_data, pd.DataFrame):
        ref_data = pl.from_pandas(ref_data)
    result = Compressor.compress_cloud(ref_data, skeleton_data, param, threshold)
    result = result.to_pandas()
    dbh_error_tab = pd.DataFrame(
        columns=["stem_id", "estimated_dbh", "rel_dbh_diff", "true_dbh", "dbh_diff"]
    )

    # Estimating each stem's dbh
    for index, row in true_dbh.iterrows():
        stem_id = row["stem_id"].astype(int)
        subset = ["X", "Y", "Z"]  # Columns to keep
        stem_xyz = result.loc[result["stem_id"] == stem_id, subset]
        mean_X, mean_Y, mean_dbh = estimate_dbh(stem_xyz, stem_id, dbh_height=1.20)

        # Calculating the difference between the estimated and field-measured dbh
        dbh_diff = abs(mean_dbh - row["true_dbh"])
        rel_dbh_diff = abs((mean_dbh - row["true_dbh"]) / row["true_dbh"])

        new_row = pd.DataFrame(
            {
                "stem_id": [stem_id],
                "estimated_dbh": [mean_dbh],
                "rel_dbh_diff": [rel_dbh_diff],
                "true_dbh": [row["true_dbh"]],
                "dbh_diff": [dbh_diff],
            }
        )

        if dbh_error_tab.empty:
            dbh_error_tab = new_row.copy()
        else:
            dbh_error_tab = pd.concat([dbh_error_tab, new_row], ignore_index=True)

    # Handling NaN values for the SI method.
    # If the SI_threshold is too low, some stems may not be evaluated because
    # of the amount of points removed. If so, a value of 99 is returned to
    # indicate that the optimization failed.
    nan_count = dbh_error_tab["estimated_dbh"].isna().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} stems could not be evaluated.")
        return 99, dbh_error_tab

    else:
        # Fitness equation to optimize based on the relative difference between the
        # estimated and true dbh: Σ (([true_dbh - estimated_dbh] / true_dbh)^2)
        dbh_error_sum = ((1 / (dbh_error_tab["true_dbh"])) * (
                dbh_error_tab["true_dbh"] - dbh_error_tab["estimated_dbh"]) ** 2).sum()
        return dbh_error_sum, dbh_error_tab


def custom_mutation(individual, p_limits):
    """This function performs a custom mutation on an individual.
    Parameters:
        individual (creator.Individual): The individual to mutate.
        p_limits (DataFrame): The limits of the parameters.
    Returns:
        tuple: A tuple containing the mutated individual.
    """
    potential_mutant = generate_individual(p_limits)  # creating mutant genes
    mutated_individual = []
    for i, gene in enumerate(individual):
        if random.random() < 0.05:  # Here, each gene has a 5% potential of mutation
            mutated_individual.append(potential_mutant[i])
        else:
            mutated_individual.append(gene)
    return (creator.Individual(mutated_individual),)


def main_optimizer(true_dbh, ref_data, skeleton_data, p_limits, threshold):
    """This function optimizes the compression parameters using a genetic algorithm.
    Parameters:
        true_dbh (DataFrame): The data of the true dbh.
        ref_data (DataFrame): The data of the reference point cloud.
        skeleton_data (DataFrame): The data of the skeleton.
        p_limits (DataFrame): The limits of the parameters.
        threshold (str): The threshold to use for the optimization.
    Returns:
        tuple: A tuple containing the optimized relation table and the optimized parameter table.
    """
    p_column = list(p_limits["Parameter"])

    # Creating FitnessMax class
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    # Creating Individual class
    creator.create("Individual", array.array, typecode="f", fitness=creator.FitnessMax)

    if threshold == "Skeleton index":
        max_r_data = skeleton_data[["oldX", "oldY", "oldZ", "X", "Y", "Z"]].copy()
        max_r_data = locator.convert2spherical(max_r_data, dimension1=["oldX", "oldY", "oldZ"], dimension2=["X", "Y", "Z"])
        max_r = max_r_data["r"].max()

        p_limits.loc[p_limits["Parameter"] == "SI_threshold", "Max"] = round(max_r, 4)

        del max_r_data
        gc.collect()

    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, p_limits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    evaluate_partial = partial(
        evaluate_individual,
        param_list=p_column,
        ref_data=ref_data,
        skeleton_data=skeleton_data,
        true_dbh=true_dbh,
        threshold=threshold,
    )

    toolbox.register("evaluate", evaluate_partial)
    toolbox.register("mate", tools.cxTwoPoint)
    mutate_partial = partial(custom_mutation, p_limits=p_limits)
    toolbox.register("mutate", mutate_partial)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(64)
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=10, stats=stats, halloffame=hof
    )
    GA_relation_tab = pd.DataFrame()
    GA_param_tab = pd.DataFrame()
    index = 1
    for ind in hof:
        param = dict(zip(p_column, ind))
        dbh_error_sum, dbh_error_tab = function2optimize(
            ref_data, skeleton_data, true_dbh, param, threshold
        )
        dbh_error_tab["individual"] = "hof%s" % index
        GA_relation_tab = pd.concat([GA_relation_tab, dbh_error_tab])
        df_param = pd.DataFrame([param], index=["row1"])
        df_param["individual"] = "hof%s" % index
        GA_param_tab = pd.concat([GA_param_tab, df_param])
        index += 1

    index = 1
    for ind in pop:
        param = dict(zip(p_column, ind))
        dbh_error_sum, dbh_error_tab = function2optimize(
            ref_data, skeleton_data, true_dbh, param, threshold
        )
        dbh_error_tab["individual"] = "pop%s" % index
        GA_relation_tab = pd.concat([GA_relation_tab, dbh_error_tab])
        df_param = pd.DataFrame([param], index=["row1"])
        df_param["individual"] = "pop%s" % index
        GA_param_tab = pd.concat([GA_param_tab, df_param])
        index += 1

    return GA_relation_tab, GA_param_tab


def calculate_limits(data1, data2, margin_ratio=0.1):
    """Calculates the limits of the data for plotting purposes.
    Parameters:
        data1 (array): The first data array.
        data2 (array): The second data array.
        margin_ratio (float): The margin ratio to apply.
    Returns:
        tuple: A tuple containing the minimum and maximum values of the data.
    """
    y_min = min(np.nanmin(data1), np.nanmin(data2))
    y_max = max(np.nanmax(data1), np.nanmax(data2))
    margin = (y_max - y_min) * margin_ratio
    return y_min - margin, y_max + margin


if __name__ == "__main__":

    # 1-SELECT THRESHOLD
    threshold = select_threshold(utilisation="optimizer")
    if threshold is None:
        print(" ")
        print(colored("Process canceled by user...", 'red'))
        exit()

    # 2-CHOOSE WORKING DIRECTORY
    work_directory = easygui.diropenbox(
        title="WORKING DIRECTORY",
        msg="Select working directory containing input files "
        "(stem_information.xlsx - individual_stem_files)",
    )
    if work_directory is None:
        print(" ")
        print(colored("Process canceled by user...", 'red'))
        exit()
    os.chdir(work_directory)
    try:
        stem_info = reader.read_xlsx("stem_information.xlsx", sheet_name="stem_infos")
    except FileNotFoundError:
        errmsg = "The stem_information.xlsx file could not be found."
        error_message(errmsg)
        exit()

    # USER'S ENTRY VALIDATION
    # Naming the reference and skeleton output files
    if threshold == "Skeleton index":
        ref_file = "ref_stems_SI.ply"
        skeleton_file = "ref_skeleton_SI.csv"
    else:
        ref_file = "ref_stems_FI.ply"
        skeleton_file = "ref_skeleton_FI.csv"

    # Generate a reference file (or use an existing one)
    generate_reference_file = search_file("optimization")
    if generate_reference_file is None:
        print(" ")
        print(colored("Process canceled by user...", 'red'))
        exit()
    if generate_reference_file == "No":
        ref_file = easygui.fileopenbox(
            title="REFERENCE FILE",
            msg="Select the grouped reference *.ply file for optimization",
        )

    # Generate a skeleton (or use an existing one)
    generate_skeleton = search_file("skeleton")
    if generate_skeleton is None:
        print(" ")
        print(colored("Process canceled by user...", 'red'))
        exit()
    elif generate_skeleton == "Yes":
        param_sk = {
            "voxel_size": 0.01,
            "search_radius": 0.1,
            "max_relocation_dist": 0.21,
        }
    else:
        skeleton_file = easygui.fileopenbox(
            title="SKELETON FILE", msg="Select the skeleton *.csv file for optimization"
        )


    # Creating the output folder (if non-existent)
    if threshold == "Skeleton index":
        output_folder = "optimizer_results_SI"
    else:
        output_folder = "optimizer_results_FI"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 3-GENERATE REFERENCE FILE (from single stem files)
    if generate_reference_file == "No":  # verifies if the file exists
        print(" ")
        print(colored("Reading reference cloud..."))
        ref_data = reader.read_ply(ref_file)
        print("Reference cloud successfully loaded.")
        ref_data = valid.validate_columns(ref_data, threshold)
        if ref_data is None:
            print(" ")
            print(colored("Process canceled by user...", 'red'))
            exit()
    elif generate_reference_file == "Yes":
        dfs = []
        i = 1
        n = len(stem_info)
        for index, row in stem_info.iterrows():
            stem_file = (
                work_directory + r"\individual_stem_files" + "\\" + row["ind_stem_file"]
            )
            if stem_file.endswith(".csv"):
                try:
                    stem_data = reader.read_csv(stem_file, sep=None)
                except FileNotFoundError:
                    errmsg = (
                        f"The file {stem_file} could not be found."
                        f"\n\nPlease make sure the file exists in the"
                        f"'individual_stem_files' folder and try again."
                    )
                    error_message(errmsg)
                    exit()
            elif stem_file.endswith(".ply"):
                try:
                    stem_data = reader.read_ply(stem_file)
                except FileNotFoundError:
                    errmsg = (
                        f"The file {stem_file} could not be found."
                        f"\n\nPlease make sure the file exists in the"
                        f"'individual_stem_files' folder and try again."
                    )
                    error_message(errmsg)
                    exit()
            else:
                errmsg = (
                    "The file format is not supported."
                    "Please use *.ply or *.csv files."
                )
                error_message(errmsg)
                exit()
            stem_data = valid.validate_columns(stem_data, threshold)
            if stem_data is None:
                print(" ")
                print(colored("Process canceled by user...", 'red'))
                exit()
            stem_data = valid.validate_data(stem_data, "point_indexation", threshold)
            if "stem_id" not in stem_data.columns:
                stem_data.insert(0, "stem_id", row["stem_id"])
            if threshold == "Skeleton index":
                dfs.append(stem_data)
            else:
                print("")
                print(colored("Processing individual stem files..."))
                print("This may take a moment. Please wait.")
                print("Current: " + row["ind_stem_file"] + " (%s of %s)" % (i, n))
                # Fraternity index calculation
                stem_data2 = fraternity_indexer.arc_separator(stem_data)
                # Points with no "brother" are given a FI of 5
                stem_data3 = stem_data2.replace([np.inf, -np.inf], 5)
                dfs.append(stem_data3)
                i = i + 1
        ref_data = pd.concat(dfs, ignore_index=True)
        writer.write_ply(ref_file, ref_data)
        print(" ")
        print(
            "Stem identification successfully completed with the %s threshold."
            % threshold
        )
    else:
        print(" ")
        print(colored("Process canceled by user...", 'red'))
        exit()

    # Saving untreated results
    untreated_output = pd.DataFrame(columns=["stem_id", "estimated_dbh", "true_dbh"])
    for index, row in stem_info.iterrows():
        stem_id = row["stem_id"]
        subset = ["X", "Y", "Z"]
        stem_xyz = ref_data.loc[ref_data["stem_id"] == stem_id, subset]
        mean_X, mean_Y, mean_dbh = estimate_dbh(stem_xyz, stem_id, dbh_height=1.20)
        dbh_diff = mean_dbh - row["true_dbh"]
        new_row = pd.DataFrame(
            {
                "stem_id": [stem_id],
                "estimated_dbh": [mean_dbh],
                "true_dbh": [row["true_dbh"]],
                "dbh_diff": [dbh_diff],
            }
        )
        untreated_output = pd.concat([untreated_output, new_row], ignore_index=True)
    writer.write_xlsx("untreated_relation.xlsx", untreated_output)

    # 4-SKELETONIZATION
    if generate_skeleton == "No":
        skeleton_data = reader.read_csv(skeleton_file, ",")
        data, ref_data = valid.validate_data(ref_data, "skeleton")
    elif generate_skeleton == "Yes":
        data, ref_data = valid.validate_data(ref_data, "skeleton")
        print(" ")
        print(colored("Generating skeleton..."))
        print("This may take a moment. Please wait.")
        ref_data = pl.from_pandas(ref_data)
        skeleton = skeletonizer.generate_skeleton(ref_data, param_sk)
        ref_data = ref_data.to_pandas()
        skeleton = skeleton.to_pandas()
        print(" ")
        print("Skeletonization completed " + colored("successfully", "C14") + ".")
        print("Beginning the compression parameters optimization...")

        # ADD missing columns to the skeleton file
        cols_to_add = [col for col in data.columns if col not in skeleton.columns or col == "line_id"]
        skeleton_data = pd.merge(data[cols_to_add], skeleton, on="line_id", how="left")
        skeleton_data["stem_id"] = skeleton_data["stem_id"].astype(int)
        writer.write_csv(skeleton_file, skeleton_data)
    else:
        print(" ")
        print(colored("Process canceled by user...", 'red'))
        exit()

    # Add missing columns to the ref_data file
    if threshold == "Skeleton index":
        ref_data = pd.merge(
            ref_data, data[["line_id", "stem_id"]], on="line_id", how="left"
        )
    else:
        ref_data = pd.merge(
            ref_data,
            data[["line_id", "stem_id", "1st_neighbor_dist"]],
            on="line_id",
            how="left",
        )
    ref_data["stem_id"] = ref_data["stem_id"].astype(int)

    # Selection of stems for optimization
    stem_info = stem_info.loc[stem_info["optimize"] == 1, ["stem_id", "true_dbh"]]
    ref_data = ref_data[ref_data["stem_id"].isin(stem_info["stem_id"])]
    skeleton_data = skeleton_data[skeleton_data["line_id"].isin(ref_data["line_id"])]

    # 5-PARAMETRIZATION BY OPTIMIZATION
    param_limits = pd.DataFrame(columns=["Parameter", "Min", "Max", "Step"])
    if threshold == "Fraternity index":
        param_limits["Parameter"] = ["m1", "m2", "b", "FI_threshold"]
        param_limits["Min"] = [0, -1, 0, ref_data["1st_neighbor_dist"].min()]
        param_limits["Max"] = [0, 1, 0, ref_data.loc[ref_data["1st_neighbor_dist"] < 5, "1st_neighbor_dist"].max()]
        param_limits["Step"] = [0.01, 0.01, 0.001, 0.001]
    else:
        param_limits["Parameter"] = ["m1", "m2", "b", "SI_threshold"]
        param_limits["Min"] = [0, -1, 0, 0]
        param_limits["Max"] = [0, 1, 0, 1]
        param_limits["Step"] = [0.01, 0.01, 0.001, 0.001]
    GA_relation_tab, GA_param_tab = main_optimizer(
        stem_info, ref_data, skeleton_data, param_limits, threshold
    )

    # 6-SAVING BEST OUTPUT AND PLOTTING RESULTS
    plot_individual = ["hof1"]  # Hall of fame 1 (winner)
    dataplot = GA_relation_tab[GA_relation_tab["individual"].isin(plot_individual)]
    dataplot_param = GA_param_tab[GA_param_tab["individual"].isin(plot_individual)]

    # Saving output files
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime("%I%M%S%p")
    relation_file_path = os.path.join(
        output_folder, "optimized_relation_" + formatted_time + ".xlsx"
    )
    writer.write_xlsx(relation_file_path, dataplot)
    param_file_path = os.path.join(
        output_folder, "optimized_param_" + formatted_time + ".xlsx"
    )
    writer.write_xlsx(param_file_path, dataplot_param)
    print(" ")
    print("Optimization completed " + colored("successfully", "C14") + ".")
    print("Plotting the results...")

    # Plotting the results
    method = "Untreated"
    GA_relation_tab_UN = untreated_output

    if threshold == "Skeleton index":
        method2 = r"$\bf{Compression\ with\ Skeleton\ Index\ (SI)}$"
    else:
        method2 = r"$\bf{Compression\ with\ Fraternity\ Index\ (FI)}$"

    true_dbh2 = dataplot["true_dbh"] * 100
    estimated_dbh2 = dataplot["estimated_dbh"] * 100
    dbh_diff2 = dataplot["dbh_diff"] * 100
    rel_dbh_diff2 = dataplot["rel_dbh_diff"] * 100

    dataplot_UN = GA_relation_tab_UN[
        GA_relation_tab_UN["stem_id"].isin(dataplot["stem_id"])
    ]
    estimated_dbh_UN = dataplot_UN["estimated_dbh"] * 100
    dbh_diff_UN = dataplot_UN["dbh_diff"] * 100
    rel_dbh_diff_UN = (
        abs(
            (dataplot_UN["estimated_dbh"] - dataplot_UN["true_dbh"])
            / dataplot_UN["true_dbh"]
        )
        * 100
    )

    m1 = dataplot_param.iloc[0, 0]
    m2 = dataplot_param.iloc[0, 1]
    b = dataplot_param.iloc[0, 2]
    compressor_eq = f"RD = {m1:.3f} r$^2$ + {m2:.3f} r + {b:.3f}"

    if threshold == "Skeleton index":
        SI_threshold = dataplot_param.iloc[0, 3]
        threshold_text = f"SI_threshold : {SI_threshold:.4f}"
    else:
        FI_threshold = dataplot_param.iloc[0, 3]
        threshold_text = f"FI_threshold : {FI_threshold:.4f}"

    # Create the graphs
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    axs[0, 0].set_title(method)
    axs[0, 1].set_title(method2 + "\n" + compressor_eq + "\n" + threshold_text)

    # First row : estimated_dbh vs true_dbh
    axs[0, 0].scatter(true_dbh2, estimated_dbh_UN, color="black", s=5, zorder=2)
    axs[0, 0].plot(true_dbh2, true_dbh2, color="darkturquoise", zorder=1)  # Ligne y = x
    axs[0, 0].set_ylabel("LIDAR-DBH (cm)")

    axs[0, 1].scatter(true_dbh2, estimated_dbh2, color="black", s=5, zorder=2)
    axs[0, 1].plot(true_dbh2, true_dbh2, color="darkturquoise", zorder=1)  # Ligne y = x

    # Sync limits for the first row
    y_min, y_max = calculate_limits(estimated_dbh_UN, estimated_dbh2)
    axs[0, 0].set_ylim(y_min, y_max)
    axs[0, 1].set_ylim(y_min, y_max)

    # Second row : dbh_diff vs true_dbh
    axs[1, 0].scatter(true_dbh2, dbh_diff_UN, color="black", s=5, zorder=2)
    axs[1, 0].axhline(0, color="darkturquoise", zorder=1)  # Ligne y = 0
    axs[1, 0].set_ylabel("AE (cm)")

    axs[1, 1].scatter(true_dbh2, dbh_diff2, color="black", s=5, zorder=2)
    axs[1, 1].axhline(0, color="darkturquoise", zorder=1)  # Ligne y = 0

    # Sync limits for the second row
    y_min, y_max = calculate_limits(dbh_diff_UN, dbh_diff2)
    axs[1, 0].set_ylim(y_min, y_max)
    axs[1, 1].set_ylim(y_min, y_max)

    # Third row : rel_dbh_diff vs true_dbh (with log-transformed y-axis)
    axs[2, 0].scatter(true_dbh2, rel_dbh_diff_UN, color="black", s=5, zorder=2)
    axs[2, 0].axhline(0, color="darkturquoise", zorder=1)
    axs[2, 0].set_ylabel("RE (%)")
    axs[2, 0].set_xlabel("Field-DBH (cm)")

    axs[2, 1].scatter(true_dbh2, rel_dbh_diff2, color="black", s=5, zorder=2)
    axs[2, 1].axhline(0, color="darkturquoise", zorder=1)
    axs[2, 1].set_xlabel("Field-DBH (cm)")

    # Sync limits for the third row
    y_min, y_max = calculate_limits(rel_dbh_diff_UN, rel_dbh_diff2)
    axs[2, 0].set_ylim(y_min, y_max)
    axs[2, 1].set_ylim(y_min, y_max)

    # Adjust plot layout
    plt.tight_layout()

    # Save the plot
    plt_path = os.path.join(
        output_folder, "optimized_graph_results_" + formatted_time + ".png"
    )
    plt.savefig(plt_path)

    plt.show()
