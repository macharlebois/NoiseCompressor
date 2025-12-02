from datetime import datetime
import easygui
import matplotlib.pyplot as plt
import os
import pandas as pd
import polars as pl
from pointcloud.utility import (
    search_file,
    select_threshold,
    colored,
)
from pointcloud import (
    writer as writer,
    validator as valid,
    skeletonizer as skeletonizer,
)
from step1_OPTIMIZER import (
    main_optimizer,
    estimate_dbh,
    calculate_limits,
)

import io
from plyfile import PlyData
import urllib.request

global timer



if __name__ == "__main__":

    stem_info = "https://github.com/macharlebois/Sample_Data/raw/refs/heads/main/noise_compressor/demo_dataset/stem_information.xlsx"
    stem_info = pd.read_excel(stem_info, sheet_name="stem_infos")

    # 1-SELECT THRESHOLD
    threshold = select_threshold(utilisation="optimizer")
    while threshold == 'Fraternity index':
        print(" ")
        print(colored("Demo available for Skeleton index only", 'C3'))
        threshold = select_threshold(utilisation="optimizer")
    if threshold is None:
        print(" ")
        print(colored("Process canceled by user...", 'C3'))
        exit()

    # Generate a reference file (or use an existing one)
    generate_reference_file = search_file("optimization")  # ANSWER 'YES' FOR DEMO
    while generate_reference_file == "No":
        print(" ")
        print(colored("REFERENCE FILE REQUIRED FOR DEMO", 'C3'))
        generate_reference_file = search_file("optimization")
    if generate_reference_file is None:
        print(" ")
        print(colored("Process canceled by user...", 'C3'))
        exit()

    # Generate a skeleton (or use an existing one)
    generate_skeleton = search_file("skeleton")  # ANSWER 'YES' FOR DEMO
    while generate_skeleton == "No":
        print(" ")
        print(colored("SKELETON REQUIRED FOR DEMO", 'C3'))
        generate_skeleton = search_file("skeleton")
    if generate_skeleton is None:
        print(" ")
        print(colored("Process canceled by user...", 'C3'))
        exit()

    # 2-CHOOSE OUTPUT DIRECTORY
    work_directory = easygui.diropenbox(title="OUTPUT DIRECTORY", msg="Select your output directory")
    if work_directory is None:
        print(" ")
        print(colored("Process canceled by user...", 'C3'))
        exit()

    output_folder = work_directory + '\\' + 'optimizer_results_SI_demo'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Naming the reference and skeleton output files
    ref_file = work_directory + '\\' + 'ref_stems_SI.ply'
    skeleton_file = work_directory + '\\' + 'ref_skeleton_SI.csv'

    # 3-GENERATE REFERENCE FILE (from single stem files)
    if generate_reference_file == "Yes":
        dfs = []
        i = 1
        n = len(stem_info)
        print(" ")
        print(colored("Generating reference file..."))
        print("This may take a moment. Please wait.")

        for index, row in stem_info.iterrows():
            file = row["ind_stem_file"]
            print("Currently indexing: " + file + " (%s of %s)" % (i, n))

            stem_file = (f"https://github.com/macharlebois/Sample_Data/raw/refs/heads/main/"
                         f"noise_compressor/demo_dataset/individual_stem_files/{file}")
            with urllib.request.urlopen(stem_file) as response:
                ply_data = PlyData.read(io.BytesIO(response.read()))
            vertex_data = ply_data["vertex"].data
            stem_data = pd.DataFrame(vertex_data)
            stem_data["stem_ID"] = stem_data["stem_ID"].astype(int)
            stem_data = stem_data.rename(columns={"stem_ID": "stem_id"})

            stem_data = valid.validate_columns(stem_data, threshold)
            stem_data = valid.validate_data(stem_data, "point_indexation", threshold)
            dfs.append(stem_data)
            i = i + 1

        ref_data = pd.concat(dfs, ignore_index=True)
        writer.write_ply(ref_file, ref_data)
        print(" ")
        print("Stem identification successfully completed with the %s threshold." % threshold)
    else:
        print(" ")
        print(colored("Process canceled by user...", 'C3'))
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
    untreated_output_dir = work_directory + '\\' + 'untreated_relation.xlsx'
    writer.write_xlsx(untreated_output_dir, untreated_output)

    # 4-SKELETONIZATION
    if generate_skeleton == "Yes":
        data, ref_data = valid.validate_data(ref_data, "skeleton")
        print(" ")
        print(colored("Generating skeleton..."))
        print("This may take a moment (up to 20 minutes). Please wait.")
        param_sk = {
            "voxel_size": 0.01,
            "search_radius": 0.1,
            "max_relocation_dist": 0.21,
        }
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
        print(colored("Process canceled by user...", 'C3'))
        exit()

    # Add missing columns to the ref_data file
    if threshold == "Skeleton index":
        ref_data = pd.merge(
            ref_data, data[["line_id", "stem_id"]], on="line_id", how="left"
        )
    ref_data["stem_id"] = ref_data["stem_id"].astype(int)

    # Selection of stems for optimization
    stem_info = stem_info.loc[stem_info["optimize"] == 1, ["stem_id", "true_dbh"]]
    ref_data = ref_data[ref_data["stem_id"].isin(stem_info["stem_id"])]
    skeleton_data = skeleton_data[skeleton_data["line_id"].isin(ref_data["line_id"])]

    # 5-PARAMETRIZATION BY OPTIMIZATION
    param_limits = pd.DataFrame(columns=["Parameter", "Min", "Max", "Step"])
    if threshold == "Skeleton index":
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

    method2 = r"$\bf{Compression\ with\ Skeleton\ Index\ (SI)}$"

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

    SI_threshold = dataplot_param.iloc[0, 3]
    threshold_text = f"SI_threshold : {SI_threshold:.4f}"

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

    print("RESULTS SAVED IN : " + colored(output_folder, "C14"))
