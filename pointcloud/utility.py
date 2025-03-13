"""This module performs various utility tasks.

It requires that the following packages be installed within the Python
environment you are running this script in: datetime, easygui (0.98.3)

This file can be imported as a module and contains the following functions:
* calculate_time - calculates the time taken to perform a task
* error_message - displays an error message to the user
* search_file - allows the use of an existing file or the choice to create a new one
* select_threshold - allows the user to select a compression threshold
* set_parameters - allows the user to set the required parameters values
* validate_output - ensures that the user has entered all required values
"""

from datetime import datetime
import easygui
import os

global timer


def calculate_time(string, reset_timer=True):
    """This function calculates the time taken to perform a task."""
    global timer
    t_calc = datetime.now() - timer
    print("%s, duration:%s" % (string, t_calc))
    if reset_timer:
        timer = datetime.now()


def error_message(errmsg, title="AN ERROR OCCURRED"):
    """This function displays an error message to the user."""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "infographics"))
    img_path = os.path.join(project_dir, "erricon.png")
    easygui.buttonbox(errmsg, title, choices=["OK"], image=img_path)


def search_file(usage):
    """This function allows the user to generate or use an existing skeleton
    and returns the user's choice."""
    if usage == "optimization":
        msg = (
            "Individual stem files will be grouped into a single reference file for optimization."
            "\n\nIf a grouped reference file has already been created,"
            "you can answer 'No' to the following question."
            "\n\nDo you wish to create a grouped reference file?"
        )
        title = "CREATE REFERENCE FILE"
        choices = ["Yes", "No"]
        print_msg = "Create reference file ? : "
    elif usage == "skeleton":
        msg = (
            "Each point cloud must have a skeleton generated before compression."
            "\n\nIf a skeleton has already been generated for this point cloud, "
            "you can answer 'No' to the following question."
            "\n\nDo you wish to generate a new skeleton for this point cloud?"
        )
        title = "GENERATE SKELETON"
        choices = ["Yes", "No"]
        print_msg = "Generate new skeleton ? : "
    else:
        return None

    answer = easygui.buttonbox(msg, title, choices)
    print("##############################################")
    print(print_msg, end=" ")
    print(answer)

    return answer


def select_threshold(utilisation="compressor"):
    """This function allows the user to select a compression threshold
    and returns the selected threshold."""
    if utilisation == "optimizer":
        msg = (
            "Select threshold to optimize :\n"
            "\n  - Skeleton index "
            "\n  - Fraternity index (requires frame number, polar angle and distance to LiDAR)"
        )
    else:
        msg = (
            "Select threshold for compression:\n"
            "\n  - Skeleton index "
            "\n  - Fraternity index (requires frame number, polar angle and distance to LiDAR)"
        )

    title = "SELECT COMPRESSION THRESHOLD"
    choices = ["Skeleton index", "Fraternity index"]

    threshold = easygui.buttonbox(msg, title, choices)

    print("##############################################")
    print("Selected threshold : ", end=" ")
    print(threshold)

    return threshold


def set_parameters(usage="compression", threshold="Fraternity index"):
    """This function allows the user to set the required parameters values
    for skeletonization or compression and returns the user's settings."""
    if usage == "skeleton":
        text = (
            "Define the skeletonization parameters: "
            "\n(or press OK to continue with default values)"
        )
        title = "SKELETON PARAMETERS"

        input_list = [" voxel_size", " search_radius", " max_relocation_dist"]
        default_list = [0.01, 0.1, 0.21]

    else:
        text = (
            "Define the compression parameters: "
            "\n(or press OK to continue with default values)"
            "\n\n Optimized parameter values and threshold can be"
            " found on the 'optimized_graph_results.png' file "
            "(see optimizer results folder)."
        )
        title = "COMPRESSION PARAMETERS"

        if threshold == "Fraternity index":
            input_list = [" m1", " m2", " b", " FI_threshold"]
            default_list = [0.00, 0.98, 0.00, 0.171]
        else:
            input_list = [" m1", " m2", " b", " SI_threshold"]
            default_list = [0.00, 0.98, 0.00, 1.000]

    output = easygui.multenterbox(text, title, input_list, default_list)

    def validate_output(output_values, usage):
        """This function ensures that the user has entered all required values."""
        errmsg = ""
        if usage == "skeleton":
            for i, name in enumerate(input_list):
                if output_values[i].strip() == "":
                    errmsg += f"The {''.join(name)} parameter is missing.\n\n"
                if float(output_values[i].strip()) <= 0:
                    errmsg += "All parameters must be greater than 0.\n\n"
        else:
            for i, name in enumerate(input_list):
                if output_values[i].strip() == "":
                    errmsg += f"The {''.join(name)} parameter is missing.\n\n"
            if float(output_values[3].strip()) < 0:
                errmsg += "Threshold value must be positive.\n\n"
        if errmsg == "":
            output_values = [float(i) for i in output_values]
        else:
            output_values = easygui.multenterbox(
                errmsg, title, input_list, output_values
            )
            output_values = validate_output(output_values, usage)

        return output_values

    output = validate_output(output, usage)

    return output
