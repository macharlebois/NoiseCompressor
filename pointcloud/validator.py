"""This module validates data according to their intended use.

This script requires that the following packages be installed within the Python
environment you are running this script in:
easygui (0.98.3), json, os, pandas (1.5.3), tkinter

This file can be imported as a module and contains the following functions:
* load_selections - load previous selections from a JSON file
* save_selections - save selections in a JSON file
* validate_columns - validate columns (in association with variables)
* validate_data - validates data according to their intended use
"""

import easygui
import json
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox


# File name for storing previous selections
previous_selection_file = "data_validation.json"


def load_selections():
    """Load previous selections from a JSON file."""
    if os.path.exists(previous_selection_file):
        with open(previous_selection_file, "r") as f:
            return json.load(f)
    return {}


def save_selections(selections):
    """Save selections in a JSON file."""
    with open(previous_selection_file, "w") as f:
        json.dump(selections, f)


def validate_columns(data, method):
    """This function validates columns association with variables
    to ensure that the data is correctly formatted and renames columns
    (if necessary).

    Returns
    -------
    pandas.DataFrame
        The dataframe with correctly named columns.
    """
    if method == "Skeleton index":
        variables = ["X", "Y", "Z", "relative_z"]
    else:
        variables = [
            "X",
            "Y",
            "Z",
            "polar",
            "frame_num",
            "dist_lidar",
            "relative_z",
        ]

    # Listing column names from dataframe
    column_names = list(data.columns)

    previous_selections = load_selections()

    # Check if the columns are already associated with the variables
    if previous_selections:
        matching_selections = all(
            var in previous_selections and previous_selections[var] in column_names
            for var in variables
        )
        if matching_selections:
            # If all variables are already associated with columns,
            # this renames and returns columns according to the previous selections
            rename_dict = {
                previous_selections[var]: var
                for var in variables
                if var in previous_selections
            }
            return data.rename(columns=rename_dict)

    # Stocking associations in a dictionary
    column_selection = {}

    cancelled = False

    def on_closing():
        """Function to manage window closing."""
        nonlocal cancelled
        cancelled = True
        root.destroy()

    def submit():
        """Function to retrieve selections."""
        selections = [combobox.get() for combobox in column_selection.values()]
        # Verification of empty selections
        if "" in selections:
            messagebox.showerror("Error", "Please select a column for each variable.")
            return

        # Verification of duplicate selections
        if len(selections) != len(set(selections)):
            messagebox.showerror(
                "Error", "Each column can be associated with only one variable."
            )
        else:
            # Association of variables with selected columns
            for var, combobox in column_selection.items():
                column_selection[var] = combobox.get()
            save_selections(column_selection)
            root.destroy()

    root = tk.Tk()
    root.minsize(500, 100)
    root.title("Please associate each variables with the correct column.")
    root.attributes("-topmost", True)

    # Attaching the close function to the window closing event
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Creating a drop-down list for each variable
    for var in variables:
        frame = ttk.Frame(root)
        frame.pack(pady=5)

        label = ttk.Label(frame, text=var, justify="left", width=15)
        label.pack(side="left")

        combobox = ttk.Combobox(frame, values=[""] + column_names, state="readonly")
        combobox.pack(side="right")
        default_value = previous_selections.get(var, "")
        if default_value in column_names:
            combobox.set(default_value)
        else:
            combobox.set("")  # Making sure that the default values are valid
        column_selection[var] = combobox

    submit_button = ttk.Button(root, text="Submit", command=submit)
    submit_button.pack(pady=20)

    root.mainloop()

    if cancelled:
        return None

    # Renaming columns to match associated variables
    rename_dict = {column_selection[var]: var for var in variables}
    renamed_data = data.rename(columns=rename_dict)

    return renamed_data


def validate_data(data, usage, threshold=None):
    """This function validates data according to its intended use :
    skeleton, point_indexation, compressor, optimizer
    :param data: pandas dataframe
    :param usage: str, usage type
    :param threshold: str, threshold type
    :return: original dataframe with IDs and dataframe for usage"""

    if usage == "skeleton":
        needed_columns = pd.Index(["X", "Y", "Z", "relative_z"])
        columns_upper_data = data.columns.str.upper()
        for column in needed_columns:
            try:
                position = columns_upper_data.get_loc(column.upper())
                old_name = data.columns[position]
                data = data.rename(columns={old_name: column})
            except KeyError:
                raise ValueError(
                    "The '%s' column could not be found in the point cloud." % column
                )
            if not data[column].dtype.kind == "f":
                raise ValueError("The '{}' column is not of type float".format(column))
        if "line_id" not in data.columns:
            data = data.reset_index(drop=False)
            data.rename(columns={"index": "line_id"}, inplace=True)
            data["line_id"] = data["line_id"].astype("int64")
        if not data["line_id"].is_unique:
            data = data.reset_index(drop=False)
            data.rename(columns={"index": "line_id"}, inplace=True)
            data["line_id"] = data["line_id"].astype("int64")
        data_xyz = data[["line_id", "X", "Y", "Z", "relative_z"]]

        return data, data_xyz

    elif usage == "point_indexation":
        if threshold == "Fraternity index":
            needed_columns = pd.Index(
                ["X", "Y", "Z", "relative_z", "dist_lidar", "polar", "frame_num"]
            )
        else:
            needed_columns = pd.Index(["X", "Y", "Z", "relative_z"])
        columns_upper_data = data.columns.str.upper()
        for column in needed_columns:
            try:
                position = columns_upper_data.get_loc(column.upper())
                old_name = data.columns[position]
                data = data.rename(columns={old_name: column})
            except KeyError:
                raise ValueError(
                    "The '%s' column could not be found in the point cloud." % column
                )

        return data

    elif usage == "compressor":
        if threshold == "Fraternity index":
            needed_columns = pd.Index(
                ["X", "Y", "Z", "1st_neighbor_dist", "relative_z"]
            )
        else:
            needed_columns = pd.Index(["X", "Y", "Z", "relative_z"])
        columns_upper_data = data.columns.str.upper()
        for column in needed_columns:
            try:
                position = columns_upper_data.get_loc(column.upper())
                old_name = data.columns[position]
                data = data.rename(columns={old_name: column})
            except KeyError:
                raise ValueError(
                    "The '%s' column could not be found in the point cloud." % column
                )

        return data

    elif usage == "optimizer":
        errmsg = ""
        title = "AN ERROR OCCURRED"
        ok_btn = "OK"
        if threshold == "Fraternity index":
            threshold_limits = data.loc[data["Parameter"] == "FI_threshold"]
        else:
            threshold_limits = data.loc[data["Parameter"] == "SI_threshold"]

        if data.isnull().values.any() or (
            data.astype(str).applymap(lambda x: x.strip() == "").values.any()
        ):
            errmsg += (
                "VALUE ERROR - Limits are missing."
                "\n\nPlease make sure all limits are defined in the 'param_limits.xlsx' file."
            )
            easygui.msgbox(errmsg, title, ok_btn)
            return None
        for index, row in data.iterrows():
            if row["Max"] < row["Min"]:
                errmsg += (
                    "VALUE ERROR - The upper limit must be greater than the lower limit."
                    "\n\nPlease correct the 'param_limits.xlsx' file and try again."
                )
                easygui.msgbox(errmsg, title, ok_btn)
                return None
        for index, row in threshold_limits.iterrows():
            if row["Max"] < 0 or row["Min"] < 0:
                errmsg += (
                    "VALUE ERROR - The threshold limits must be positive."
                    "\n\nPlease correct the 'param_limits.xlsx' file and try again."
                )
                easygui.msgbox(errmsg, title, ok_btn)
                return None

        return data
