"""This module converts pandas dataframes.

This script requires that the following packages be installed within the Python
environment you are running this script in:
numpy (1.24.2), pandas (1.5.3), plyfile (0.7.4)

This file can be imported as a module and contains the following functions:
* df2array - converts a pandas DataFrame to a numpy structured array
* write_csv - writes a pandas or dask DataFrame to a .csv file
* write_ply - writes a pandas or dask DataFrame to a .ply file
* write_xlsx - writes a pandas or dask DataFrame to a .xlsx file

------------------------------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import plyfile


def pd2array(data):
    """Convert a pandas DataFrame object to a numpy structured array."""
    v = data.values
    cols = data.columns
    types = [(cols[i], data[k].dtype.type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    for i, k in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z


def write_csv(file, data):
    """This function converts a pandas dataframe to a csv file."""
    data.to_csv(file, encoding="utf-8", index=False)


def write_ply(file, data):
    """This function converts a pandas dataframe to a ply file."""
    for col in data.columns:
        if data[col].dtype == "int64":
            data[col] = data[col].astype(np.uintc)
        if data[col].dtype == "float64":
            data[col] = data[col].astype(np.double)
    if "." not in file:
        file += ".ply"
    if isinstance(data, pd.DataFrame):
        output_cloud = pd2array(data)
    else:
        raise TypeError("Unknown variable type: expected a pandas DataFrame")

    plyfile.PlyData(
        [
            plyfile.PlyElement.describe(
                output_cloud,
                "vertex",
            )
        ]
    ).write(file)


def write_xlsx(file, data):
    """This function converts a pandas dataframe to a xlsx file."""
    data.to_excel(file, index=None, header=True)
