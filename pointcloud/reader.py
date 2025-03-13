"""This module reads data of different formats and returns pandas dataframe.

It requires that the following packages be installed within the Python
environment you are running this script in: pandas (1.5.3), plyfile (0.7.4)

This script uses 'utility' module from local 'pointcloud' package.

This file can be imported as a module and contains the following functions:
* read_ply - reads a ply file and returns a pandas dataframe
* read_csv - reads a csv file and returns a pandas dataframe
* read_xlsx - reads a xlsx file and returns a pandas dataframe
"""

import pandas as pd
import plyfile


def read_csv(file, sep=","):
    """Reads a csv file and returns a pandas dataframe."""
    data = pd.read_csv(file, encoding="utf-8", sep=sep)
    return data


def read_ply(file):
    """Reads a ply file and returns a pandas dataframe."""
    plydata = plyfile.PlyData.read(file)
    data = pd.DataFrame(plydata.elements[0].data)
    return data


def read_xlsx(file, sheet_name=0):
    """Reads a xlsx file and returns a pandas dataframe."""
    data = pd.read_excel(file, sheet_name=sheet_name)
    return data
