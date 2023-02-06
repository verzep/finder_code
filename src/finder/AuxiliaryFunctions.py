#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:18:32 2021

@author: andreas
"""
import numpy as np
import pandas as pd

"""
JUST SOME UTILS
"""


def GetLineOfOptima(df, x_selector, y_selector, includeAllColumns=False):
    """
    For each unique value in `x_selector`, find the entry which maximizes 'y_selector" and return a pd.DataFrame
    containing the indexes with the best entry, the unique 'x_selector' values and the associated 'y_selector' values.

    This gives the *line of optima*, i.e., the optimal entry for each unique value of `x_selector`.



    Parameters
    ----------
    df: pd.DataFrame
        the dataframe with the data.
    x_selector: str
        the name of the  column you want to explore (usually, in FINDER, is `threshold`);
    y_selector: str
        the name of the column you want to use as metric (in FINDER, it is usually 'similarityscore');
    includeAllColumns: bool, default = False
        if False, just return the values of `y_selector` for each point. If True, return all columns.
    Returns
    -------
    df_opt = pd.DataFrame
        a DataFrame with columns ['idx', 'x_selector', 'y_selector'].
        If includeAllColumns== True, contains all the columns of `df` instead.
    """

    # unique entries of 'x_selector'
    x_sel_sort = np.sort(np.unique(df[x_selector]))
    idxs = []

    # for each `x_sel`, find the index leading to the maximum `y_selector` value.
    for x_sel in x_sel_sort:
        mark_ = df[x_selector] == x_sel
        if np.sum(mark_) == 0:
            continue
        else:
            idxs.append((df.loc[mark_, y_selector]).idxmax())

    df_opt = pd.DataFrame()
    df_opt["idx"] = idxs

    if includeAllColumns:
        cols = df.columns
    else:
        cols = [x_selector, y_selector]

    # generate the dataframe with the optima only
    for c in cols:
        df_opt[c] = np.asarray(df.loc[idxs, c])

    return df_opt


def GetClusterDistribution(labels):
    """
    Return an array containing the number of points with a given label.

    Position `i` in the array account for label 'i`. Its value will be the number of elements with label `i`.
    NOTE: the points that are considered noise (label == -1) are not included.

    Parameters
    ----------
    labels: the labels

    Returns
    -------

    """
    cl_sizes = []
    for c in np.unique(labels):
        if c == -1:
            continue
        cl_sizes.append(np.sum(labels == c))
    return cl_sizes


# Print iterations progress
def printProgressBar(
        iteration,
        total,
        prefix="",
        suffix="",
        decimals=1,
        length=100,
        fill="█",
        printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
