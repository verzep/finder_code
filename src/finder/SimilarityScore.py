# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
from .AuxiliaryFunctions import printProgressBar

def GetSimilarityScorePerCluster(XC, phasespace, df_clusters=[]):
    if len(df_clusters) == 0:
        df_clusters = getClusterSizesAll(XC, phasespace)

    cli_similarityScore, similarityScore = getSimilarityScore(
        XC, phasespace, df_clusters
    )

    df_clusters["similarityScore"] = cli_similarityScore
    phasespace["similarityScore"] = similarityScore

    return (
        phasespace,
        df_clusters,
    )  # cli_similarityScore,similarityScore; #phaseSpace,df_clusters;


def getSimilarityScoreByThreshold(XC, PS, clusterInfo, n_dim):
    #
    t1 = time.time()
    # Get the info from clusterInfo
    cli_index = clusterInfo["index"]

    cli_similarityScore = np.zeros(
        [
            len(cli_index),
        ],
        dtype=int,
    )
    n = len(PS)
    similarityScoreMatrix = np.zeros(shape=(n, n))
    similarityScore = np.zeros(shape=(n,))
    ps_index = PS.index
    # 1 define a list of the unique threshold values
    threshold_list = np.unique(PS["threshold"])

    # centers,radii = __computeCenters_Radii(XC,PS);

    # Compute the similarities
    progress_i = 0
    printProgressBar(
        progress_i,
        len(PS),
        prefix="Postprocessing progress:",
        suffix="Complete",
        length=50,
    )
    for th in threshold_list:
        PS_th = PS[PS["threshold"] == th]

        for i, ps in PS_th.iterrows():
            mark_i = cli_index == i
            for j in PS_th.index[PS_th.index <= i]:  # np.arange(i+1):
                mark_j = cli_index == j
                if not (i == j):
                    s_ij = getSimilarityScore_ij(i, j, PS, clusterInfo)

                    if (type(s_ij) == bool) and (s_ij == False):
                        continue

                    score = np.sum(s_ij[0])

                    cli_similarityScore[mark_i] += s_ij[0]
                    cli_similarityScore[mark_j] += s_ij[1]

                    similarityScoreMatrix[
                        ps_index == j, ps_index == i
                    ] = score  # /Normalize here?  eg  /np.int(np.max(PS.loc[j,"labels"]) + 1)
                    similarityScoreMatrix[
                        ps_index == i, ps_index == j
                    ] = score  # /Normalize here?  eg  /np.int(np.max(PS.loc[i,"labels"]) + 1)
                # else:
                # we do not include similarity with self

            progress_i += 1
            printProgressBar(
                progress_i,
                len(PS),
                prefix="Progress:",
                suffix="Complete",
                length=50,
            )

    print(
        "Computing similarity scores: "
        + str(np.round(time.time() - t1, 2))
        + " seconds"
    )

    # ***************************
    # Collect data
    # ***************************
    for i, ps in PS.iterrows():
        similarityScore[ps_index == i] = np.sum(
            similarityScoreMatrix[ps_index == i, :]
        )

    return cli_similarityScore, similarityScore



def getSimilarityScore(XC, PS, clusterInfo, n_dim):
    """

    Compute similarity score for the whole phase-space.

    **steps**:
    1. Compute the similarities
    2. Collect data

    Parameters
    ----------
    XC:
        The data points (#TODO: not used, remove)
    PS:
        The phase space information

    clusterInfo: pd.DataFrame
        The information about the dataframe, as produced by `getClusterSizesAll`.

    Returns
    -------

    """

    t1 = time.time()
    # Get the info from clusterInfo
    cli_index = clusterInfo["index"]

    cli_similarityScore = np.zeros(
        [
            len(cli_index),
        ],
        dtype=int,
    )
    n = len(PS)
    similarityScoreMatrix = np.zeros(shape=(n, n))
    similarityScore = np.zeros(shape=(n,))
    ps_index = PS.index

    # centers,radii = __computeCenters_Radii(XC,PS);

    # Compute the similarities
    progress_i = 0
    printProgressBar(
        progress_i,
        len(PS),
        prefix="Postprocessing progress:",
        suffix="Complete",
        length=50,
    )
    for i, ps in PS.iterrows():
        mark_i = cli_index == i
        for j in PS.index[PS.index <= i]:  # np.arange(i+1):
            mark_j = cli_index == j

            if not (i == j):
                s_ij = getSimilarityScore_ij(i, j, PS, clusterInfo)

                if (type(s_ij) == bool) and (s_ij == False):
                    continue

                score = np.sum(s_ij[0])

                cli_similarityScore[mark_i] += s_ij[0]
                cli_similarityScore[mark_j] += s_ij[1]

                similarityScoreMatrix[
                    ps_index == j, ps_index == i
                ] = score  # /Normalize here?  eg  /np.int(np.max(PS.loc[j,"labels"]) + 1)
                similarityScoreMatrix[
                    ps_index == i, ps_index == j
                ] = score  # /Normalize here?  eg  /np.int(np.max(PS.loc[i,"labels"]) + 1)
            # else:
            # we do not include similarity with self

        progress_i += 1
        printProgressBar(
            progress_i,
            len(PS),
            prefix="Progress:",
            suffix="Complete",
            length=50,
        )
    print(
        "Computing similarity scores: "
        + str(np.round(time.time() - t1, 2))
        + " seconds"
    )

    # ***************************
    # Collect data
    # ***************************
    for i, ps in PS.iterrows():
        similarityScore[ps_index == i] = np.sum(
            similarityScoreMatrix[ps_index == i, :]
        )

    return cli_similarityScore, similarityScore


def getClusterSizesAll(XC, PS, n_dim:int):
    """
    Return a pd.DataFrame with the info about the clustering.
    To do so, compute centers and radii for the subsets in each cluster.
    The information is given by:

    * labels
    * clusterSize
    * threshold
    * sigma
    * index
    * centers
    * radii

    Parameters
    ----------
    XC:
        The points
    PS:
        The phase-space
    n_dim:
        problem dimensionality

    Returns
    -------
    A pd.DataFrame with the info about the clustering.
    """

    cl_sizes = np.array([], dtype=int)
    thresholds = np.array([], dtype=int)
    sigmas = np.array([])
    labels = np.array([], dtype=int)
    index = np.array([], dtype=int)
    centers = np.zeros((0, n_dim))
    radii = np.array([])

    for idx, df1_row in PS.iterrows():

        labels_i = df1_row["labels"]
        l_ = np.arange(np.max(labels_i) + 1)

        if l_.shape[0] == 0:
            continue

        labels = np.concatenate((labels, l_))
        cl_sizes = np.concatenate(
            (cl_sizes, np.asarray([np.sum(labels_i == l) for l in l_]))
        )
        thresholds = np.concatenate(
            (thresholds, df1_row["threshold"] * np.ones_like(l_))
        )
        sigmas = np.concatenate(
            (
                sigmas,
                df1_row["sigma"]
                * np.ones(
                    [
                        len(l_),
                    ]
                ),
            )
        )
        index = np.concatenate((index, idx * np.ones_like(l_)))

        centers_new, radii_new = __computeCenters_Radii_rowPS(XC, df1_row, n_dim)
        centers = np.concatenate((centers, centers_new))
        radii = np.concatenate((radii, radii_new))

    df_clusterSizes = pd.DataFrame()
    df_clusterSizes["labels"] = labels
    df_clusterSizes["clusterSize"] = cl_sizes
    df_clusterSizes["threshold"] = thresholds
    df_clusterSizes["sigma"] = sigmas
    df_clusterSizes["index"] = index
    df_clusterSizes["center"] = list(centers)
    df_clusterSizes["radius"] = radii

    return df_clusterSizes


def getSimilarityScore_ij(i, j, PS, clusterInfo):
    """
    Compute the similarity score between two subsets.

    Parameters
    ----------
    i: int
        first  index
    j: int
        second  index
    PS: pd.DataFrame
        the phase space
    clusterInfo

    Returns
    -------
    [s_i, s_j] : list
        The scores for the i and j component.

    """
    labels_1, labels_2 = PS.loc[i, "labels"], PS.loc[j, "labels"]

    #    clusterInfo_1 = clusterInfo.loc[clusterInfo['index']==i,:];
    #    clusterInfo_2 = clusterInfo.loc[clusterInfo['index']==j,:];
    centers_1 = clusterInfo.loc[clusterInfo["index"] == i, "center"].values
    centers_2 = clusterInfo.loc[clusterInfo["index"] == j, "center"].values

    radii_1 = clusterInfo.loc[clusterInfo["index"] == i, "radius"].values
    radii_2 = clusterInfo.loc[clusterInfo["index"] == j, "radius"].values

    # return zero score if no clusters selected or
    # if one cluster is selected which covers most points
    if (np.max(labels_1) == -1) or (
        np.max(labels_2) == -1
    ):  # cluster are noise
        return False
    elif (np.max(labels_1) == 0) and (
        np.sum(labels_1 == 0) / len(labels_1) > 0.5
    ):
        return False
    elif (np.max(labels_2) == 0) and (
        np.sum(labels_2 == 0) / len(labels_2) > 0.5
    ):
        return False

    # ******************************
    similarityMatrix = np.zeros(
        (np.max(labels_1) + 1, np.max(labels_2) + 1), dtype=int
    )
    similarityMatrix[:] = -1
    # ******************************

    for i1 in np.unique(labels_1):
        if i1 == -1:  # exclude the noise cluster
            continue
        for i2 in np.unique(labels_2):
            if i2 == -1:  # exclude  the noise cluster
                continue

            if similarityMatrix[i1, i2] == 0:
                continue

            if __No_OverlapClusters_Distance(
                i1, i2, centers_1, centers_2, radii_1, radii_2
            ):
                similarityMatrix[i1, i2] = 0
                continue

            if __OverlapClusters_NumberOfLocs(i1, i2, labels_1, labels_2):
                similarityMatrix[i1, :] = 0
                similarityMatrix[:, i2] = 0
                similarityMatrix[i1, i2] = 1
                break
            else:
                similarityMatrix[i1, i2] = 0

    similarityMatrix[similarityMatrix == -1] = 0

    s_i = np.sum(similarityMatrix, axis=1)
    s_j = np.sum(similarityMatrix, axis=0)
    return [s_i, s_j]


def __No_OverlapClusters_Distance(
    i1, i2, centers_1, centers_2, radii_1, radii_2
):
    """

    Return true if the two subsets **do not** overlap.
    This is done by verifying that the distance between their centers is larger than the sum of their radii.

    Parameters
    ----------
    i1: int
        index of the first subset
    i2: int
        index of the second subset
    centers_1:
        list of coordinates of the centers of the first cluster
    centers_2:
        list of coordinates of the centers of the second cluster
    radii_1:
        list of radii of the first cluster
    radii_2:
        list of radii of the second cluster

    Returns
    -------
    bool
    Return True if subset `i1` and subset `i2` do not overlap.
    """
    c1, c2 = centers_1[i1], centers_2[i2]
    r1, r2 = radii_1[i1], radii_2[i2]
    return np.linalg.norm(c2 - c1) > r1 + r2


def __OverlapClusters_NumberOfLocs(i1, i2, labels_1, labels_2):
    """

    Given two labels 'i1' and 'i2' (i.e., two clusters) check if the number of overlapping points among the two is larger
    than the number of non-overlapping points.

    Parameters
    ----------
    i1: int
        label of the first subset
    i2: int
        label of the second subset
    labels_1:
        list of all labels of the first cluster
    labels_2:
        list of all labels of the second cluster
    Returns
    -------
    bool
        True if the number of overlapping points is larger than the number of non-overlapping points.
    """
    # compute the number of element of the first cluster with label == `i1`
    no_locs_1 = np.sum(labels_1 == i1)
    # compute the number of element of the second cluster with label == `i2`
    no_locs_2 = np.sum(labels_2 == i2)
    # compute the overlap between the two clusters (i.e., the number of matching points)
    no_locs_overlap = np.sum((labels_1 == i1) * (labels_2 == i2))

    # return true if the majority of points overlaps.
    if (no_locs_overlap / no_locs_1 > 0.5) and (
        no_locs_overlap / no_locs_2 > 0.5
    ):
        return True
    else:
        return False


def __computeCenters_Radii_rowPS(XC, ps, n_dim):
    """
    For each parameter configuration in a single row of the dataset, compute the center and the radius of each cluster.

    * `centers`:
    a list of arrays each of dimension (`no_clusters`,2),
    storing the coordinate of the center of each cluster
    Note that `no_clusters` varies for each element of the list.

    * 'radii':
    a list of arrays of size ('no_clusters').
    Contains the radius of each cluster, defined as the distance of the farthest point from the center.
    Note that `no_clusters` varies for each element of the list.


        Parameters
        ----------
        XC:
            the datapoints.
        ps:
            a row of the phaspace.
        n_dim:
            problem dimensionality

        Returns
        -------
        centers, radii
    """

    no_clusters = np.max(ps["labels"]) + 1

    centers_i = np.zeros((no_clusters, n_dim))
    radii_i = np.zeros((no_clusters,))

    # go through all clusters:
    for icl in np.unique(ps["labels"]):
        if icl == -1:
            continue
        XCm = XC[(ps["labels"] == icl)]

        c = np.median(XCm, axis=0)
        centers_i[icl, :] = c
        radii_i[icl] = np.max(np.linalg.norm(XCm - c, axis=1))

    return centers_i, radii_i


def __computeCenters_Radii(XC, PS):
    """
    #TODO: this is not used!

    For each parameter configuration, compute the center and the radius of each cluster.

    * `centers` :

    a list of arrays each of dimension (`no_clusters`,2),
    storing the coordinate of the center of each cluster
    Note that `no_clusters` varies for each element of the list.

    * `radii` :

    a list of arrays of size ('no_clusters').
    Contains the radius of each cluster, defined as the distance of the farthest point from the center.
    Note that `no_clusters` varies for each element of the list.


        Parameters
        ----------
        XC:
            the datapoints
        PS:
            the phase space

        Returns
        -------
        centers, radii
    """

    centers = []
    radii = []

    # loop through the data
    for i, ps in PS.iterrows():
        # compute the number of labels for that particular configuration of parameters.
        no_clusters = np.max(ps["labels"]) + 1
        # initialize the centers and the radii for the configuration
        centers_i = np.zeros((no_clusters, 2))
        radii_i = np.zeros((no_clusters,))

        # go through all clusters:
        for icl in np.unique(ps["labels"]):
            if icl == -1:  # ignore noise
                continue
            # take all the point  with label icl
            XCm = XC[(ps["labels"] == icl)]
            # compute the median
            c = np.median(XCm, axis=0)
            # store the center position in the icl position
            centers_i[icl, :] = c
            # store the radius of the cluster, computed as the distance of the farthest point from the center.
            radii_i[icl] = np.max(np.linalg.norm(XCm - c, axis=1))

        centers.append(centers_i)
        radii.append(radii_i)

    return centers, radii
