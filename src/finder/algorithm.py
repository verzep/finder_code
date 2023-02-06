import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .AuxiliaryFunctions import GetLineOfOptima, printProgressBar
from .DbscanLoop import DbscanLoop

from .SimilarityScore import (
    getClusterSizesAll,
    getSimilarityScore,
    getSimilarityScoreByThreshold,
)


class Finder:
    def __init__(
            self,
            threshold: int = 10,
            points_per_dimension=15,
            algo="DbscanLoop",
            minmax_threshold=[5, 21],
            one_two_d="twoD",
            similarity_score_computation="threshold",
            log_thresholds=False,
            log_sigmas=True,
            adaptive_sigma_boundaries=False,
            decay=0.5,
            n_dim: int = None
    ):

        """
        The model class for the `FINDER` algorithm. It uses the sklearn API.
        It explores the two parameters (`threshold` and `sigma`) of the given `algo` and finds the best combination.
        Note that in the paper (and sometimes in the code) the parameters are
        called 'minPts'(=threshold) and 'epsilon'(=sigma) instead.

        Parameters
        ----------
        threshold: int, default = 10
            The default value for the `threshold` value, also called `minPts`.
        points_per_dimension:
            the number of values for each of the two axis  (`threshold` and `sigma`) of the phase space.
        algo: str, default="DbscanLoop"
            the algorithm to be used. Can be either "dbscan" or "Dbscanloop" todo: fix the capital D
        minmax_threshold: list, default = [3,30],
            The minimum and maximum values for the "threshold" parameter.
        one_two_d: str, default = "twoD"
            A string to define the kind of optimization that one wish to carry.
            Possible values are ["oneD", "oneD_thresholds", "twoD"]
        similarity_score_computation: str, default = 'total'
            Select the way in which the similarity score is computed:

            * `total`: compare each parameter combination with any other.
                The number of comparison will then be (p**2)*(p**2 - 1)/2 = O(p**4),
                where p is `points_per_dimension`.
            * 'threshold: compute the similarity only between parameter-combination sharing the same threshold
                (i.e., line by line).
                The number of comparison will then be p*(p)*(p- 1)/2 = O(p**3),
            Applies to the the "twoD" setting only
        log_thresholds: Bool, default = 'False'
            If `True`, the threshold-values are selected in log scale.
        log_sigmas: Bool, default = 'True'
            If `True`, the sigma-values are selected in log scale.
        decay: float, default = 0.5
            The value used to select the selected parameters.
        n_dim: int, default = None
            The dimensionality of your data, presumably 2D or 3D. If `None`, it will be inferred using the data.

        """

        self.threshold = np.int(threshold)
        self.no_points_sigma = points_per_dimension
        self.no_points_thresholds = points_per_dimension
        self.algo = algo
        self.minmax_threshold = minmax_threshold
        assert one_two_d in [
            "oneD",
            "oneD_thresholds",
            "twoD",
        ], 'Possible values are ["oneD", "oneD_thresholds", "twoD"]'
        self.one_two_d = one_two_d
        self.decay = decay
        assert similarity_score_computation in [
            "total",
            "threshold",
        ], 'possible values are ["total", "threshold"]'
        self.similarity_score_computation = similarity_score_computation
        self.log_thresholds = log_thresholds
        self.log_sigmas = log_sigmas
        self.adaptive_sigma_boundaries = adaptive_sigma_boundaries
        self.n_dim = n_dim

    def fit(self, XC, XC_params=None, **kwargs):
        """
        Generate the labels using FINDER algorithm.

        **Steps**:
         1. Set Boundaries: Get min max of threshold and sigma
         2. Clustering: Compute phase spaces
         3. Postprocessing: Compute similarity score (can be skipped)

        Parameters
        ----------
        XC:
            Points to be clustered
        XC_params:
            The points used to compute the boundaries. If `Null`, `XC` will be used.
        kwargs

        Returns
        -------

        clustering labels

        """
        print("Analysing " + str(len(XC)) + " points")

        if XC_params is None:
            XC_params = XC

        if self.n_dim is None:
            self.n_dim = np.array(XC).shape[1]

        t_1 = time.time()

        # Step 1: Get min max of threshold and sigma
        if self.one_two_d == "oneD":
            params = self.__getParams_Sigmas(XC_params)
        if self.one_two_d == "oneD_thresholds":
            params = self.__getParams_Thresholds(XC_params)
        elif self.one_two_d == "twoD":
            params = self.__getParams_SigmasThresholds(XC_params)

        t_2 = time.time()

        # Step 2: Compute phase spaces
        phasespace = self.__phaseSpace(XC, params)
        self.phasespace = phasespace
        t_3 = time.time()

        # Step 3: Compute similarity score
        if ("skipSimilarityScore" in kwargs.keys()) and (
                kwargs["skipSimilarityScore"] == True
        ):
            print("skipping the similarity-score computation")
            data = self.__phaseSpacePostProcess(
                XC, phasespace, skipSimilarityScore=True
            )

            selth = self.phasespace.loc[
                np.argmin(
                    np.abs(self.phasespace["threshold"] - self.threshold)
                ),
                "threshold",
            ]
            mark = self.phasespace["threshold"] == selth
            selected_parameters = self.phasespace.loc[
                self.phasespace.loc[mark, "no_clusters"].idxmax()
            ]
            labels = selected_parameters["labels"]

        else:
            data = self.__phaseSpacePostProcess(XC, phasespace)
            # Step 3: Get parameterset
            if self.one_two_d == "twoD":
                labels, selected_parameters = self.__get_consensus_clustering(
                    data, XC
                )
            else:
                print("USING 1D OPTIMIZER")
                (
                    labels,
                    selected_parameters,
                ) = self.__get_consensus_clustering_1d(data, XC)

        t_4 = time.time()

        # labels,selected_parameters = self.__get_consensus_clustering2(data,XC);
        print(
            "Comp time Step 1 (set boundaries): "
            + str(np.round(t_2 - t_1, 2))
            + " seconds"
        )
        print(
            "Comp time Step 2 (clustering): "
            + str(np.round(t_3 - t_2, 2))
            + " seconds"
        )
        print(
            "Comp time Step 3 (postprocessing): "
            + str(np.round(t_4 - t_3, 2))
            + " seconds"
        )
        print("Selected parameters: " + str(selected_parameters))

        # Save data
        self.computationTimes = {
            "Step1": t_2 - t_1,
            "Step2": t_3 - t_2,
            "Step3": t_4 - t_3,
        }
        self.data = data
        self.labels = labels
        self.selected_parameters = selected_parameters

        return labels

    def ComputeClusters(self, sigma, threshold, XC):
        """

        Cluster the point using the selected `self.algo` (DBSCAN or Dbscanloop).
        This is basically a iteration of `self.algo` in a given phase-space point.

        Parameters
        ----------
        sigma:
            radius.
        threshold:
            number of point inside the radius.
        XC:
            data-points.

        Returns
        -------

        The labels

        """
        threshold = int(threshold)

        if self.algo == "dbscan":
            DB = DBSCAN(eps=sigma, min_samples=threshold).fit(XC)
            labels_ = DB.labels_

        elif self.algo == "DbscanLoop":
            DBL = DbscanLoop(eps=sigma, min_samples=threshold).fit(XC)
            labels_ = DBL.labels_

        else:
            print("ALGORITHM NOT RECOGNIZED !!")

        return labels_

    def __getParams_SigmasThresholds(self, XC):

        """
        Compute the parameter range for both `threshold` and `sigma`.

        Parameters
        ----------
        XC: the datapoints

        Returns
        -------
        pd.Dataframe
        """

        if self.adaptive_sigma_boundaries:
            minmax_sigma = self.__determine_sigma_boundaries_adaptive(XC)
        else:
            minmax_sigma = self.__determine_sigma_boundaries(XC)

        if self.log_sigmas:
            sigmas = self.__getLogDistribution(
                minmax_sigma[0], minmax_sigma[1], self.no_points_sigma
            )
        else:
            sigmas = np.linspace(
                minmax_sigma[0], minmax_sigma[1], self.no_points_sigma
            )

        if self.log_thresholds:
            thresholds = np.linspace(
                self.minmax_threshold[0],
                self.minmax_threshold[1],
                self.no_points_sigma,
            )
            thresholds = np.unique(np.round(thresholds))
        else:
            thresholds = np.arange(
                self.minmax_threshold[0], self.minmax_threshold[1]
            )

        # thresholds = self.__getLogDistribution(self.minmax_threshold[0],
        # self.minmax_threshold[1],
        # self.no_points_thresholds)

        # #LINEAR
        # #sigmas = np.linspace(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma)
        # #thresholds = np.linspace(self.minmax_threshold[0], self.minmax_threshold[1], self.no_points_sigma)
        #
        # #thresholds = np.unique(np.round(thresholds))
        #
        # thresholds = np.arange(self.minmax_threshold[0], self.minmax_threshold[1])
        # print("done")

        s_all = []
        t_all = []
        for s in sigmas:
            for t in thresholds:
                s_all.append(s)
                t_all.append(t)

        print("Sigmas are:")
        print(sigmas)
        print("Thresholds are:")
        print(thresholds)

        params = pd.DataFrame(
            data={"sigma": np.asarray(s_all), "threshold": np.asarray(t_all)}
        )
        return params

    def __getParams_Sigmas(self, XC):

        """
        Compute the parameter range for `sigma` for a fixed `threshold` value.
        The value for `threshold` is 'self.threshold'.

        Parameters
        ----------
        XC: the datapoints

        Returns
        -------
        params:pd.Dataframe
            the values of the parameters that FINDER will explore.
        """

        if self.adaptive_sigma_boundaries:
            minmax_sigma = self.__determine_sigma_boundaries_adaptive(XC)
        else:
            minmax_sigma = self.__determine_sigma_boundaries(XC)

        if self.log_sigmas:
            sigmas = self.__getLogDistribution(
                minmax_sigma[0], minmax_sigma[1], self.no_points_sigma
            )
        else:
            sigmas = np.linspace(
                minmax_sigma[0], minmax_sigma[1], self.no_points_sigma
            )

        params = pd.DataFrame(
            data={
                "sigma": sigmas,
                "threshold": self.threshold * np.ones_like(sigmas),
            }
        )
        return params

    def __getParams_Thresholds(self, XC):

        """
        Compute the parameter range for `threshold` for a fixed `sigma` value.
        The value for `sigma` is computed using a KNN.

        Parameters
        ----------
        XC: the datapoints

        Returns
        -------
        params: pd.Dataframe
            the values of the parameters that FINDER will explore.
        """

        k = 10
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC)

        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9.
        # This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        # CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma = np.quantile(nPt_distance, 0.5)

        if self.log_thresholds:
            thresholds = np.linspace(
                self.minmax_threshold[0],
                self.minmax_threshold[1],
                self.no_points_sigma,
            )
            thresholds = np.unique(np.round(thresholds))
        else:
            thresholds = np.arange(
                self.minmax_threshold[0], self.minmax_threshold[1]
            )

        params = pd.DataFrame(
            data={
                "sigma": sigma * np.ones_like(thresholds),
                "threshold": thresholds,
            }
        )
        return params

    def __determine_sigma_boundaries_adaptive(self, XC):
        """

        Parameters
        ----------
        XC

        Returns
        -------

        """
        k = self.minmax_threshold[0] + 1  # k         = self.threshold+1;
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC)
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]
        sigma_min = np.quantile(nPt_distance, 0.1)

        k = self.minmax_threshold[1] + 1  # k         = self.threshold+1;
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC)
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]
        sigma_max = np.quantile(nPt_distance, 0.9)

        minmax_sigma = [sigma_min, sigma_max]

        print(
            "Boundaries for sigma    : "
            + str(minmax_sigma[0])
            + " , "
            + str(minmax_sigma[1])
        )

        return minmax_sigma

    def __determine_sigma_boundaries(self, XC):

        """
        OLDER VERSION, THE NEW ONE IS COMMENTED

        Parameters
        ----------
        XC

        Returns
        -------

        """

        k = self.threshold + 1
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC)
        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        # CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma_min = np.quantile(nPt_distance, 0.1)
        sigma_max = np.quantile(nPt_distance, 0.9)
        minmax_sigma = [sigma_min, sigma_max]

        print(
            "Boundaries for sigma    : "
            + str(minmax_sigma[0])
            + " , "
            + str(minmax_sigma[1])
        )

        return minmax_sigma

    def __getLogDistribution(self, min_x, max_x, n):

        """
        Return an array with the log-distribution of 'n' points between
        `min_x` and `max_x`.
        """
        min_log = np.log(min_x)
        max_log = np.log(max_x)

        log_vec = np.linspace(min_log, max_log, n)

        vec = np.exp(log_vec)
        vec = np.unique(vec)  # why is this needed?

        return vec

    def __phaseSpace(self, XC, params):
        """

        Given a set of point to be clustered,
        compute the cluster for each combination of parameters.
        This returns the phase-space,
        which is the set of labels associated with each parameter combination.
        Also computational times are included.

        Parameters
        ----------
        XC:
            data-points
        params:
            list of parameters to compute the phasespace
        Returns
        -------
        ps: pd.Dataframe
        the phase space

        """

        labels_all = []
        times = []

        t1_all = time.time()

        printProgressBar(
            0,
            len(params),
            prefix="Clustering progress:",
            suffix="Complete",
            length=50,
        )

        for index, param in params.iterrows():
            t1 = time.time()
            labels_ = self.ComputeClusters(
                param["sigma"], param["threshold"], XC
            )
            t2 = time.time()
            labels_all.append(labels_)
            times.append(t2 - t1)
            printProgressBar(
                index + 1,
                len(params),
                prefix="Progress:",
                suffix="Complete",
                length=50,
            )
            # print("Computing time for sigma = "+str(np.round(param['sigma'],2))+" and minPts ="+ str(param['threshold'])+" : " + str(np.round(t2-t1,2)) );

        print(
            "Computing clusters : "
            + str(np.round(time.time() - t1_all, 2))
            + " seconds"
        )
        ps = params
        ps["labels"] = labels_all
        ps["time"] = times

        return ps

    def __phaseSpacePostProcess(
            self, XC, PS, skipSimilarityScore: bool = False
    ):

        """
        Postprocess the phasespace, computing the similarities between the clusters.

        **steps**:

        1. Initialize Cluster information
        2. Compute similarity scores (optional)

        Parameters
        ----------
        XC:
            The points to be clustered.
        PS: pd.DataFrame
            The phase-space, i.e., the set of labels for each parameter configuration.
        skipSimilarityScore: bool, default=False
            if 'True', avoid computing the similarities.

        Returns
        -------
        PS: pd.DataFrame
            the phasespace with the postporcessing information.

        """

        print("Postprocessing..")

        n = len(PS)

        no_clusters = np.zeros(shape=(n,), dtype=np.int)
        similarityScore = np.zeros(shape=(n,))
        # no_locs                  = np.zeros(shape=(n,),dtype=np.int)
        times = np.zeros(shape=(n,))

        # similarityScoreMatrix    = np.zeros(shape=(n,n))

        # ***********************************************
        # Preprocess: Initialize Cluster information
        # ***********************************************
        clusterInfo = getClusterSizesAll(XC, PS, self.n_dim)
        cli_index = clusterInfo["index"]
        cli_similarityScore = np.zeros(
            [
                len(cli_index),
            ],
            dtype=int,
        )

        for i, ps in PS.iterrows():
            no_clusters[i] = np.int(np.max(ps["labels"]) + 1)
        PS["no_clusters"] = no_clusters

        # ***************************
        # Compute similarity scores
        # ***************************
        ###

        if skipSimilarityScore == True:
            for i, ps in PS.iterrows():
                similarityScore[i] = np.nan
        else:
            # ***************************************
            # Preprocess: get centers and radii
            # ***************************************

            if self.similarity_score_computation == "total":
                cli_similarityScore, similarityScore = getSimilarityScore(XC, PS, clusterInfo, self.n_dim)
            if self.similarity_score_computation == "threshold":
                (cli_similarityScore, similarityScore) = getSimilarityScoreByThreshold(XC, PS, clusterInfo, self.n_dim)

        clusterInfo["similarityScore"] = cli_similarityScore
        PS["similarityScore"] = similarityScore

        self.clusterInfo = clusterInfo

        return PS

    def __get_consensus_clustering_1d(self, PS, XC):
        """
        Find the best clustering parameters, defined as the one with the highest `similarityScore`.

        This is supposed to be used when optimizing only one parameter (`sigma` or `threshold`).

        Parameters
        ----------
        PS: pd.DataFrame
            the phasepace
        XC:
            The points (#todo: not used, remove)
        Returns
        -------
        labels, selected_parameters:
            the labels of the chosen  configuration and
            the parameters of the chosen configuration
        """
        # take the similarity scores
        similarity = np.asarray(PS["similarityScore"])
        # find the maximum
        max_score = np.max(similarity)
        # take its index
        idx = np.argwhere(similarity == max_score)[-1][0]
        # store the sigma
        sigma_selected = PS.loc[idx, "sigma"]
        # store the threshold
        threshold_selected = PS.loc[idx, "threshold"]
        # store the labels
        labels = PS.loc[idx, "labels"]

        # store the selected parameters
        selected_parameters = {
            "sigma": sigma_selected,
            "threshold": threshold_selected,
        }

        print(
            "Selected threshold , sigma : "
            + str(threshold_selected)
            + " , "
            + str(sigma_selected)
        )
        return labels, selected_parameters

    def __get_consensus_clustering(self, PS, XC, decay=None):
        """
        Find the best parameter configuration fot the 2-d setting, i.e., when optimizing both 'sigma' and 'threshold'.

        **steps**:

        1. Get optimal sigma for each theta
        2. Normalize the line of optima
        3. choose the theta for which the similarity score is < decay

        Parameters
        ----------
        PS: pd.DataFrame
            the phasepace
        XC:
            The points (#todo: not used, remove)
        decay:
            The threshold value for selecting theta among the optima. Must be in (0,1).
        Returns
        -------
        labels, selected_parameters:
            the labels of the chosen  configuration and
            the parameters of the chosen configuration
        """
        if decay is None:
            decay = self.decay
        # 1. Get optimal sigma for each theta
        df_opt_th = GetLineOfOptima(PS, "threshold", "similarityScore")

        line_of_optima = df_opt_th["idx"]
        line_of_optima_sim = np.array(df_opt_th["similarityScore"])
        self.df_opt_th = df_opt_th
        self.line_of_optima = line_of_optima
        # 2. Normalize the diagonal
        opt_normalized = (line_of_optima_sim - line_of_optima_sim.min()) / (
                line_of_optima_sim.max() - line_of_optima_sim.min()
        )

        # 3. choose the theta for which the similarity score is < decay
        ind = np.where(opt_normalized < decay)[0][0]
        optim = PS.iloc[line_of_optima[ind]]

        labels = optim["labels"]
        selected_parameters = {
            "sigma": optim["sigma"],
            "threshold": optim["threshold"],
        }

        print(
            "Selected threshold , sigma : "
            + str(optim["threshold"])
            + " , "
            + str(optim["sigma"])
        )
        return labels, selected_parameters

    def plotPhaseSpace(self, ax=None):

        sigmas = np.unique(self.phasespace["sigma"])
        thresholds = np.unique(self.phasespace["threshold"])

        sigma_opt = self.selected_parameters["sigma"]
        threshold_opt = self.selected_parameters["threshold"]

        sigma_opt_idx = np.where(sigmas == sigma_opt)[0][0]
        threshold_opt_idx = np.where(thresholds == threshold_opt)[0][0]

        index_opt = np.where(sigmas == sigma_opt)[0][0]

        # Compute similarity Matrix
        similarity = []
        for i, row in self.phasespace.iterrows():
            similarity.append(row["similarityScore"])
        sim_matr = np.round(
            np.flipud(np.array(similarity).reshape(15, -1).T), 2
        )

        # Compute line of optima (max_list) for it
        max_list = []
        for t, s in enumerate(sim_matr.argmax(1)):
            max_list.append(s)
        max_list = max_list[::-1]

        # phase space

        if ax is None:
            fig, ax = plt.subplots()

        sns.heatmap(
            sim_matr,
            xticklabels=np.round(sigmas, 2),
            yticklabels=np.flipud(np.round(thresholds)),
            ax=ax,
            cbar=False,
            cmap="Reds",
        )
        from matplotlib.patches import Rectangle

        for i, j in enumerate(max_list[::-1]):
            ax.add_patch(
                Rectangle((j, i), 1, 1, fill=False, edgecolor="blue", lw=3)
            )
        ax.add_patch(
            Rectangle(
                (sigma_opt_idx, len(thresholds) - 1 - threshold_opt_idx),
                1,
                1,
                fill=False,
                edgecolor="green",
                lw=3,
            )
        )
        # sns.lineplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        # sns.scatterplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        # ax.axvline(sigma_min,c='r');
        # ax.axvline(sigma_max,c='r');
        # ax.axvline(sigma_opt,c='g');
        # #plt.annotate('Selected value', (sigma_opt,FD.phasespace['similarityScore'][index_opt]))
        # trans = ax.get_xaxis_transform()
        # plt.text(sigma_opt, .5, 'Selected value', transform=trans,rotation=90)
        ax.set_xlabel("eps (nm)")
        ax.set_ylabel("minPts")

        plt.show()
