from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from utils import save, scale_features


class clustering_methods():
    """Useful object to manage cluster methods. Inside, the various models fitted to the data are saved.
    Each function that creates clustering models saves the return values 
    ​​in csv files if the must_save flag is equal to True
    """
    def __init__(self):
        self.km = None
        self.fcm = None
        self.ac = None
        self.db = None
        self.must_save = True
        self.sc_km = None
        self.sc_fcm = None
        self.sc_ac = None
        self.sc_db = None


    def kmenas_cluster(self, X, n_clusters : int =17, must_save : bool =True):
        """This function performs kmeans clustering.
        Before clustering, budget the data with the scale_features function
        which in turn calls the sklearn StandardScale () function

        Args:
            X (_type_): Data on which to do kmeans clustering
            n_clusters (int, optional): Defaults to 16.
            must_save (bool, optional): If it is True then the function outputs are also saved in csv files.
                                        Defaults to True.

            Furthermore, the function saves the kmeans model within itself.

        Returns:
            _type_: Data labels after clustering (kmeans.predict)
            _type_: Coordinates of cluster centers.
                    If the algorithm stops before fully converging (see tol and max_iter),
                    these will not be consistent with labels_.
            _type_: Compute clustering and transform data to cluster-distance space.
        """
        self.km = None
        self.sc_km = None
        X_std, self.sc_km = scale_features(X)
        km = KMeans(n_clusters=n_clusters)
        all_distances = km.fit_transform(X_std)
        # What kmeans.transform(X) already returns is the distance of the norm L2 from each center of the cluster
        y_km = km.predict(X_std)
        centers_cluster = km.cluster_centers_
        if must_save:
            save(y_km, 'kmenas_cluster_y_predicted')
            save(centers_cluster, 'kmenas_cluster_centers_cluster')
            save(all_distances, 'kmenas_cluster_all_distances')
        self.km = km
        return y_km, centers_cluster, all_distances

    def kmenas_predict(self, X):
        if self.sc_km or self.km is not None:
            X_std = self.sc_km.transform(X)
            y_predicted = self.km.predict(X_std)
            return y_predicted
        else: print("not find any k-means model in the object")

    def fuzzy_cluster(self, X, n_clusters : int =17, must_save : bool =True):
        """This function performs fuzzy-c-means clustering.
        Before clustering, budget the data with the scale_features function
        which in turn calls the sklearn StandardScale () function

        Furthermore, the function saves the fuzzy-c-means model within itself.

        Args:
            X (_type_): Data on which to do fuzzy-c-means clustering
            n_clusters (int, optional): Defaults to 16.
            must_save (bool, optional): If it is True then the function outputs are also saved in csv files.
                                        Defaults to True.

        Returns:
            _type_: Fuzzy predictions, a single lable is assigned to each data
            _type_: Soft fuzzy predictions.
                    An array of probabilities of belonging to each cluster is assigned to the data.
            _type_: Returns the center of each cluster.
        """
        self.fcm = None
        self.sc_fcm = None
        X_std, self.sc_fcm = scale_features(X)
        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(X_std)
        y_fcm_hard = fcm.predict(X_std)
        y_fcm_soft = fcm.soft_predict(X_std)
        fcm_centers = fcm.centers
        if must_save:
            save(y_fcm_hard, 'fuzzy_cluster_y_hard')
            save(y_fcm_soft, 'fuzzy_cluster_y_soft')
        self.fcm = fcm
        return y_fcm_hard, y_fcm_soft, fcm_centers

    def fuzzy_predict(self, X):
        if self.sc_fcm or self.fcm is not None:
            X_std = self.sc_fcm.transform(X)
            y_predicted_hard = self.fcm.predict(X_std)
            y_predicted_soft = self.fcm.soft_predict(X_std)
            return y_predicted_hard, y_predicted_soft
        else: print("not find any fuzzy-c-means model in the object")

    def agglomerative_cluster(self, X, n_clusters : int =16, affinity : str ='euclidean', linkage : str ='complete', must_save : bool =True):
        """It is a "bottom up" approach (from bottom to top) in which we start
        by inserting each element in a different cluster and then proceed to the gradual unification
        of clusters two by two.

        Furthermore, the function saves the agglomearative model within itself.

        Args:
            X (_type_): Data on which to do agglomerative clustering
            n_clusters (int, optional): Defaults to 16.
            affinity (str, optional): Metric used to compute the linkage.
                                    Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
                                    If linkage is “ward”, only “euclidean” is accepted.
                                    If “precomputed”, a distance matrix (instead of a similarity matrix) is needed
                                     as input for the fit method. Defaults to 'euclidean'.
            linkage (str, optional): {‘ward’, ‘complete’, ‘average’, ‘single’}
                                    Which linkage criterion to use. The linkage criterion determines which
                                    distance to use between sets of observation.
                                    The algorithm will merge the pairs of cluster that minimize this criterion.
                                    Defaults to 'complete'.
            must_save (bool, optional): If it is True then the function outputs are also saved in csv files.
                                        Defaults to True.

        Returns:
            _type_: Return the leables of any data after clustering.
        """
        self.ac = None
        self.sc_ac = None
        X_std, self.sc_ac = scale_features(X)
        ac = AgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        y_ac = ac.fit_predict(X_std)
        if must_save:
            save(y_ac, 'agglomerative_cluster_labels')
        self.ac = ac
        return y_ac

    def agglomerative_predict(self, X):
        if self.sc_ac or self.ac is not None:
            X_std = self.sc_ac.transform(X)
            y_predicted = self.ac.predict(X_std)
            return y_predicted
        else: print("not find any agglomerative model in the object")

    def dbscan_cluster(self, X, eps=0.2, min_samples : int =5, metric : str='euclidean', must_save : bool=True):
        """DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
        Finds core samples of high density and expands clusters from them. Good for data which contains
        clusters of similar density.

        Furthermore, the function saves the dbscan model within itself.

        Args:
            X (_type_): Data on which to do agglomerative clustering
            eps (float, optional): The maximum distance between two samples for one to be considered as
                                in the neighborhood of the other. This is not a maximum bound on the distances
                                of points within a cluster. This is the most important DBSCAN parameter to choose
                                appropriately for your data set and distance function.
                                Defaults to 0.2.
            min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point
                                        to be considered as a core point. This includes the point itself.
                                        Defaults to 5.
            metric (str, optional): The metric to use when calculating distance between instances in a feature array.
                                    If metric is a string or callable, it must be one of the options allowed by
                                    sklearn.metrics.pairwise_distances for its metric parameter.
                                    If metric is “precomputed”, X is assumed to be a distance matrix and must be
                                    square. X may be a Glossary, in which case only “nonzero” elements may be
                                    considered neighbors for DBSCAN.
                                    Defaults to 'euclidean'.
            must_save (bool, optional): If it is True then the function outputs are also saved in csv files.
                                        Defaults to True.

        Returns:
            _type_: Return the leables of any data after clustering.
        """
        self.db = None
        self.sc_db = None
        X_std, self.sc_db = scale_features(X)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        y_db = db.fit_predict(X_std)
        if must_save:
            save(y_db, 'dbscan_cluster')
        self.db = db
        return y_db

    def dbscan_predict(self, X):
        if self.sc_db or self.db is not None:
            X_std = self.sc_db.transform(X)
            y_predicted = self.db.predict(X_std)
            return y_predicted
        else: print("not find any dbscan model in the object")

    def reset(self):
        """Function that is used to bring all the variables of the object to the initial settings.
        """
        self.km = None
        self.fcm = None
        self.ac = None
        self.db = None
        self.must_save = True
        self.sc_km = None
        self.sc_fcm = None
        self.sc_ac = None
        self.sc_db = None

    def get_sc_km(self):
        return self.sc_km

    def get_sc_fcm(self):
        return self.sc_fcm

    def get_sc_ac(self):
        return self.sc_ac

    def get_sc_db(self):
        return self.sc_db

    def get_km(self):
        return self.km

    def get_fcm(self):
        return self.fcm

    def get_ac(self):
        return self.ac

    def get_db(self):
        return self.db

    def get_must_save(self):
        return self.must_save

    def set_must_save(self, must_save: bool):
        self.must_save = must_save

    def all_clustering_predict(self, X):
        """It performs data predictions on everyone on all kinds of possible cluster methods.

        Args:
            X (_type_): Data on all types of clustering will be performed.

        Returns:
            _type_: Data labels after kmeans clustering.
            _type_: Data labels after fuzzy-c-means clustering.
                    A leable belonging to a single cluster.
            _type_: Data labels after fuzzy-c-means clustering.
                    An array of probabilities of belonging to each cluster is assigned to the data.
            _type_: Data labels after agglomerative clustering.
            _type_: Data labels after dbscan clustering.
        """
        y_km = self.kmenas_predict(X)
        y_fcm_hard, y_fcm_soft = self.fuzzy_predict(X)
        y_ac = self.agglomerative_predict(X)
        y_db = self.dbscan_predict(X)

        return y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db


def all_clustering(X):
    """It performs data fitting and predictions on everyone on all kinds of possible cluster methods.

    Args:
        X (_type_): Data on all types of clustering will be performed

    Returns:
        _type_: Returns an object of type clustering_methods in which all the models are saved inside.
        _type_: Data labels after kmeans clustering.
        _type_: Data labels after fuzzy-c-means clustering.
                A leable belonging to a single cluster.
        _type_: Data labels after fuzzy-c-means clustering.
                An array of probabilities of belonging to each cluster is assigned to the data.
        _type_: Data labels after agglomerative clustering.
        _type_: Data labels after dbscan clustering.
    """
    cluster_obj = clustering_methods()
    y_km, _, _ = cluster_obj.kmenas_cluster(X)
    y_fcm_hard, y_fcm_soft, _ = cluster_obj.fuzzy_cluster(X)
    y_ac = cluster_obj.agglomerative_cluster(X)
    y_db = cluster_obj.dbscan_cluster(X)

    return cluster_obj, y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db
