from sklearn.cluster import KMeans
#  difference between MinMaxScaler and scale_features (io porto tutto con mean=0 and std=1)
# (https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler)
from fcmeans import FCM
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from utils import save


class clustering_methods():
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


    def kmenas_cluster(self, X, n_clusters=16, must_save=True):
        self.km = None
        self.sc_km = None
        X_std, self.sc_km = _scale_features(X)
        km = KMeans(n_clusters=n_clusters)
        all_distances = km.fit_transform(X_std)
        # Ciò che kmeans.transform(X) restituisce è già la distanza della norma L2 da ciascun centro del cluster
        y_km = km.predict(X_std)
        centers_cluster = km.cluster_centers_
        if must_save:
            save(y_km, 'kmenas_cluster_y_predicted')
            save(centers_cluster, 'kmenas_cluster_centers_cluster')
            save(all_distances, 'kmenas_cluster_all_distances')
        self.km = km
        return y_km, centers_cluster, all_distances

    def kmenas_predict(self, X):
        X_std = self.sc_km.transform(X)
        y_predicted = self.km.predict(X_std)
        return y_predicted

    def fuzzy_cluster(self, X, n_clusters=16, must_save=True):
        self.fcm = None
        self.sc_fcm = None
        X_std, self.sc_fcm = _scale_features(X)
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
        X_std = self.sc_fcm.transform(X)
        y_predicted_hard = self.fcm.predict(X_std)
        y_predicted_soft = self.fcm.soft_predict(X_std)
        return y_predicted_hard, y_predicted_soft

    def agglomerative_cluster(self, X, n_clusters=16, affinity='euclidean', linkage='complete', must_save=True):
        self.ac = None
        self.sc_ac = None
        X_std, self.sc_ac = _scale_features(X)
        ac = AgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        y_ac = ac.fit_predict(X_std)
        if must_save:
            save(y_ac, 'agglomerative_cluster_labels')
        self.ac = ac
        return y_ac

    def agglomerative_predict(self, X):
        X_std = self.sc_ac.transform(X)
        y_predicted = self.ac.predict(X_std)
        return y_predicted

    def dbscan_cluster(self, X, eps=0.2, min_samples=5, metric='euclidean', must_save=True):
        self.db = None
        self.sc_db = None
        X_std, self.sc_db = _scale_features(X)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        y_db = db.fit_predict(X_std)
        if must_save:
            save(y_db, 'dbscan_cluster')
        self.db = db
        return y_db

    def dbscan_predict(self, X):
        X_std = self.sc_db.transform(X)
        y_predicted = self.db.predict(X_std)
        return y_predicted

    def reset(self):
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
        y_km = self.kmenas_predict(X)
        y_fcm_hard, y_fcm_soft = self.fuzzy_predict(X)
        y_ac = self.agglomerative_predict(X)
        y_db = self.dbscan_predict(X)

        return y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db


def all_clustering(X):
    cluster_obj = clustering_methods()
    y_km, _, _ = cluster_obj.kmenas_cluster(X)
    y_fcm_hard, y_fcm_soft, _ = cluster_obj.fuzzy_cluster(X)
    y_ac = cluster_obj.agglomerative_cluster(X)
    y_db = cluster_obj.dbscan_cluster(X)

    return cluster_obj, y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db

def _scale_features(data):
    sc = StandardScaler()
    return sc.fit_transform(data), sc.fit(data)
