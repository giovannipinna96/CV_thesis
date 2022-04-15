import pstats
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler #TODO difference between MinMaxScaler and scale_features???
from fcmeans import FCM
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from utils import scale_features


def get_clusters(X, n_clusters=16, get_centers=False):
    y_pred = create_cluster(X, n_clusters, get_centers)

    return y_pred


def create_cluster(X, n_clusters, get_centers):
    centers_cluster = None
    km = KMeans(n_clusters=n_clusters)
    all_distances = km.fit_transform(X)
    # Ciò che kmeans.transform(X) restituisce è già la distanza della norma L2 da ciascun centro del cluster
    y_predicted = km.predict(X)
    if get_centers:
        centers_cluster = km.cluster_centers_

    return y_predicted, centers_cluster, all_distances

def create_cluster_fuzzy(X_train, X_test, n_clusters):
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(X_train)
    y_hard = fcm.predict(X_test)
    y_soft = fcm.soft_predict(X_test)

    return y_hard, y_soft

def create_agglomerative_cluster(X, n_clusters=3, affinity='euclidean', linkage='complete'):
    X_std = scale_features(X)
    ac = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = ac.fit_predict(X_std)
    return labels

def create_dbscan_cluster(X, eps=0.2, min_samples=5, metric='euclidean'):
    X_std = scale_features(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    y_db = db.fit_predict(X_std)
    return y_db
