from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_clusters(self, X, n_clusters=17, get_centers=False):
    y_pred = create_cluster(X, n_clusters, get_centers)
    
    return y_pred


def _create_cluster(self, X, n_clusters, get_centers):
    centers_cluster = None
    km = KMeans(n_clusters=n_clusters)
    y_predicted = km.fit_predict(X)
    if get_centers:
        centers_cluster = km.cluster_centers_

    return y_predicted, centers_cluster