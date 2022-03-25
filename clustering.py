from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_clusters(X, n_clusters=16, get_centers=False):
    y_pred = create_cluster(X, n_clusters, get_centers)
    
    return y_pred


def create_cluster( X, n_clusters, get_centers):
    centers_cluster = None
    km = KMeans(n_clusters=n_clusters)
    all_distances = km.fit_transform(X)
    y_predicted = km.predict(X) #Ciò che kmeans.transform(X) restituisce è già la distanza della norma L2 da ciascun centro del cluster
    if get_centers:
        centers_cluster = km.cluster_centers_

    return y_predicted, centers_cluster, all_distances