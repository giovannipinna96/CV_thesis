from sklearn.cluster import KMeans
# TODO difference between MinMaxScaler and scale_features (io porto tutto con mean=0 and std=1)
# (https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler)
from fcmeans import FCM
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from utils import scale_features, save


def kmenas_cluster(X, n_clusters=16, must_save=True):
    X_std = scale_features(X)
    km = KMeans(n_clusters=n_clusters)
    all_distances = km.fit_transform(X_std)
    # Ciò che kmeans.transform(X) restituisce è già la distanza della norma L2 da ciascun centro del cluster
    y_predicted = km.predict(X_std)
    centers_cluster = km.cluster_centers_
    if must_save:
        save(y_predicted, 'kmenas_cluster_y_predicted')
        save(centers_cluster, 'kmenas_cluster_centers_cluster')
        save(all_distances, 'kmenas_cluster_all_distances')
    return y_predicted, centers_cluster, all_distances


def fuzzy_cluster(X_train, X_test, n_clusters=16, must_save=True):
    X_std_train = scale_features(X_train)
    X_std_test = scale_features(X_test)
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(X_std_train)
    y_hard = fcm.predict(X_std_test)
    y_soft = fcm.soft_predict(X_std_test)
    if must_save:
        save(y_hard, 'fuzzy_cluster_y_hard')
        save(y_soft, 'fuzzy_cluster_y_soft')
    return y_hard, y_soft


def agglomerative_cluster(X, n_clusters=16, affinity='euclidean', linkage='complete', must_save=True):
    X_std = scale_features(X)
    ac = AgglomerativeClustering(
        n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = ac.fit_predict(X_std)
    if must_save:
        save(labels, 'agglomerative_cluster_labels')
    return labels


def dbscan_cluster(X, eps=0.2, min_samples=5, metric='euclidean', must_save=True):
    X_std = scale_features(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    y_db = db.fit_predict(X_std)
    if must_save:
        save(y_db, 'dbscan_cluster')
    return y_db
