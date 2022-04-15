# TODO implement methods for dim-reduction, PCA, t-SNE, LDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from umap import UMAP


def _scale_features(data):
    sc = StandardScaler()
    return sc.fit_transform(data)


def dim_reduction_pca(X, y, n_components=2):
    pca = PCA(n_components=n_components)
    X_std = _scale_features(X)
    pca.fit(X_std, y)
    X_reducted = pca.transform(X_std)
    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    return X_reducted, explained_variance_ratio, singular_values


def dim_reduction_lda(X, y, n_components=None):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_std = _scale_features(X)
    lda.fit(X_std, y)
    X_reducted = lda.transform(X_std)
    explained_variance_ratio = lda.explained_variance_ratio_
    return X_reducted, explained_variance_ratio


def dim_reduction_tSNE(X, n_components=2, learning_rate='auto', init='random', n_iter=1000):
    X_std = _scale_features(X)
    return TSNE(n_components=n_components, learning_rate=learning_rate,
                init=init, n_iter=n_iter).fit_transform(X_std)


def dim_reduction_umap(X, n_components=2, init='random', random_state=0):
    X_std = _scale_features(X)
    return UMAP(n_components=n_components, init=init, random_state=random_state).fit_transform(X_std)
