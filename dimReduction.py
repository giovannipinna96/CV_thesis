from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from umap import UMAP

from utils import scale_features


def dim_reduction_pca(X, y, n_components=2):
    """Perform sklearn PCA after a data normalization.

    Args:
        X (_type_): our data
        y (_type_): our lables
        n_components (int, optional): number of component to consider in PCA. Defaults to 2.

    Returns:
        First: matrix that contain the new value of the data
        Second: array that contain the explained variance ratio for each eigenvector
        Third:  array that contains the singular values that pca found
    """
    pca = PCA(n_components=n_components)
    X_std = scale_features(X)
    pca.fit(X_std, y)
    X_reducted = pca.transform(X_std)
    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    return X_reducted, explained_variance_ratio, singular_values


def dim_reduction_lda(X, y, n_components=None):
    """Perform sklearn LDA after a data normalization.

    Args:
        X (_type_): our data
        y (_type_): our lables
        n_components (_type_, optional): number of component to consider in LDA. Defaults to None.

    Returns:
        First: matrix that contain the new value of the data
        Second: array that contain the explained variance ratio
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_std = scale_features(X)
    lda.fit(X_std, y)
    X_reducted = lda.transform(X_std)
    explained_variance_ratio = lda.explained_variance_ratio_
    return X_reducted, explained_variance_ratio


def dim_reduction_tSNE(X, n_components=2, learning_rate='auto', init='random', n_iter=1000):
    """Perform skelarn t-SNE after a data normalization.

    Args:
        X (_type_): our data
        n_components (int, optional):number of component to consider in t-SNE. Defaults to 2.
        learning_rate (str, optional): Defaults to 'auto'.
        init (str, optional): Defaults to 'random'.
        n_iter (int, optional): Defaults to 1000.

    Returns:
        numpy.ndarray: matrix that contain the new value of the data reducted with t-SNE
    """
    X_std = scale_features(X)
    return TSNE(n_components=n_components, learning_rate=learning_rate,
                init=init, n_iter=n_iter).fit_transform(X_std)


def dim_reduction_umap(X, n_components=2, init='random', random_state=0):
    """Perform UMAP after a data normalization. For perform UMAP has been used the libary "umap-learn".

    Args:
        X (_type_): our data
        n_components (int, optional): Defaults to 2.
        init (str, optional): Defaults to 'random'.
        random_state (int, optional): Defaults to 0.

    Returns:
        numpy.ndarray: matrix that contain the new value of the data reducted with UMAP
    """
    X_std = scale_features(X)
    return UMAP(n_components=n_components, init=init, random_state=random_state).fit_transform(X_std)
