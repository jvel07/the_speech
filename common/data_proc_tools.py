import librosa
import numpy as np
#import bob.io.audio
#import bob.io.base.test_utils
import scipy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, LinearDiscriminantAnalysis


def fit_PCA(_x_train, n_components, svd_solver, whiten):
    pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten).fit(_x_train)
    # B = pca.transform(_x_train_dem)
    return pca


def perform_PCA_manual(_x_train_bea, _x_train_dem, n_components, svd_solver, whiten):
    # calculate the mean of each column
    M = np.mean(_x_train_bea.T, axis=1)
    # center columns by subtracting column means
    C = _x_train_bea - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigen-decomposition of covariance matrix
    values, vectors = np.linalg.eig(V)
    # project data
    P = vectors.T.dot(C.T)
    print(P)
    return P


def standardize_data(x_t):
    scaler = preprocessing.StandardScaler().fit(x_t)
    scaled_data = scaler.transform(x_t)
    print("Data standardized...")
    return scaled_data


def normalize_data(x_t):
    scaler = preprocessing.Normalizer().fit(x_t)
    norm_data = scaler.transform(x_t)
    print("Data normalized...")
    return norm_data


def min_max_scaling(_x):
    scaler = MinMaxScaler().fit(_x)
    return scaler


# Select LDA/PCA components
def get_var_ratio_lda(X, y):
    lda = LinearDiscriminantAnalysis(n_components=None)
    lda.fit(X, y)
    return lda.explained_variance_ratio_


def get_var_ratio_pca(X):
    pca = PCA(n_components=None)
    pca.fit(X)
    return pca.explained_variance_ratio_


def sel_pca_comp(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components




