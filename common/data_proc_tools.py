import random
from collections import Counter, defaultdict

import numpy as np
# import bob.io.audio
# import bob.io.base.test_utils
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler


def fit_PCA(_x_train, n_components, svd_solver, whiten):
    pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten).fit(_x_train)
    # B = pca.transform(_x_train_dem)
    return pca.transform(_x_train)


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


def fit_standardize_data(x_t):
    scaler = preprocessing.StandardScaler().fit(x_t)
    print("Data standardized...")
    return scaler


def normalize_data(x_t):
    scaler = preprocessing.Normalizer().fit(x_t)
    norm_data = scaler.transform(x_t)
    #print("Data normalized...")
    return norm_data


def fit_normalize_data(x_t):
    scaler = preprocessing.Normalizer().fit(x_t)
   # print("Data fitted as normalized...")
    return scaler


def min_max_scaling(_x):
    scaler = MinMaxScaler().fit_transform(_x)
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


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices



