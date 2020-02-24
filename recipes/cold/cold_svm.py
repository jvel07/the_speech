import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.metrics import make_scorer

from recipes.cold import cold_helper as ch

# work_dir = '/home/egasj/PycharmProjects/the_speech'  # ubuntu machine
work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'  # windows machine

# com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
com_values = [1e-5]

# retrieving groups for stratified group k-fold CV
groups = ch.read_utt_spk_lbl()

# iterating over the gaussians
for g in [64]: #[2, 4, 8, 16, 32, 64, 128]:
    print("CV Process (gaussians):", g)
    # Loading Test, and Combined (Train+Dev)
    # X_test, Y_test, X_combined, Y_combined = ch.load_data(g)
    X_test, Y_test, X_combined, Y_combined = ch.load_compare_data()
    # X, Y = shuffle(X_train, Y_train, random_state=0)

    # Power Transformer
    # scaler = preprocessing.PowerTransformer().fit(X_combined)
    # X_train_pow = scaler.transform(X_combined)
    # X_test_pow = scaler.transform(X_test)

    # X_train_pow, X_test_pow = ch.power_n(X_combined, X_test)

    # Normalize data
    normalizer = preprocessing.StandardScaler().fit(X_combined)
    X_train_norm = normalizer.transform(X_combined)
    X_test_norm = normalizer.transform(X_test)

    # PCA
    pca = PCA(n_components=0.99) # KernelPCA(kernel='linear', fit_inverse_transform=True, n_components=3500, eigen_solver='arpack')
    X_train_norm = pca.fit_transform(X_train_norm)
    X_test_norm = pca.transform(X_test_norm)

    for c in com_values:
        # groups = pre_groups[indi]
        # uar, clf = ch.train_model_stkgroup_cv(X_train_norm, Y_combined, 5, c, groups, g)  # With group-strat
        clf = ch.train_model(X_train_norm, Y_combined, c)
        print("With:", c, "->")

        # Evaluating on test
        posteriors = clf._predict_proba_lr(X_test_norm)
        y_p = np.argmax(posteriors, axis=1)
        print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_p))
        one = sk.metrics.recall_score(Y_test, y_p, pos_label=0)
        two = sk.metrics.recall_score(Y_test, y_p, pos_label=1)
        uar_ = (one + two) / 2
        print(uar_)
        # np.savetxt(work_dir + "/data/cold/posteriors/mean_final_comp_post_{}_{}g.txt".format(
        #     str(c), str(g)), posteriors, fmt='%.7f')


def uar_scoring(y_true, y_pred, **kwargs):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar_ = (one + two) / 2
    return uar_


my_scorer = make_scorer(uar_scoring, greater_is_better=True)

