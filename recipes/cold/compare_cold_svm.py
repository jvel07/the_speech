import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
from classifiers.cross_val import StatifiedGroupK_Fold

from classifiers.cold import cold_helper as ch


def uar_scoring(y_true, y_pred):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar_ = (one + two) / 2
    return uar_


com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
# com_values = [1e-5]

# retrieving groups for stratified group k-fold CV
groups = ch.read_utt_spk_lbl()

# Loading Test, and Combined (Train+Dev)
# X_test, Y_test, X_combined, Y_combined = ch.load_data()  # Load fishers data
# X_test, Y_test, X_combined, Y_combined = ch.load_compare_data()  # Load compare data
X_train, X_test, Y_train, Y_test= ch.load_combined_fandc(64)  # Load fish and compare combined
# X, Y = shuffle(X_train, Y_train, random_state=0)

# Power Transformer
# scaler = preprocessing.PowerTransformer().fit(X_combined)
# X_train_pca = scaler.transform(X_combined)
# X_test_pca = scaler.transform(X_test)

# Normalize data
# normalizer = preprocessing.StandardScaler().fit(X_combined)
# X_train_norm = normalizer.transform(X_combined)
# X_test_norm = normalizer.transform(X_test)

# PCA
# scaler = PCA(n_components=0.95)
# X_train_norm = scaler.fit_transform(X_train_norm)
# X_test_norm = scaler.transform(X_test_norm)

for c in com_values:
    # groups = pre_groups[indi]
    uar, clf = ch.train_model_stkgroup_cv(X_train, Y_train, 5, c, groups, 0)  # With group-strat
    # clf = ch.train_model(X_train_norm, Y_combined, c)
    print("With:", c, "->", uar)

# Evaluating on test
# posteriors = clf._predict_proba_lr(X_test_norm)
# y_p = np.argmax(posteriors, axis=1)
# print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_p))
# uar_ = uar_scoring(Y_test, y_p)
# print(uar_)
#
# np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_dev2_posteriors_{}.txt".format(str(c)),
#                    posteriors, fmt='%.7f')
