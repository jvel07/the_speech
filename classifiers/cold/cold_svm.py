import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from classifiers.cold import cold_helper as ch

com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
# com_values = [1]

# retrieving groups for stratified group k-fold CV
groups = ch.read_utt_spk_lbl()

# iterating over the gaussians
for g in [2, 4, 8, 16, 32, 64, 128]:
    print("CV Process (gaussians):", g)
    # Loading Train, Dev, Test, and Combined (T+D)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, X_combined, Y_combined = ch.load_data(g)
    # X, Y = shuffle(X_train, Y_train, random_state=0)

    # Normalize data
    scaler = preprocessing.Normalizer().fit(X_combined)
    X_train_norm = scaler.transform(X_combined)
    X_test_norm = scaler.transform(X_test)

    for c in com_values:
        # groups = pre_groups[indi]
        uar, clf = ch.train_model_stkgroup_cv(X_train_norm, Y_combined, 5, c, groups, g)  # With group-strat
        print("With:", c, "->", uar)

    # Evaluating on test
    posteriors = clf._predict_proba_lr(X_test_norm)
    y_p = np.argmax(posteriors, axis=1)
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_p))
    one = sk.metrics.recall_score(Y_test, y_p, pos_label=0)
    two = sk.metrics.recall_score(Y_test, y_p, pos_label=1)
    uar_ = (one + two) / 2
    print(uar_)
