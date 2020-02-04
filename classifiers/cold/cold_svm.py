import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.utils import shuffle

from classifiers.cold import cold_helper as ch

# Loading Train, Dev, Test and Combined (T+D)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test, X_combined, Y_combined = ch.load_data()
# X, Y = shuffle(X_train, Y_train, random_state=0)

X_total = np.concatenate((X_combined, X_test))
Y_total = np.concatenate((Y_combined, Y_test))

# Normalize data
scaler = preprocessing.Normalizer().fit(X_total)
X_train_norm = scaler.transform(X_total)
X_test_norm = scaler.transform(X_test)

# Resampling, Strat k-fold Cross-val, SVM
com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
# com_values = [1]

X_resampled, Y_resampled = ch.resample_data(X_train_norm, Y_total, r=123456)  # resampling
for c in com_values:
    # for number in [137, 42, 15986, 4242, 7117, 15, 1, 923, 25, 9656]:
    uar, clf = ch.train_model_stk_cv(X_resampled, Y_resampled, 4, c)
    print("With:", c, "->", uar)

# Evaluating on test
posteriors = clf._predict_proba_lr(X_test_norm)
y_p = np.argmax(posteriors, axis=1)
print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_p))
one = sk.metrics.recall_score(Y_test, y_p, pos_label=0)
two = sk.metrics.recall_score(Y_test, y_p, pos_label=1)
uar_ = (one + two) / 2


