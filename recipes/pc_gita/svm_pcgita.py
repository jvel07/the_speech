from sklearn import preprocessing

from sklearn import svm
from classifiers.svm_utils import svm_fits
import numpy as np
import recipes.pc_gita.utils_pcgita as pcutils
from sklearn.model_selection import cross_validate

# Loading data: 'fisher' or 'ivecs'
x_train, y_train = pcutils.load_data(512, 'monologue', 'fisher')
# x_train, y_train = pcutils.load_data_alternate(64, 'monologue')

# Scale data
# x_train = preprocessing.PowerTransformer().fit_transform(x_train)
# x_train = preprocessing.Normalizer().fit_transform(x_train)
# x_train = preprocessing.RobustScaler().fit_transform(x_train)
x_train = preprocessing.StandardScaler().fit_transform(x_train)

# Training data and evaluating (stratified k-fold CV)
for c in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]:
    # svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    # scores = cross_validate(svc, x_train, y_train, cv=5, scoring=('roc_auc', 'accuracy', 'f1'))
    # sorted(scores.keys())
    # print("with c", c, "-->", np.mean(scores["test_accuracy"]), np.mean(scores["test_roc_auc"]), np.mean(scores["test_f1"]))
    list_scores = []
    for seed in [44654]:# [1367, 684531, 8754, 3215, 54, 3551, 63839845, 11538, 148111, 4310]:
        scores = svm_fits.train_simple_skfcv_pca(x_train, y_train, n_folds=10, c=c, seed=seed)
        list_scores.append(scores)
        print("with c", c, "-->", scores["accuracy"], scores["auc"], scores["f1"])