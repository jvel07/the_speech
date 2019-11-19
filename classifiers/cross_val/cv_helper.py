import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from classifiers.cross_val.StatifiedGroupK_Fold import StratifiedGroupKfold
from common import data_proc_tools as tools


def grid_search(_x_train, _y_train):
    parameters = [#{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 20, 30, 100]},
                  {'kernel': ['linear'], 'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 20, 30, 100]}
                  ]

    gd_sr = GridSearchCV(estimator=SVC(),
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(_x_train, np.ravel(_y_train))
    best_c = gd_sr.best_params_
    print(gd_sr.best_params_)
    #print(gd_sr.best_estimator_)
    print("Best complexity value:", best_c['C'])
    return best_c['C']


def train_model_grid_search_cv(_x_train, _y_train, n_splits, groups):
    predicciones = []
    ground_truths = []
    skf = StratifiedKFold(n_splits=n_splits)
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for train, test in skf.split(_x_train, _y_train, groups):
        best_c = grid_search(_x_train[train], np.ravel(_y_train[train]))
        svc = svm.LinearSVC(C=best_c, verbose=0, max_iter=965000)  # class_weight='balanced',
        svc.fit(_x_train[train], np.ravel(_y_train[train]))
        y_pred = svc.predict(_x_train[test])
        predicciones.append(y_pred)
        ground_truths.append(_y_train[test])

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))

    return predicciones, ground_truths


# train SVM model with stratified cross-validation
def train_model_cv(_x_train, _y_train, n_splits, _c):
    scores = []
    # scores2 = []
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(_x_train, _y_train):
        svc = svm.LinearSVC(C=_c, verbose=0, max_iter=965000)  # class_weight='balanced',
        x_train, x_test, y_train, y_test = \
            _x_train[train_index], _x_train[test_index], _y_train[train_index], _y_train[test_index]
        print('Train: %s | test: %s' % (train_index, test_index))
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        scores.append(svc.score(x_test, y_test))
        # scores2.append(accuracy_score(y_test, y_pred))
        # pp['final-average'] = predicciones+ground_truths

    return scores


# train SVM model with stratified group cross-validation
def train_model_stratk_group(X, y, n_groups, n_splits, _c):
    sgkf = StratifiedGroupKfold(n_splits=n_splits)
    svc = svm.LinearSVC(C=_c, verbose=0, max_iter=965000)  # class_weight='balanced',
    predicciones = []
    ground_truths = []
    fold = 0
    for train_index, test_index in sgkf.split(X, y, n_groups):
        #fold = + 1
        #np.savetxt("/home/jose/PycharmProjects/the_speech/data/tr_idx_fold_{}".format(fold), train_index)
        #np.savetxt("/home/jose/PycharmProjects/the_speech/data/tst_idx_fold_{}".format(fold), test_index)
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        svc.fit(X_train, np.ravel(y_train))
        y_pred = svc.predict(X_test)
        predicciones.append(y_pred)
        ground_truths.append(y_test)

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))
    print(sk.metrics.accuracy_score(ground_truths, predicciones))

    return predicciones, ground_truths


def best_pca_components(x):
    var = tools.get_var_ratio_pca(x)
    return tools.sel_pca_comp(var, 0.95)


def train_model_cv_PCA(_x_train, _y_train, n_splits):
    predicciones = []
    ground_truths = []
    # svc = svm.LinearSVC(C=_c, verbose=1, max_iter=965000) #class_weight='balanced',
    skf = StratifiedKFold(n_splits=n_splits)
    for train, test in skf.split(_x_train, _y_train):
        comp = best_pca_components(_x_train[train])
        pca_train = PCA(n_components=comp, svd_solver='auto', whiten=False)
        pca_train.fit(_x_train[train])
        x_train_reduced = pca_train.transform(_x_train[train])
        # pca_test = PCA(n_components=comp, svd_solver='full', whiten=False)
        # pca_test.fit(_x_train[test])
        x_test_reduced = pca_train.transform(_x_train[test])

        best_c = grid_search(x_train_reduced, _y_train[train])
        svc = svm.LinearSVC(C=best_c, verbose=1, max_iter=965000)
        svc.fit(x_train_reduced, np.ravel(_y_train[train]))

        y_pred = svc.predict(x_test_reduced)
        predicciones.append(y_pred)
        ground_truths.append(_y_train[test])
        # pp['final-average'] = predicciones+ground_truths

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))

    return predicciones, ground_truths


def train_model_cv_LDA(_x_train, _y_train, n_splits):
    predicciones = []
    ground_truths = []
    # svc = svm.LinearSVC(C=_c, verbose=1, max_iter=965000) #class_weight='balanced',
    skf = StratifiedKFold(n_splits=n_splits)
    for train, test in skf.split(_x_train, _y_train):
        lda = LDA(n_components=8)
        lda2 = LDA(n_components=8)
        lda.fit(_x_train[train], _y_train[train])
        x_train_reduced = lda.transform(_x_train[train])
        lda2.fit(_x_train[test], _y_train[test])
        test_x_reduced = lda.transform(_x_train[test])
        y_encoded = encode_labels_alz(y)

        best_c = grid_search(x_train_reduced, _y_train[train])
        svc = svm.LinearSVC(C=best_c, verbose=0, max_iter=965000)
        svc.fit(x_train_reduced, np.ravel(y_encoded[train]))
        # svc.fit(_x_train[train], np.ravel(_y_train[train]))

        y_pred = svc.predict(test_x_reduced)
        predicciones.append(y_pred)
        ground_truths.append(y_encoded[test])
        # pp['final-average'] = predicciones+ground_truths

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))

    return predicciones, ground_truths