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
    #svc = svm.LinearSVC(C=_c, verbose=0, max_iter=965000)  # class_weight='balanced',
    for train_index, test_index in skf.split(_x_train, _y_train):
        x_train, x_test, y_train, y_test = \
            _x_train[train_index], _x_train[test_index], _y_train[train_index], _y_train[test_index]
        svc = svm.LinearSVC(C=_c, verbose=0, max_iter=965000)  # class_weight='balanced',
        #print('Train: %s | test: %s' % (train_index, test_index))
        svc.fit(x_train, y_train)
        #y_pred = svc.predict(x_test)
        scores.append(svc.score(x_test, y_test))
        # scores2.append(accuracy_score(y_test, y_pred))
        # pp['final-average'] = predicciones+ground_truths

    return scores


# train SVM model with stratified group kfold cross-validation for augmented data
def train_model_stratk_group(_x_train, _y_train, n_splits, groups, _c):
    orig_75_idx_aug = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                       208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296]
    orig_75_idx_augv2 = list(range(0, 1575))[::5]
    orig_75_idx_augv3 = list(range(0, 2025))[::9]

    sgkf = StratifiedGroupKfold(n_splits=n_splits)
    #svc = svm.LinearSVC(C=_c, verbose=0, max_iter=965000)  # class_weight='balanced',
    scores = []
    for train_index, test_index in sgkf.split(_x_train, _y_train, groups):
        x_train, x_test, y_train, y_test = \
            _x_train[train_index], _x_train[test_index], _y_train[train_index], _y_train[test_index]
        #print('Train: %s | test: %s' % (train_index, test_index))
        svc = svm.LinearSVC(C=_c, verbose=0, max_iter=965000)  # class_weight='balanced',
        svc.fit(x_train, y_train)
        # Getting the indexes for the 75 speakers only
        test_index_75 = list(set(orig_75_idx_aug).intersection(test_index))
        test_index_75.sort()
        #print('TEST75: ', test_index_75, 'test_ind:', test_index)
        # Testing on 75
        scores.append(svc.score(_x_train[test_index_75], _y_train[test_index_75]))
        #scores.append(svc.score(x_test, y_test))

    return scores


# CV-TESTING ONLY, over the 75 speakers only (takes svc trained on augmented data)
def test_75(svc, X, y, groups):
    sgkf = StratifiedGroupKfold(n_splits=5)
    scores = []
    for train_index, test_index in sgkf.split(X[::4], y[::4], groups[::4]):
        x_train, x_test, y_train, y_test = \
            X[train_index], X[test_index], y[train_index], y[test_index]
        print('\n TESTING PART: Train: %s | test: %s' % (train_index, test_index))
        scores.append(svc.score(x_test, y_test))
    return scores



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


def train_model_stratkf_LDA(_x_train, _y_train, n_splits, _c):
    scores = []
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(_x_train, _y_train):
        x_train, x_test, y_train, y_test = \
            _x_train[train_index], _x_train[test_index], _y_train[train_index], _y_train[test_index]

        lda = tools.fit_LDA(x_train, y_train, 110)
        x_train_lda = lda.transform(x_train)
        x_test_lda = lda.transform(x_test)

        svc = svm.LinearSVC(C=_c, verbose=0, max_iter=1000)  # class_weight='balanced',
        svc.fit(x_train_lda, y_train)
        # y_pred = svc.predict(x_test)
        scores.append(svc.score(x_test_lda, y_test))
        # scores2.append(accuracy_score(y_test, y_pred))
        # pp['final-average'] = predicciones+ground_truths

    return scores