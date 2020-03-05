import numpy as np
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from classifiers.cross_val import StatifiedGroupK_Fold


def resample_data(X, Y, r):
    smote_enn = RandomUnderSampler(random_state=r)
    X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
    indi = smote_enn.sample_indices_
    return X_resampled, y_resampled, indi


# train SVM model with stratified cross-validation
def train_model_stk_cv(X, Y, n_splits, _c):
    # posteriors = []
    skf = StratifiedKFold(n_splits=n_splits)
    svc = svm.LinearSVC(C=_c, verbose=0, max_iter=3000)  # class_weight='balanced',
    list_uar = []
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test, y_train, y_test = \
            X[train_index], X[test_index], Y[train_index], Y[test_index]
        # clf = CalibratedClassifierCV(base_estimator=svc, cv=7).fit(x_train, y_train)
        # Training the SVM
        svc.fit(x_train, y_train)
        posteriors = svc._predict_proba_lr(x_test)
        # mean_post = np.mean(posteriors, axis=0)
        # np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_dev2_posteriors_{}.txt".format(str(c)),
        #            mean_post, fmt='%.7f')
        y_p = np.argmax(posteriors, axis=1)
        # print("Confusion matrix:\n", sk.metrics.confusion_matrix(y_test, y_p))
        one = sk.metrics.recall_score(y_test, y_p, pos_label=0)
        two = sk.metrics.recall_score(y_test, y_p, pos_label=1)
        uar = (one + two) / 2
        # print("UAR:", uar, "\n")
        list_uar.append(uar)

    uar_tot = np.mean(list_uar)
    return uar_tot, svc


# train SVM model with stratified GROUP k-fold cross-validation with data resampling
def train_sgkf_cv_resample(X, Y, n_splits, _c, groups, gaussians):
    seeds = [137, 895642, 15986, 4242, 7117, 1255, 1, 923, 75, 9656]
    array_posteriors = np.zeros((len(Y), 2))

    sgkf = StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=n_splits)
    svc = svm.LinearSVC(C=_c, verbose=0, max_iter=3000)  # class_weight='balanced',
    for number in seeds:
        # print("\nSEED", number)
        for train_index, test_index in sgkf.split(X, Y, groups):
            x_train, x_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
            X_resampled, Y_resampled, indi = resample_data(x_train, y_train, r=number)  # resampling
            svc.fit(X_resampled, Y_resampled)  # Training the SVM
            posteriors = svc._predict_proba_lr(x_test)  # getting probs (posteriors)
            array_posteriors[test_index] = posteriors
            # print("index:", test_index)
            # print("post:", array_posteriors[test_index])
            # print("shape:", array_posteriors[test_index].shape)
        # np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_cv_post_compare _{}_{}g.txt".format(str(_c), str(gaussians)),
        #     array_posteriors, fmt='%.7f')
        y_p = np.argmax(array_posteriors, axis=1)
        one = sk.metrics.recall_score(Y, y_p, pos_label=0)
        two = sk.metrics.recall_score(Y, y_p, pos_label=1)
        uar_tot = (one + two) / 2

    return uar_tot, svc


def do_pca(x1, x2, n_comp):
    pca = PCA(n_components=n_comp)
    x1 = pca.fit_transform(x1)
    x2 = pca.transform(x2)
    return x1, x2


def evaluate_auc_score(model, y_true, x_test):
    y_pred = model._predict_proba_lr(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    return roc_auc_score(y_true, y_pred.round())


def evaluate_accuracy(model, y_true, x_test):
    y_pred = model._predict_proba_lr(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred.round())


def evaluate_f1(model, y_true, x_test):
    y_pred = model._predict_proba_lr(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred.round())


# SVM with simple stratified kfold cross validation
def train_simple_skfcv(X, Y, n_folds, c, metric_type='accuracy'):
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    # kf = KFold(n_splits=n_folds, shuffle=True)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    folds_scores = []
    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        if metric_type == 'accuracy':
            folds_scores.append(evaluate_accuracy(model=svc, y_true=y_test, x_test=x_test))
            acc = np.mean(folds_scores)
            return acc
        elif metric_type == 'auc':
            folds_scores.append(evaluate_auc_score(model=svc, y_true=y_test, x_test=x_test))
            auc = np.mean(folds_scores)
            return auc
        elif metric_type == 'f1':
            folds_scores.append(evaluate_f1(model=svc, y_true=y_test, x_test=x_test))
            f1 = np.mean(folds_scores)
            return f1


# SVM with simple stratified kfold cross validation
def train_simple_skfcv_pca(X, Y, n_folds, c, metric_type='accuracy'):
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    # kf = KFold(n_splits=n_folds, shuffle=True)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    folds_scores = []
    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        x_train, x_test = do_pca(x_train, x_test, 0.97)
        svc.fit(x_train, y_train)
        if metric_type == 'accuracy':
            folds_scores.append(evaluate_accuracy(model=svc, y_true=y_test, x_test=x_test))
            acc = np.mean(folds_scores)
            return acc
        elif metric_type == 'auc':
            folds_scores.append(evaluate_auc_score(model=svc, y_true=y_test, x_test=x_test))
            auc = np.mean(folds_scores)
            return auc
        elif metric_type == 'f1':
            folds_scores.append(evaluate_f1(model=svc, y_true=y_test, x_test=x_test))
            f1 = np.mean(folds_scores)
            return f1


def train_model(X, Y, c):
    seeds = [137, 895642, 15986, 4242, 7117, 1255, 1564111, 923, 75, 9656]
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    for number in seeds:
        X_resampled, Y_resampled, indi = resample_data(X, Y, r=number)  # resampling
        svc.fit(X_resampled, Y_resampled)  # Training the SVM

    return svc
