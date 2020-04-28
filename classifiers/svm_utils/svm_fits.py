import numpy as np
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, make_scorer
from sklearn.svm import SVC

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


def do_lda(x1, x2, Y):
    pca = LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
    x1 = pca.fit_transform(x1, Y)
    x2 = pca.transform(x2)
    return x1, x2


def evaluate_auc_score(y_true, y_pred):
    # y_pred = model._predict_proba_lr(x_test)
    # y_pred = np.argmax(y_pred, axis=1)
    return roc_auc_score(y_true, y_pred)


def evaluate_accuracy(y_true, y_pred):
    # y_pred = model._predict_proba_lr(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred.round())


def evaluate_f1(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred.round())


def evaluate_uar_score(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return recall_score(y_true, y_pred, labels=[2, 1, 0], average='macro')


# SVM with stratified kfold cross validation
def train_simple_skfcv(X, Y, n_folds, c, seed):
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=100000)  # , class_weight='balanced')
    # kf = LeaveOneOut()
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    array_posteriors = np.zeros((len(Y), 2))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # print('test_index', test_index)
        svc.fit(x_train, y_train)
        posteriors = svc._predict_proba_lr(x_test)
        # scores2.append(svc.score(x_test, y_test))
        array_posteriors[test_index] = posteriors
    # acc = evaluate_accuracy(Y, array_posteriors)
    print(array_posteriors)
    uar = evaluate_uar_score(Y, array_posteriors)  # [:, 1])
    # f1 = evaluate_f1(Y, array_posteriors)
    # np.savetxt('probs_mask_cv_{}_fisher'.format(c), array_posteriors)

    # scores = {"accuracy": acc, "uar": uar, "f1": f1, "posteriors": array_posteriors}
    scores = {"uar": uar, "posteriors": array_posteriors}

    return scores, array_posteriors


# SVM with stratified kfold cross validation and pca
def train_simple_skfcv_pca(X, Y, n_folds, c, seed):
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=10000)  # class_weight='balanced',
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    array_posteriors = np.zeros((len(Y), 2))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        x_train, x_test = do_pca(x_train, x_test, 0.97)
        # x_train, x_test = do_lda(x_train, x_test, y_train)
        svc.fit(x_train, y_train)
        posteriors = svc._predict_proba_lr(x_test)
        array_posteriors[test_index] = posteriors
    acc = evaluate_accuracy(Y, array_posteriors)
    auc = roc_auc_score(Y, array_posteriors[:, 1])
    f1 = evaluate_f1(Y, array_posteriors)
    scores = {"accuracy": acc, "auc": auc, "f1": f1}
    return scores, array_posteriors


def train_model_resample(X, Y, c, seed):
    # seeds = [137, 895642, 15986, 4242, 7117, 1255, 1564111, 923, 75, 9656]
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    # for number in seeds:
    X_resampled, Y_resampled, indi = resample_data(X, Y, r=seed)  # resampling
    svc.fit(X_resampled, Y_resampled)  # Training the SVM

    return svc


# train and evaluate normally
def train_model_normal(X, Y, X_t, Y_t, c):
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=20000)  # , class_weight='balanced')
    # X, Y, idx= resample_data(X, Y, r=545412)  # resampling
    svc.fit(X, Y)
    y_prob = svc._predict_proba_lr(X_t)
    y_pred = np.argmax(y_prob, axis=1)
    uar = recall_score(Y_t, y_pred, labels=[1, 0], average='macro')

    return uar, y_prob


def uar_scoring(y_true, y_pred, **kwargs):
    uar = recall_score(y_true, y_pred, labels=[1, 0], average='macro')
    return uar


my_scorer = make_scorer(uar_scoring, greater_is_better=True)


# train and evaluate normally with rbf
def grid_skfcv_gpu(X, Y, params, metrics):
    from thundersvm import SVC as thunder
    for metric in metrics:
        kf = StratifiedKFold(n_splits=10)
        svc = GridSearchCV(
            thunder(gpu_id=0, probability=False, class_weight='balanced'),
            params, scoring=metric,
            n_jobs=6, cv=kf, refit=True, verbose=1)
        svc.fit(X, Y)
        print("Best parameters set found on development set:")
        print()
        print(svc.best_params_, "\n Best Estimator:")
        print(svc.best_estimator_)
        print("Grid scores on development set:")
        means = svc.cv_results_['mean_test_score']
        stds = svc.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, svc.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
    return svc


# train and evaluate normally with rbf
def grid_cv_cpu(X, Y, params):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=25647)
    svc = GridSearchCV(
        SVC(probability=False), params, scoring=my_scorer, n_jobs=6, cv=kf, refit=True, verbose=1
    )
    svc.fit(X, Y)
    print("Best parameters set found:")
    print()
    print(svc.best_params_)
    print("Grid scores:")
    print()
    means = svc.cv_results_['mean_test_score']
    stds = svc.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    return svc


# Train SVM with the thunder library (GPU usage)
def train_skfcv_SVM_gpu(X, Y, n_folds, c, kernel, gamma):
    from thundersvm import SVC as thunder
    svc = thunder(kernel=kernel, C=c, gamma=gamma,  class_weight='balanced', probability=True, max_iter=500000, gpu_id=0)
    # kf = LeaveOneOut()
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    array_posteriors = np.zeros((len(Y), len(np.unique(Y))))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        posteriors = svc.predict_proba(x_test)
        array_posteriors[test_index] = posteriors

    return array_posteriors, svc


def train_skfcv_SVM_cpu(X, Y, n_folds, c):
    svc = svm.LinearSVC(C=c, max_iter=100000, class_weight='balanced')
    # kf = LeaveOneOut()
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    array_posteriors = np.zeros((len(Y), len(np.unique(Y))))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        posteriors = svc._predict_proba_lr(x_test)
        array_posteriors[test_index] = posteriors

    return array_posteriors, svc

def train_skfcv_RBF_cpu(X, Y, n_folds, c, gamma):
    svc = svm.SVC(kernel='rbf', gamma=gamma, probability=True, C=c, verbose=0, max_iter=100000, class_weight='balanced')
    # kf = LeaveOneOut()
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    array_posteriors = np.zeros((len(Y), len(np.unique(Y))))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        posteriors = svc.predict_proba(x_test)
        array_posteriors[test_index] = posteriors

    return array_posteriors, svc


def train_svm_gpu(X, Y, X_eval, c, kernel, gamma):
    from thundersvm import SVC as thunder
    svc = thunder(kernel=kernel, C=c, probability=True, gamma=gamma, class_weight='balanced', max_iter=100000, gpu_id=0)
    svc.fit(X, Y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob


def train_linearsvm_cpu(X, Y, X_eval, c):
    svc = svm.LinearSVC(C=c,  class_weight='balanced', max_iter=100000)
    svc.fit(X, Y)
    y_prob = svc._predict_proba_lr(X_eval)
    return y_prob


def train_rbfsvm_cpu(X, Y, X_eval, c, gamma):
    svc = svm.SVC(kernel='rbf', gamma=gamma, probability=True, C=c, verbose=0, max_iter=100000, class_weight='balanced')
    svc.fit(X, Y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob