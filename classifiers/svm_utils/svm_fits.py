import numpy as np
import sklearn as sk
# from imblearn.under_sampling import RandomUnderSampler
# from nested_cv import NestedCV
from sklearn import svm, linear_model, ensemble
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, GridSearchCV, cross_val_score, cross_validate, \
    RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, make_scorer, precision_score
from sklearn.svm import SVC
# import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from mango import Tuner, scheduler
from scipy.stats import uniform, stats
# from mango.domain.distribution import loguniform

from classifiers.cross_val import StatifiedGroupK_Fold
from common import metrics

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

    return array_posteriors


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
def train_lsvm_normal(X, Y, X_t, Y_t, c):
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=20000)  # , class_weight='balanced')
    # X, Y, idx= resample_data(X, Y, r=545412)  # resampling
    svc.fit(X, Y)
    y_prob = svc._predict_proba_lr(X_t)
    y_pred = np.argmax(y_prob, axis=1)
    uar = recall_score(Y_t, y_pred, labels=[1, 0], average='macro')

    return uar, y_prob


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
    svc = thunder(kernel=kernel, C=c, gamma=gamma, class_weight='balanced', probability=True, max_iter=500000, gpu_id=0)
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
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=None)
    array_posteriors = np.zeros((len(Y), len(np.unique(Y))))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        posteriors = svc._predict_proba_lr(x_test)
        array_posteriors[test_index] = posteriors

    return array_posteriors, svc


# def recall_macro(y_true, y_pred, **kwargs):
#     one = recall_score(y_true, y_pred, pos_label=0)
#     return one
#
# recall_macro_scorer = make_scorer(recall_macro, greater_is_better=True)


def train_nested_cv_lsvm(X, Y, inner_folds, outer_folds, metric):
    svc = svm.LinearSVC(max_iter=100000, class_weight='balanced')
    # svc = svm.NuSVC(max_iter=100000, class_weight='balanced')
    p_grid = {'C': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]}
    # p_grid = {'nu': [0.2, 0.5]}

    # CV generator inner (n_splits), outter (n_repeats)
    # cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_folds)
    clf = GridSearchCV(estimator=svc, param_grid=p_grid, cv=inner_folds, scoring=metric,
                       refit=True)  # scoring=['precision_macro', 'recall_macro', 'accuracy', 'f1'], refit=False)
    clf.fit(X, Y)
    nested_score = cross_val_score(clf, X, Y, cv=outer_folds, scoring='f1', n_jobs=-1)

    print(nested_score)
    print(np.mean(nested_score), np.std(nested_score))

    return np.mean(nested_score)
    # Define parameters for function
    # nested_CV_search = NestedCV(model=svc, params_grid=p_grid, outer_kfolds=outer_folds, inner_kfolds=inner_folds, n_jobs=-1,
    #                             cv_options={'metric': accuracy_score, 'sqrt_of_score': True, 'randomized_search_iter': 30,
    #                                         'metric_score_indicator_lower': 'False',
    #                                         })
    # nested_CV_search.fit(X=X, y=Y)
    # print('\nOuter scores:\n{0} \nmean outer scores:\n{1} \nbest params:\n{2} \nmodel variance:\n{3}'.format(nested_CV_search.outer_scores,
    #                                                                              np.mean((nested_CV_search.best_inner_score_list)),
    #                                                                              nested_CV_search.best_params,
    #                                                                              nested_CV_search.variance))


def train_svm_gpu(X, Y, X_eval, c, kernel='linear', gamma='auto'):
    from thundersvm import SVC as thunder
    svc = thunder(kernel=kernel, C=c, probability=True, gamma=gamma, class_weight='balanced', max_iter=100000, gpu_id=0)
    svc.fit(X, Y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob


def train_svr_gpu(X, Y, X_eval, c, kernel='linear', nu=0.5):
    from thundersvm import NuSVR as thunder
    svc = thunder(kernel=kernel, C=c, max_iter=100000, gpu_id=0, nu=nu, gamma='auto')
    svc.fit(X, Y)
    y_prob = svc.predict(X_eval)
    return y_prob


def train_linearsvm_cpu(X, Y, X_eval, c, class_weight):
    svc = svm.LinearSVC(C=c, class_weight=class_weight, max_iter=100000)
    svc.fit(X, Y)
    y_prob = svc._predict_proba_lr(X_eval)
    return y_prob, svc


def train_xgboost_regressor(X, Y, X_eval):
    model = xgb.XGBRegressor(tree_method='gpu_hist')
    xgb.train(X, Y)
    y_pred = model.predict(X_eval)
    return y_pred


def train_svr_cpu(X, Y, X_eval, c, kernel='linear', nu=0.5):
    svc = svm.NuSVR(kernel=kernel, C=c, max_iter=100000, nu=nu, gamma='auto')
    svc.fit(X, Y)
    y_prob = svc.predict(X_eval)
    return y_prob


def train_rbfsvm_cpu(X, Y, X_eval, c, gamma):
    svc = svm.SVC(kernel='rbf', gamma=gamma, probability=True, C=c, verbose=0, max_iter=100000, class_weight='balanced')
    svc.fit(X, Y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob


def leave_one_out_cv(X, y, c):
    loo = LeaveOneOut()
    svc = svm.LinearSVC(C=c, class_weight='balanced', max_iter=100000)
    array_posteriors = np.zeros((len(y), len(np.unique(y))))
    list_trues = []

    for train_index, test_index in loo.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(X_train, y_train)

        y_prob = svc._predict_proba_lr(X_test)
        array_posteriors[test_index] = y_prob
        # preds = np.argmax(array_posteriors, axis=1)
        preds = array_posteriors[:, 1]
        list_trues.append(y_test)

    return preds.round(), np.squeeze(list_trues), array_posteriors


def skfcv_svm_cpu(X, Y, n_folds, c, kernel):
    svc = svm.LinearSVC(C=c,  class_weight='balanced', max_iter=100000)
    # svc = svm.SVC(kernel=kernel, probability=True, C=c, verbose=0, max_iter=100000, class_weight='balanced')
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    array_posteriors = np.zeros((len(Y), len(np.unique(Y))))
    list_trues = np.zeros((len(Y),))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        posteriors = svc._predict_proba_lr(x_test)
        # posteriors = svc.predict_proba(x_test)
        array_posteriors[test_index] = posteriors
        preds = np.argmax(array_posteriors, axis=1)
        list_trues[test_index] = y_test

    return preds, list_trues, array_posteriors


def skfcv_svr_cpu(X, Y, n_folds, c, kernel):
    svc = svm.SVR(kernel=kernel, C=c, verbose=0, max_iter=100000)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    trues = np.zeros((len(Y),))
    preds = np.zeros((len(Y),))

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        pred = svc.predict(x_test)
        preds[test_index] = pred
        trues[test_index] = y_test

    return preds, trues


def normalCV_NuSVR_cpu(X, Y, n_folds, c, kernel):
    svc = svm.NuSVR(kernel=kernel, C=c, verbose=0, max_iter=100000)
    kf = KFold(n_splits=n_folds, random_state=None)

    array_preds = np.zeros((len(Y),))
    list_trues = np.zeros((len(Y),))

    for train_index, test_index in kf.split(X=X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        pred = svc.predict(x_test)
        array_preds[test_index] = pred
        list_trues[test_index] = y_test

    return array_preds, list_trues


def loocv_NuSVR_cpu(X, Y, c, kernel):
    svc = svm.NuSVR(kernel=kernel, C=c, verbose=0, max_iter=100000)
    loo = LeaveOneOut()

    array_preds = np.zeros((len(Y),))
    list_trues = np.zeros((len(Y),))

    for train_index, test_index in loo.split(X=X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        pred = svc.predict(x_test)
        array_preds[test_index] = pred
        list_trues[test_index] = y_test

    return array_preds, list_trues


def feat_selection_spearman(x, y, keep_feats):
    corr_list = []
    for idx_column_feature in range(len(x[1])):
        corr, _ = stats.pearsonr(y, x[:, idx_column_feature])  # take corr
        # print("y", y.shape)
        # print("x", x[:, idx_column_feature].shape)
        corr_list.append(abs(corr))  # collect the corr (abs) values
    ordered_asc = sorted(corr_list, reverse=True)  # sort desc the corr list
    min_corr = ordered_asc[0:keep_feats]  # pick n most correlating # min_corr = # n higher correlated
    indices = [index for index, item in enumerate(corr_list) if
               item in set(min_corr)]  # take the indices that correspond to the min_corr values in the corr_list
    return indices


def loocv_NuSVR_cpu_pearson(X, Y, c, kernel, keep_feats):
    from thundersvm import NuSVR as thunder
    svc = thunder(kernel=kernel, C=c, verbose=0, max_iter=100000)
    # svc = svm.NuSVR(kernel=kernel, C=c, verbose=0, max_iter=100000)
    loo = LeaveOneOut()

    array_preds = np.zeros((len(Y),))
    list_trues = np.zeros((len(Y),))

    for train_index, test_index in loo.split(X=X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # doing feature selection based on the most correlated features
        selected_idx_train = feat_selection_spearman(x_train, y_train, keep_feats)
        x_train_selected = x_train[:, selected_idx_train]
        x_test_selected = x_test[:, selected_idx_train]
        # keepfeats = find(corr >= min_corr)
        # print(x_train_selected.shape)

        svc.fit(x_train_selected, y_train)
        pred = svc.predict(x_test_selected)
        array_preds[test_index] = pred
        list_trues[test_index] = y_test

    return array_preds, list_trues


def loocv_SVR_cpu(X, Y, c, kernel):
    svc = svm.SVR(kernel=kernel, C=c, verbose=0, max_iter=100000)
    loo = LeaveOneOut()

    array_preds = np.zeros((len(Y),))
    list_trues = np.zeros((len(Y),))

    for train_index, test_index in loo.split(X=X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        pred = svc.predict(x_test)
        array_preds[test_index] = pred
        list_trues[test_index] = y_test

    return array_preds, list_trues


def skfcv_PCA_svmlinear_cpu(X, Y, n_folds, c, pca=0.97):
    # svc = svm.LinearSVC(C=c,  class_weight='balanced', max_iter=100000)
    svc = svm.SVC(kernel='linear', probability=True, C=c, verbose=0, max_iter=100000, class_weight='balanced')

    kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    array_posteriors = np.zeros((len(Y), len(np.unique(Y))))
    list_trues = np.zeros((len(Y),))
    pca = PCA(n_components=pca)

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # PCA
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        svc.fit(x_train, y_train)
        posteriors = svc.predict_proba(x_test)
        array_posteriors[test_index] = posteriors
        preds = np.argmax(array_posteriors, axis=1)
        list_trues[test_index] = y_test

    return preds, list_trues, array_posteriors

######## scoring #####
def uar_scoring(y_true, y_pred, **kwargs):
    uar = recall_score(y_true, y_pred, labels=[1, 0], average='macro')
    return uar

my_scorer = make_scorer(uar_scoring, greater_is_better=True)


#  check https://github.com/ARM-software/mango/
def train_mango_skcv(X, Y, n_splits):
    from thundersvm import SVC as thunder
    param_space = {
        # 'kernel': ['rbf', 'linear'],
        # 'gamma': uniform(0.1, 4),  # 0.1 to 4.1
        'C': [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1] #loguniform(-6, 6)  # 10^-7 to 10
                   }

    # @scheduler.serial
    def objectiveSVM(args_list):
        results = []
        for hyper_par in args_list:
            svc = svm.LinearSVC(**hyper_par, max_iter=100000,
                                class_weight='balanced')
            # svc = thunder(**hyper_par, max_iter=100000,
            #               class_weight='balanced')
            result = cross_val_score(svc, X, Y, scoring=my_scorer, n_jobs=-1, cv=n_splits).mean()
            results.append(result)

        return results

    tuner = Tuner(param_dict=param_space, objective=objectiveSVM)
    results = tuner.maximize()
    return results

def train_skfcv_RBF_cpu(X, Y, n_folds, c, gamma):
    svc = svm.SVC(kernel='rbf', gamma=gamma, probability=True, C=c, verbose=0, max_iter=100000, class_weight='balanced')
    # kf = LeaveOneOut()
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    array_posteriors = np.zeros((len(Y), len(np.unique(Y))))
    list_trues = []

    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svc.fit(x_train, y_train)
        posteriors = svc.predict_proba(x_test)
        array_posteriors[test_index] = posteriors
        list_trues.append(y_test)

    return array_posteriors, svc
