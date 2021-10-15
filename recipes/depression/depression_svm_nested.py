import os

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import recall_score, roc_auc_score, f1_score, mean_squared_error, make_scorer, accuracy_score
from sklearn.model_selection import KFold, GridSearchCV

from sklearn import svm
from classifiers.svm_utils import svm_fits
from scipy import stats
import numpy as np

# load data
from classifiers.svm_utils.svm_fits import feat_selection_spearman
from common import util, metrics
from common.metrics import calculate_sensitivity_specificity


def pearson_scoring(y_true, y_pred, **kwargs):
    corr, _ = stats.pearsonr(y_true, y_pred)
    return corr


my_scorer = make_scorer(pearson_scoring, greater_is_better=True)

task = 'depression'
exp_info = ['xvecs', '23mfcc', 'sre16_VAD']  # feat_type, frame-level feat, DNN class

file = '/home/jvel/PycharmProjects/the_speech/data/depression/{0}-{2}-0del-512dim-{1}-train.{0}' \
    .format(exp_info[0], exp_info[2], exp_info[1])
# file = '/home/jvel/PycharmProjects/the_speech/data/depression/ivecs/ivecs-20mfcc-0del-256g-depression.ivecs'
df = pd.read_csv(file, delimiter=' ', header=None)

# load labels
label_file = '../../data/depression/labels/labels_2.csv'
df_lbl = pd.read_csv(label_file, delimiter=',')
data = df_lbl.drop(['filename'], axis=1)  # dropping 'fileName' column
data.fillna(0, inplace=True)

data['Sex'] = data['Sex'].astype('category')  # setting the 'sex' column as category
data['Sex'] = data['Sex'].cat.codes  # encoding cat to numbers

gender, age = data.Sex.values, data.Age.values

# x_train['Sex'] = x_train['Sex'].astype('category')  # setting the 'sex' column as category
# x_train['Sex'] = x_train['Sex'].cat.codes  # encoding cat to numbers

# x_train = x_train.drop(['Age', 'Sex'], axis=1)  # dropping sex and age columns

concat_sex = True
if concat_sex:
    df['512'] = gender

concat_age = True
if concat_age:
    df['513'] = age

X = df.values
y = df_lbl.label.values

# std data
std = True
if std:
    std_scaler = preprocessing.StandardScaler()
    X = std_scaler.fit_transform(X)

# train SVR

# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)

keep_feats_flag = False
# keep_feats = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350]
# keep_feats = [150, 514]
keep_feats = ['None']

feats_cc_scores = []
feats_mrsq_scores = []
feats_auc_scores = []
feats_uar_scores = []

for n in keep_feats:
    print("\n ***** KEEP FEATS: {} ******".format(n))
    # enumerate splits
    outer_results = list()
    corr_scores = []
    uar_scores = []
    auc_scores = []
    f1_scores = []
    spec_scores = []
    sens_scores = []
    eer_scores = []

    array_preds = np.zeros((len(y),))
    list_trues = np.zeros((len(y),))

    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # doing feature selection based on the most correlated features
        if keep_feats_flag:
            selected_idx_train = feat_selection_spearman(X_train, y_train, n)
            X_train = X_train[:, selected_idx_train]
            X_test = X_test[:, selected_idx_train]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
        # define the model
        model = svm.NuSVR(kernel='linear', max_iter=100000)
        # define search space
        space = {'C': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]}
        # define search
        search = GridSearchCV(model, space, scoring=my_scorer, cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # appending preds and trues
        array_preds[test_ix] = yhat
        list_trues[test_ix] = y_test

    # Pearson's CC
    corr, _ = stats.pearsonr(list_trues, array_preds)
    corr_scores.append(corr)

    # binary class
    trues_bin = np.copy(list_trues)
    trues_bin[trues_bin < 13.5] = 0
    trues_bin[trues_bin >= 13.5] = 1
    preds_bin = np.copy(array_preds)
    preds_bin[preds_bin < 13.5] = 0
    preds_bin[preds_bin >= 13.5] = 1

    # metrics
    auc = roc_auc_score(trues_bin, array_preds)
    auc_scores.append(auc)
    uar = recall_score(trues_bin, preds_bin, average='macro')
    uar_scores.append(uar)
    sens_scores.append(recall_score(trues_bin, preds_bin))
    sensitivity, specificity, accuracy = calculate_sensitivity_specificity(trues_bin, preds_bin)
    spec_scores.append(specificity)
    f1 = f1_score(trues_bin, preds_bin)
    f1_scores.append(f1)
    # eer = metrics.calculate_eer(trues_bin, preds_bin)
    eer = mean_squared_error(list_trues, array_preds, squared=False)
    eer_scores.append(eer)

    # print("with {}:".format(c), corr)
    print("corr:", corr, "uar:", uar, "spec:", specificity, "sens:", sensitivity,
          "AUC:", auc, "F1:", f1, "RMSE:", eer)

    # report progress
    print('est=%.3f, cfg=%s' % (result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    print("***** SUMMARIZED PERFORMANCES ******")
    print('PEARSONS: %.3f (%.3f)' % (np.mean(corr_scores), np.std(corr_scores)))
    print('AUC: %.3f (%.3f)' % (np.mean(auc_scores), np.std(auc_scores)))
    print('UAR: %.3f (%.3f)' % (np.mean(uar_scores), np.std(uar_scores)))
    print('SENS: %.3f (%.3f)' % (np.mean(sens_scores), np.std(sens_scores)))
    print('SPEC: %.3f (%.3f)' % (np.mean(spec_scores), np.std(spec_scores)))
    print('F1: %.3f (%.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print('EER: %.3f (%.3f)' % (np.mean(eer_scores), np.std(eer_scores)))

    # util.results_to_csv(file_name='exp_results/results_depression_nestedCV_2.csv',#.format(task, feat_type[0]),
    #                     list_columns=['Exp. Details', 'C', 'STD', 'UAR', 'SPEC', 'SENS', 'AUC', 'f1', 'EER', 'PEARSON',
    #                                   'x-vec model', 'KeepFeats'],
    #                     # list_values=[os.path.basename(file), str(result.best_params_), std,
    #                     #              np.mean(uar_scores), np.mean(spec_scores), np.mean(sens_scores),
    #                     #              np.mean(auc_scores), np.mean(f1_scores), np.mean(eer_scores),
    #                     #              np.mean(corr_scores),
    #                     #              exp_info[2], n])
    #                     list_values=[os.path.basename(file), str(result.best_params_), std,
    #                                  uar, specificity, sensitivity,
    #                                  auc, f1, eer,
    #                                  corr,
    #                                  exp_info[2], n])
