import os

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

from common.util import plot_confusion_matrix_2
from recipes.sleepiness.sleepiness_helper import load_data_full

task = 'sleepiness'
feat_type = ['xvecs', 'mfcc', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
gaussians = [512]
# list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
list_c = [0.0001]
list_nu = [0.2]
# list_c = [0.001] # pretrainedXvecs
for ga in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, y_test,  file_n = load_data_full(
                                            gauss='512dim-DNNtrain',
                                            # gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type,
                                            n_feats=20)

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    # mean_y_train = np.mean(y_train)
    # std_y_train = np.std(y_train)
    #
    # mean_y_dev = np.mean(y_dev)
    # std_y_dev = np.std(y_dev)
    #
    # mean_y_test = np.mean(y_test)
    # std_y_test = np.std(y_test)


    # pow_scaler = preprocessing.PowerTransformer()
    # x_train = pow_scaler.fit_transform(x_train)


    std_scaler = preprocessing.StandardScaler()
    x_train = std_scaler.fit_transform(x_train)
    x_dev = std_scaler.transform(x_dev)
    x_test = std_scaler.transform(x_test)

    spear_scores = []
    for c in list_c:
        for nu in list_nu:
            preds = svm_fits.train_svr_gpu(x_train, y_train.ravel(), X_eval=x_dev, c=c, nu=nu)
            # preds = svm_fits.train_xgboost_regressor(x_train, y_train.ravel(), X_eval=x_dev)

            # preds = np.around(preds, decimals=0)
            # coef = np.corrcoef(y_dev, preds, rowvar=True)
            coef, p_std = stats.spearmanr(y_dev, preds)
            spear_scores.append(coef)
            print("with", c, "nu", nu, "- spe:", coef)
            # util.results_to_csv(file_name='exp_results/results_{}_{}.csv'.format(task, feat_type[0]),
            #                     list_columns=['Exp. details', 'SpeaC-dev', 'SpeaC-test'],
            #                     list_values=[file_n, coef, coef_test])

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = list_c[np.argmax(spear_scores)]
    print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))

    # clf = svm.LinearSVC(C=0.0001, max_iter=100000)
    # clf.fit(x_combined, y_combined.ravel())
    y_pred = svm_fits.train_svr_gpu(x_combined, y_combined.ravel(), X_eval=x_test, c=0.0001, nu=list_nu[0])

    # y_pred = np.around(y_pred, decimals=0)
    coef_test, p_2 = stats.spearmanr(y_test, y_pred)
    # coef_test2 = np.corrcoef(y_test, y_pred)

    print("Test results with", optimum_complexity, "- spe:", coef_test)

    a = confusion_matrix(y_test, np.around(y_pred), labels=np.unique(y_train))
    plot_confusion_matrix_2(a, np.unique(y_train), 'conf.png')