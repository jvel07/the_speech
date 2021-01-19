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

from common import util
from common.util import plot_confusion_matrix_2
from recipes.sleepiness.sleepiness_helper import load_data_full
from recipes.sleepiness import sleepiness_helper as sh

task = 'sleepiness'
feat_type = ['xvecs', 'mfcc', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
gaussians = [512]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
# list_c = [0.0001]
list_nu = [0.5]
# list_c = [0.001] # pretrainedXvecs

preds_dev = 0

for ga in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, y_test,  file_n = load_data_full(
                                            gauss='512dim-train_dev-9612365',
                                            # gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type,
                                            n_feats=23)

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    std_flag = False
    if std_flag == True:
        std_scaler = preprocessing.StandardScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)
        x_test = std_scaler.transform(x_test)


    spear_scores = []
    for c in list_c:
        for nu in list_nu:
            preds = svm_fits.train_svr_gpu(x_train, y_train.ravel(), X_eval=x_dev, c=c, nu=nu)
            # preds = svm_fits.train_xgboost_regressor(x_train, y_train.ravel(), X_eval=x_dev)

            preds_orig = preds
            # preds = sh.linear_trans_preds_dev(y_train=y_train, preds_dev=preds)
            coef, p_std = stats.spearmanr(y_dev, preds)

            spear_scores.append(coef)
            print("with", c, "nu", nu, "- spe:", coef)

            util.results_to_csv(file_name='exp_results/results_{}_{}_rand.csv'.format(task, feat_type[0]),
                                list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'SPE', 'STD', 'SET'],
                                list_values=[os.path.basename(file_n), ga, feat_type[2], c, coef,
                                             std_flag, 'DEV'])

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    # optimum_complexity = list_c[np.argmax(spear_scores)]
    # print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))
    #
    # # clf = svm.LinearSVC(C=0.0001, max_iter=100000)
    # # clf.fit(x_combined, y_combined.ravel())
    # y_pred = svm_fits.train_svr_gpu(x_combined, y_combined.ravel(), X_eval=x_test, c=optimum_complexity, nu=list_nu[0])
    # # y_pred = sh.linear_trans_preds_test(y_train=y_train, preds_dev=preds_orig, preds_test=y_pred)
    # coef_test, p_2 = stats.spearmanr(y_test, y_pred)
    # # coef_test2 = np.corrcoef(y_test, y_pred)
    #
    # print(os.path.basename(file_n), "\nTest results with", optimum_complexity, "- spe:", coef_test)

    # util.results_to_csv(file_name='exp_results/results_{}_{}_rand.csv'.format(task, feat_type[0]),
    #                     list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'SPE', 'STD'],
    #                     list_values=[os.path.basename(file_n), ga, feat_type[2], optimum_complexity, coef_test, std_flag])

    # a = confusion_matrix(y_test, np.around(y_pred), labels=np.unique(y_train))
    # plot_confusion_matrix_2(a, np.unique(y_train), 'conf.png', cmap='Oranges', title="Spearman CC .365")