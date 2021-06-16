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
feat_type = ['xvecs', 'spectrogram', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
gaussians = [512]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
# list_c = [0.0001]
list_nu = [0.5]
# list_c = [0.001] # pretrainedXvecs

preds_dev = 0
# xvecs-23mfcc-0del-512dim-train_dev-7234786_fbanks-test.xvecs
# Test results with 0.0001 - spe: 0.37575972110242667

srand_list = ['389743', '564896', '2656842', '2959019', '4336987', '7234786', '9612365', '423877642', '987236753',
              '764352323']
# srand_list = ['423877642', '987236753', '764352323']

for srand in srand_list:
    print("SRAND", srand)
    x_train, x_dev, x_test, y_train, y_dev, y_test,  file_n = load_data_full(
                                            # gauss='512dim-train_dev-{0}{1}'.format(srand, feat),
                                            gauss='512dim-sleepinessDNN-{0}'.format(srand),
                                            # gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type,
                                            n_feats='')


    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    std_flag = False
    if std_flag:
        std_scaler = preprocessing.StandardScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)

        x_combined = std_scaler.fit_transform(x_combined)
        x_test = std_scaler.transform(x_test)

    spear_scores = []
    for c in list_c:
        preds = svm_fits.train_svr_gpu(x_train, y_train.ravel(), X_eval=x_dev, c=c)
        # preds = svm_fits.train_xgboost_regressor(x_train, y_train.ravel(), X_eval=x_dev)

        preds_orig = np.copy(preds)
        preds = sh.linear_trans_preds_dev(y_train=y_train, preds_dev=preds)
        coef, p_std = stats.spearmanr(y_dev, preds)

        # dev_preds_dic['dev_{}_srand_{}'.format(c, srand)] = preds

        spear_scores.append(coef)
        print("with", c, "- spe:", coef)

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = list_c[np.argmax(spear_scores)]
    best_coef = np.max(spear_scores)
    print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))

    csv_name = 'results_{}_srands.csv'.format(task)
    util.results_to_csv(file_name='exp_results/{}'.format(csv_name),
                        list_columns=['Exp. Details', 'Deltas', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
                        list_values=[os.path.basename(file_n), feat_type[2], optimum_complexity, best_coef,
                                     std_flag, 'DEV', srand])

    # Saving best dev posteriors
    dev_preds = svm_fits.train_svr_gpu(x_train, y_train.ravel(), X_eval=x_dev, c=optimum_complexity, nu=0.5)
    np.savetxt('preds_{}/best_preds_dev_{}_srand_{}.txt'.format(feat_type[1], optimum_complexity, srand), dev_preds)

    # Testing the model
    y_pred = svm_fits.train_svr_gpu(x_combined, y_combined.ravel(), X_eval=x_test, c=optimum_complexity, nu=list_nu[0])
    y_pred = sh.linear_trans_preds_test(y_train=y_train, preds_dev=preds_orig, preds_test=y_pred)
    coef_test, p_2 = stats.spearmanr(y_test, y_pred)
    # coef_test2 = np.corrcoef(y_test, y_pred)

    print(os.path.basename(file_n), "\nTest results with", optimum_complexity, "- spe:", coef_test)
    print(20*'-')
    util.results_to_csv(file_name='exp_results/{}'.format(csv_name),
                        list_columns=['Exp. Details', 'Deltas', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
                        list_values=[os.path.basename(file_n), feat_type[2], optimum_complexity, coef_test,
                                     std_flag, 'TEST', srand])
    np.savetxt('preds_{}/preds_test_{}_srand_{}.txt'.format(feat_type[1], optimum_complexity, srand), y_pred)
    #
    # a = confusion_matrix(y_test, np.around(y_pred), labels=np.unique(y_train))
    # plot_confusion_matrix_2(a, np.unique(y_train), 'conf.png', cmap='Oranges', title="Spearman CC .365")