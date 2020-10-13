import os

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np
from scipy import stats

from common import util
from recipes.sleepiness.sleepiness_helper import load_data_full

task = 'sleepiness'
feat_type = ['xvecs', 'mfcc', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
gaussians = [512]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
# list_c = [0.001]
for ga in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, y_test,  file_n= load_data_full(
                                            gauss='512dim-pretrainedXvecs',
                                            # gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type,
                                            n_feats=20)



    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    # pow_scaler = preprocessing.PowerTransformer()
    # x_train = pow_scaler.fit_transform(x_train)


    # std_scaler = preprocessing.RobustScaler()
    # x_train = std_scaler.fit_transform(x_train)
    # x_dev = std_scaler.fit_transform(x_dev)


    for c in list_c:
        preds = svm_fits.train_linearSVR_cpu(x_train, y_train.ravel(), X_eval=x_dev, c=c)

        coef, p = stats.spearmanr(y_dev, preds)
        # coef_test, p_std = stats.spearmanr(y_test, preds_test)
        print("with", c, "- spe:", coef)
        # util.results_to_csv(file_name='exp_results/results_{}_{}.csv'.format(task, feat_type[0]),
        #                     list_columns=['Exp. details', 'SpeaC-dev', 'SpeaC-test'],
        #                     list_values=[file_n, coef, coef_test])
    print()
