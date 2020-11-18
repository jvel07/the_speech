import os

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

from common import util
from recipes.dementia_new8k.dementia_new8k_helper import load_data_demecia_new8k

task = 'dementia_new8k'
feat_type = ['xvecs', 'mfcc', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
gaussians = [512]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
# list_c = [1e-4]

for ga in gaussians:
    x_train, y_train, file_train = load_data_demecia_new8k(
                                            gauss='512dim-7thBEA8kAug',
                                            # gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type[0], frame_lev_type=feat_type[1],
                                            n_feats=20, n_deltas=feat_type[2], list_labels=[1, 2])
    y_train[y_train == 2] = 1  # turning labels into binary



    # Scale data
    std_flag = True
    if std_flag == True:
        std_scaler = preprocessing.StandardScaler()
        x_train = std_scaler.fit_transform(x_train)

    # PCA
    pca_flag = False
    if pca_flag == True:
        pca = PCA(n_components=0.97)
        x_train = pca.fit_transform(x_train)


    for c in list_c:
        preds, trues, posteriors = svm_fits.leave_one_out_cv(x_train, y_train.ravel(), c=c)
        acc = accuracy_score(trues, preds)
        auc = roc_auc_score(trues, preds, average='weighted')
        prec = precision_score(trues, preds)
        rec = recall_score(trues, preds)

        # np.savetxt('exp_results/posteriors_{0}_loocv_{1}_pretrained.txt'.format(c, task), posteriors)
        # np.savetxt('exp_results/trues_{0}_loocv_{1}_pretrained.txt'.format(c, task), trues)

        file_n = os.path.basename(file_train)
        print("with", c, "-", ga, "acc:", acc, " auc:", auc, " prec:", prec, " recall:", rec)
        # util.results_to_csv(file_name='exp_results/results_2_{}_{}.csv'.format(task, feat_type[0]),
        #                     list_columns=['Exp. details', 'C-value', 'Accuracy', 'AUC', 'Precision', 'Recall', 'PCA', 'STD'],
        #                     list_values=[file_n, c, acc, auc, prec, rec, pca_flag, std_flag])


    print()
    # print(std_scaler)
    print()
