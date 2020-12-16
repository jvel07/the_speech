import os

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score, make_scorer

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np
from common import util

from recipes.demencia94ABC.demencia94ABC_helper import load_data_demetia_new8k, nested_cv, join_speakers_feats, \
    group_speakers_feats

from common import metrics

task = 'demencia94ABC'

# provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0
feat_type = ['xvecs', 'mfcc', 0]

# Loading data: 'fisher' or 'ivecs's
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256]
gaussians = [2]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
# BEA16kNoAugSP
for ga in gaussians:
    x_train, y_train, file_n, enc = load_data_demetia_new8k(
        gauss='512dim-BEA16kNoAugSP',
        # gauss='{}g'.format(ga),
        task=task, feat_type=feat_type[0], frame_lev_type=feat_type[1],
        n_feats=20, n_deltas=feat_type[2], list_labels=[1, 2, 3])
    # y_train[y_train == 2] = 1  # turning labels into binary

    # for demencia 94ABC only (joining 3 wavs per spk into one)
    x_train = group_speakers_feats(x_train, 3)
    x_train = np.squeeze(join_speakers_feats(x_train))

    # Scale data
    std_flag = False
    if std_flag == True:
        std_scaler = preprocessing.StandardScaler()
        x_train = std_scaler.fit_transform(x_train)

    # PCA
    pca_flag = False
    # if pca_flag == True:
    #     pca = PCA(n_components=0.97)
    #     x_train = pca.fit_transform(x_train)

    # # Training data and evaluating (NESTED cv)
    # print("\n******WITH GAUSSIANS={} - {}-{}-{}deltas******".format(ga, feat_type[0], feat_type[1], feat_type[2]))
    # metrics = ['accuracy', 'f1', 'precision', 'recall']
    # a = make_scorer(f1_score, average='weighted', labels=[0,1,2])
    # scores = {}
    # for metric in metrics:
    #     final_score = svm_fits.train_nested_cv_lsvm(X=x_train, Y=y_train, inner_folds=5, outer_folds=10, metric=a)
    #     scores[metric] = final_score
    # util.results_to_csv(file_name='results_94ABC/results_{}_{}.csv'.format(task, feat_type[0]),
    #                     list_columns=['Exp. details', 'Accuracy', 'F1', 'Precision', 'Recall'],
    #                     list_values=[os.path.basename(file_n), scores['accuracy'], scores['f1'], scores['precision'], scores['recall']])

    # Training data and evaluating (STRATIFIED cv)
    # scores = dict.fromkeys(['exp', 'gauss', 'del', 'c', 'acc', 'f1', 'prec', 'rec', 'auc', 'auc-c0', 'auc', 'auc-c1', 'auc-c2'])
    for c in list_c:
        if pca_flag == False:
            preds, trues, posteriors = svm_fits.skfcv_svm_cpu(X=x_train, Y=y_train, n_folds=5, c=c)
        else:
            print("Training with PCA")
            preds, trues, posteriors = svm_fits.skfcv_PCA_svmlinear_cpu(X=x_train, Y=y_train, n_folds=5, c=c, pca=0.97)

        acc = accuracy_score(y_train, preds)
        auc = roc_auc_score(y_train, posteriors, multi_class='ovo', labels=np.unique(y_train))
        aucs = metrics.roc_auc_score_multiclass(actual_class=y_train, pred_class=preds)
        preds[preds == 2] = 1
        trues[trues == 2] = 1
        f1 = f1_score(trues, preds)
        prec = precision_score(trues, preds)
        rec = recall_score(trues, preds)


        print("with", c, "-", ga, "acc:", acc, " f1:", f1, " prec:", prec, " recall:", rec, 'AUC:', auc, 'auc-c0:', aucs[0], 'auc-c1:', aucs[1], 'auc-c2:', aucs[2])
        util.results_to_csv(file_name='results_94ABC/results_2_{}_{}.csv'.format(task, feat_type[0]),
                            list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC', 'auc-c0', 'auc-c1', 'auc-c2', 'PCA', 'STD'],
                            list_values=[os.path.basename(file_n), ga, feat_type[2], c, acc, f1, prec, rec, auc, aucs[0], aucs[1], aucs[2], pca_flag, std_flag])

    # list_c = [0.001, 1e-06, 0.01, 1e-05, 1]
    # for c in list_c:
    #     posteriors, clf = svm_fits.train_skfcv_SVM_cpu(x_train, y_train.ravel(), c=c, n_folds=5)
    #     y_pred = np.argmax(posteriors, axis=1)
    #     print("with", c, "-", ga, accuracy_score(y_train, y_pred),  roc_auc_score(y_train, y_pred))
