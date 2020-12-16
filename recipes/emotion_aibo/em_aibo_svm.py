import os

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np
from sklearn.utils import shuffle

import recipes.utils_recipes.utils_recipe as rutils
from common import util
from recipes.emotion_aibo import em_aibo_helper as h

task = 'emotion_aibo'
feat_info = ['xvecs', 'mfcc', 23, 0]  # provide the types of features, type of frame-level feats, number of flevel coef, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's
# gaussians = [2, 4, 8, 16, 32, 64]
gaussians = [2]

for ga in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, y_test, file_n, le = rutils.load_data_full_2(
        gauss='512dim-pretrained',
        # gauss='{}g'.format(ga),
        task=task,
        feat_info=feat_info,
        list_labels=[1,2,3,4,5]
    )

    class1, class2, class3, class4, class5, class1_lbl, class2_lbl, class3_lbl, class4_lbl, class5_lbl = h.load_synthetic_data(obs='pretrained')
    class1_lbl = le.transform(class1_lbl)
    class2_lbl = le.transform(class2_lbl)
    class3_lbl = le.transform(class3_lbl)
    class4_lbl = le.transform(class4_lbl)
    class5_lbl = le.transform(class5_lbl)

    x_train = np.concatenate((x_train, class1, class4, class5))
    y_train = np.concatenate((y_train, class1_lbl, class4_lbl, class5_lbl))

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    x_combined, y_combined = shuffle(x_combined, y_combined)

    # Scale data
    std_flag = False
    if std_flag == True:
        std_scaler = preprocessing.RobustScaler()
        std_scaler2 = preprocessing.RobustScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)
        x_combined = std_scaler2.fit_transform(x_combined)
        x_test = std_scaler2.transform(x_test)


    # PCA
    pca_flag = False
    if pca_flag == True:
        pca = PCA(n_components=0.97)
        pca2 = PCA(n_components=0.97)
        x_train = pca.fit_transform(x_train)
        x_dev = pca.transform(x_dev)
        x_combined = pca2.fit_transform(x_combined)
        x_test = pca2.transform(x_test)

    x_train_resampled, y_train_resampled = h.oversample(x_train, y_train)
    x_combined_resampled, y_combined_resampled = h.oversample(x_combined, y_combined)

    # list_gamma = [1, 0.1, 1e-2, 1e-3, 1e-4]
    list_gamma = [0.01]

    list_c2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    # list_c2 = [1e-2, 0.1, 1]
    # list_c2 = [1e-5]
    uar_scores = []

    for c in list_c2:
        for g in list_gamma:  # [1367, 684531, 8754, 3215, 54, 3551, 63839845, 11538, 148111, 4310]:
            # svc = svm_fits.grid_skfcv_gpu(x_combined, y_combined.ravel(), params=tuned_parameters, metrics=[my_scorer])

            # posteriors, clf = svm_fits.train_skfcv_SVM_gpu(x_combined, y_combined.ravel(), c=c, kernel=kernel, gamma=g, n_folds=folds)
            # posteriors, clf = svm_fits.train_skfcv_SVM_cpu(x_combined, y_combined.ravel(), c=c, n_folds=10)
            # posteriors, clf = svm_fits.train_skfcv_RBF_cpu(x_combined, y_combined.ravel(), c=c, n_folds=5, gamma=g)

            # posteriors = svm_fits.train_svm_gpu(x_combined, y_combined.ravel(), c=c, X_eval=alternate_x_test, kernel=kernel, gamma=g)
            posteriors = svm_fits.train_linearsvm_cpu(x_train_resampled, y_train_resampled.ravel(), c=c, X_eval=x_dev, class_weight='balanced')
            # posteriors = svm_fits.train_rbfsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, gamma=g)

            # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_k{}_{}_fisher_{}.txt'.format(folds, c, kernel), posteriors)
            y_pred = np.argmax(posteriors, axis=1)
            uar = recall_score(y_dev, y_pred, np.unique(y_train), average='macro')
            uar_scores.append(uar)
            print("with", c, "-", g, uar)
            # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_{}_fisher32plp_{}.txt'.format(c, kernel), posteriors)


    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = list_c2[np.argmax(uar_scores)]
    print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))

    clf = svm.LinearSVC(C=optimum_complexity, max_iter=100000, class_weight='balanced')
    clf.fit(x_combined_resampled, y_combined_resampled.ravel())
    # y_pred = svm_fits.train_svr_gpu(x_combined, y_combined.ravel(), X_eval=x_test, c=optimum_complexity, nu=list_nu[0])
    # y_pred = sh.linear_trans_preds_test(y_train=y_train, preds_dev=preds_orig, preds_test=y_pred)
    test_probs = clf._predict_proba_lr(x_test)
    test_preds = np.argmax(test_probs, axis=1)
    uar_final = recall_score(y_test, test_preds, labels=np.unique(y_train), average='macro')

    print(os.path.basename(file_n), "\nTest results with", optimum_complexity, "- UAR:", uar_final)
    util.results_to_csv(file_name='exp_results/results_{}_{}.csv'.format(task, feat_info[0]),
                        list_columns=['Exp. details', 'UAR', 'PCA', 'STD'],
                        list_values=[os.path.basename(file_n), uar_final, pca_flag, std_flag])

    # Inverse-transforming labels to their original (from 0,1...,4 to 1,...,5)
    # lbls_train = le.inverse_transform(y_train)
    # lbls_test = le.inverse_transform(y_test)
    # lbls_preds = le.inverse_transform(test_preds)

    # a = confusion_matrix(y_test, y_train, labels=np.unique(y_train))
    # util.plot_confusion_matrix_2(a, np.unique(y_train), 'conf2.png', cmap='Oranges', title="UAR 0.61")