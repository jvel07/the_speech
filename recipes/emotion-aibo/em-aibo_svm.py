import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

import recipes.utils_recipes.utils_recipe as rutils


task = 'emotion_aibo'
feat_info = ['xvecs', 'mfcc', 23, 0]  # provide the types of features, type of frame-level feats, number of flevel coef, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's
# gaussians = [2, 4, 8, 16, 32, 64]
gaussians = [2]

for ga in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, y_test = rutils.load_data_full_2(
        gauss='512dim-BEA16kNoAug',
        # gauss='{}g'.format(ga),
        task=task,
        feat_info=feat_info,
        list_labels=[1,2,3,4,5]
    )

    # x_combined = np.concatenate((x_train, x_dev))
    # y_combined = np.concatenate((y_train, y_dev))

    # Scale data
    std_scaler = preprocessing.StandardScaler()
    # pow_scaler = preprocessing.PowerTransformer()

    # x_train = std_scaler.fit_transform(x_train)
    # x_dev = std_scaler.transform(x_dev)
    # x_test = std_scaler.transform(x_test)
    # x_combined = std_scaler.fit_transform(x_combined)

    # list_gamma = [1, 0.1, 1e-2, 1e-3, 1e-4]
    list_gamma = [0.01]

    # list_c2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    # list_c2 = [1e-2, 0.1, 1]
    list_c2 = [1e-5]

    for c in list_c2:
        for g in list_gamma:  # [1367, 684531, 8754, 3215, 54, 3551, 63839845, 11538, 148111, 4310]:
            # svc = svm_fits.grid_skfcv_gpu(x_combined, y_combined.ravel(), params=tuned_parameters, metrics=[my_scorer])

            # posteriors, clf = svm_fits.train_skfcv_SVM_gpu(x_combined, y_combined.ravel(), c=c, kernel=kernel, gamma=g, n_folds=folds)
            # posteriors, clf = svm_fits.train_skfcv_SVM_cpu(x_combined, y_combined.ravel(), c=c, n_folds=10)
            # posteriors, clf = svm_fits.train_skfcv_RBF_cpu(x_combined, y_combined.ravel(), c=c, n_folds=5, gamma=g)

            # posteriors = svm_fits.train_svm_gpu(x_combined, y_combined.ravel(), c=c, X_eval=alternate_x_test, kernel=kernel, gamma=g)
            posteriors = svm_fits.train_linearsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev)
            # posteriors = svm_fits.train_rbfsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, gamma=g)

            # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_k{}_{}_fisher_{}.txt'.format(folds, c, kernel), posteriors)
            y_pred = np.argmax(posteriors, axis=1)
            print("with", c, "-", g, recall_score(y_dev, y_pred, labels=[0,1,2,3,4], average='micro'))
            # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_{}_fisher32plp_{}.txt'.format(c, kernel), posteriors)

