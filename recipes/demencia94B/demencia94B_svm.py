import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

from recipes.demencia94B.demencia94B_helper import load_data_demecia94b, nested_cv, join_speakers_feats, \
    group_speakers_feats

task = 'demencia94ABC'
feat_type = ['xvecs', 'mf']  # provide the types of features and frame-level features to use e.g.: 'fisher', 'mf'
deli = 0

# Loading data: 'fisher' or 'ivecs's
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256]
gaussians = [2]

for ga in gaussians:
    x_train, y_train = load_data_demecia94b(gauss='512dim',#'{}g'.format(ga),
                                            task=task, feat_type=feat_type[0],
                                            n_feats=23, n_deltas=deli, list_labels=[1,2,3])
    y_train[y_train==2] = 1

    # for demencia 94ABC only (joining 3 wavs per spk into one)
    x_train = group_speakers_feats(x_train, 3)
    x_train = np.squeeze(join_speakers_feats(x_train))

    # Scale data
    std_scaler = preprocessing.StandardScaler()
    # pow_scaler = preprocessing.PowerTransformer()
    # norm_scaler = preprocessing.PowerTransformer()

    # x_train = std_scaler.fit_transform(x_train)

    # pca = PCA(n_components=0.97)
    # x_train = pca.fit_transform(x_train)

    # list_gamma = [0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    list_gamma = [0.1]

    list_c2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]

    # params for rbf
    tuned_parameters = [
        # {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'C': list_c2},
        {'kernel': ['linear'], 'C': list_c2}
    ]

    for c in list_c2:
        for g in list_gamma:
            # Training data and evaluating
            # clf = svm_fits.grid_skfcv_gpu(x_train, y_train.ravel(), params=tuned_parameters, metrics=['accuracy'])
            probs, clf = svm_fits.train_skfcv_SVM_gpu(x_train, y_train.ravel(), c=c, kernel='linear', gamma=g, n_folds=20)
            # probs, clf = svm_fits.train_skfcv_SVM_cpu(x_train, y_train.ravel(), c=0.001, n_folds=5)
            # nested_cv(x_train, y_train, num_trials=30, params=tuned_parameters)

            # predicting
            # y_pred = clf.predict(x_train)

            # Metrics
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_train, y_pred)
            precision = precision_score(y_train, y_pred, average='macro', labels=[0,1])
            recall = recall_score(y_train, y_pred, average='macro', labels=[0,1])
            f1 = f1_score(y_train, y_pred, average='macro', labels=[0,1])

            print("{} gauss - with c".format(ga), c, g, "-->", acc, precision, recall, f1)
            # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_{}_fisher.txt'.format(c), posteriors)
