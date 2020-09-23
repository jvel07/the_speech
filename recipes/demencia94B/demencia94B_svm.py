import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

from recipes.demencia94B.demencia94B_helper import load_data_demecia94b, nested_cv, join_speakers_feats, \
    group_speakers_feats

task = 'demencia94ABC'
feat_type = ['xvecs', 'mf', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's
gaussians = [2, 4, 8, 16, 32, 64]
# gaussians = [2]

for ga in gaussians:
    x_train, y_train = load_data_demecia94b(
                                            gauss='512dim-demencia94B-nUBM',
                                            # gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type[0], frame_lev_type=feat_type[1],
                                            n_feats=23, n_deltas=feat_type[2], list_labels=[1,2,3])
    y_train[y_train == 2] = 1  # turning labels into binary

    # for demencia 94ABC only (joining 3 wavs per spk into one)
    x_train = group_speakers_feats(x_train, 3)
    x_train = np.squeeze(join_speakers_feats(x_train))
    # Scale data
    std_scaler = preprocessing.StandardScaler()
    # pow_scaler = preprocessing.PowerTransformer()
    # norm_scaler = preprocessing.PowerTransformer()

    x_train = std_scaler.fit_transform(x_train)

    # pca = PCA(n_components=0.97)
    # x_train = pca.fit_transform(x_train)

    # Training data and evaluating (nested cv)
    print("\n******WITH GAUSSIANS={} - {}-{}-{}deltas******".format(ga, feat_type[0], feat_type[1], feat_type[2]))
    svm_fits.train_nested_cv_lsvm(X=x_train, Y=y_train, inner_folds=5, outer_folds=10)

    # list_c = [0.001, 1e-06, 0.01, 1e-05, 1]
    # for c in list_c:
    #     posteriors, clf = svm_fits.train_skfcv_SVM_cpu(x_train, y_train.ravel(), c=c, n_folds=5)
    #     y_pred = np.argmax(posteriors, axis=1)
    #     print("with", c, "-", ga, accuracy_score(y_train, y_pred),  roc_auc_score(y_train, y_pred))