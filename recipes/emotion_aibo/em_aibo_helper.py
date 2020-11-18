import os
from itertools import zip_longest

import pandas as pd
import numpy as np
import sklearn

from recipes.utils_recipes.utils_recipe import encode_labels
from common import util
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, KMeansSMOTE, SMOTENC, SVMSMOTE, BorderlineSMOTE

work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/data/'  # ubuntu machine


# loads the data given the number of gaussians, the name of the task and the type of feature.
# Used for small datasets; loads single file containing training features.
# example: train/fisher-23mf-0del-2g-train.fisher
def load_data_demecia_new8k(gauss, task, feat_type, frame_lev_type, n_feats, n_deltas, list_labels):
    if (feat_type == 'fisher') or (feat_type == 'ivecs') or (feat_type == 'xvecs'):
        # Set data directories
        file_train = work_dir + '{}/{}/{}-{}{}-{}del-{}-{}.{}'.format(task, task, feat_type, n_feats, frame_lev_type, n_deltas, gauss, task, feat_type)
        file_lbl_train = work_dir + '{}/labels/labels.csv'.format(task)

        # Load data
        X_train = np.loadtxt(file_train)
        df_labels = pd.read_csv(file_lbl_train)
        Y_train, encoder = encode_labels(df_labels.label.values, list_labels)

        return X_train, Y_train.ravel()
    else:
        raise ValueError("'{}' is not a supported feature representation, please enter 'ivecs' or 'fisher'.".format(feat_type))


def undersample(X, Y):
    # rus = RandomUnderSampler(random_state=None, replacement=True)
    # rus = ClusterCentroids(random_state=None)
    rus = NearMiss(version=3)
    X_resampled, y_resampled = rus.fit_resample(X, Y)
    return X_resampled, y_resampled


def oversample(X, Y):
    rus = RandomOverSampler()
    # rus = SMOTE(k_neighbors=7, n_jobs=8)
    # rus = KMeansSMOTE(n_jobs=8, k_neighbors=18, kmeans_estimator=sklearn.cluster.MiniBatchKMeans(), cluster_balance_threshold=0)
    # rus = SVMSMOTE(n_jobs=8)
    # rus = BorderlineSMOTE(n_jobs=8)
    # rus = ADASYN()
    X_resampled, y_resampled = rus.fit_resample(X, Y)
    return X_resampled, y_resampled


def load_synthetic_data():
    class1 = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class1_gen/xvecs-23mfcc-0del-512dim-DNNtraindev-class1_gen.xvecs')
    class2 = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class2_gen/xvecs-23mfcc-0del-512dim-DNNtraindev-class2_gen.xvecs')
    class3 = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class3_gen/xvecs-23mfcc-0del-512dim-DNNtraindev-class3_gen.xvecs')
    class4 = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class4_gen/xvecs-23mfcc-0del-512dim-DNNtraindev-class4_gen.xvecs')
    class5 = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class5_gen/xvecs-23mfcc-0del-512dim-DNNtraindev-class5_gen.xvecs')

    class1_lbl = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class1_gen/labels_class_1')
    class2_lbl = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class2_gen/labels_class_2')
    class3_lbl = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class3_gen/labels_class_3')
    class4_lbl = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class4_gen/labels_class_4')
    class5_lbl = np.loadtxt('/media/jose/hk-data/PycharmProjects/the_speech/data/emotion_aibo/class5_gen/labels_class_5')


    return class1, class2, class3, class4, class5, class1_lbl, class2_lbl, class3_lbl, class4_lbl, class5_lbl


