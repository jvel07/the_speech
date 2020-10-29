import os
from itertools import zip_longest

import pandas as pd
import numpy as np

from recipes.utils_recipes.utils_recipe import encode_labels
from common import util

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


def create_labels():
    file_class = ['filelist_class1.txt', 'filelist_class2.txt', 'filelist_class3.txt', 'filelist_class4.txt', 'filelist_class5.txt']
    list_all_files = []
    for file in file_class:
        list_all_files.append(np.loadtxt('audio/emotion_aibo/{0}'.format(file), dtype='str'))
    for name in ['train', 'dev', 'test']:
        list_per_set = os.listdir('audio/emotion_aibo/{0}/'.format(name))
        list_per_set.sort()
        list_per_set = ["{}".format(os.path.splitext(element)[0]) for element in list_per_set]

        final_lbl = set(list_per_set).intersection(list_lbl_per_class)

