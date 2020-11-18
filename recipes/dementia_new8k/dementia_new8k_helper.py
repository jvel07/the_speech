import os
from itertools import zip_longest

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut

from recipes.utils_recipes.utils_recipe import encode_labels

work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/data/'  # ubuntu machine


# loads the data given the number of gaussians, the name of the task and the type of feature.
# Used for small datasets; loads single file containing training features.
# example: train/fisher-23mf-0del-2g-train.fisher
def load_data_demecia_new8k(gauss, task, feat_type, frame_lev_type, n_feats, n_deltas, list_labels):
    if (feat_type == 'fisher') or (feat_type == 'ivecs') or (feat_type == 'xvecs'):
        # Set data directories
        file_train = work_dir + '{0}/{1}/{2}/{2}-{3}{4}-{5}del-{6}-{7}.{8}'.format(task, task, feat_type, n_feats, frame_lev_type, n_deltas, gauss, task, feat_type)
        file_lbl_train = work_dir + '{}/labels/labels.csv'.format(task)

        # Load data
        print("Reading features in", os.path.basename(file_train))
        # X_train = np.squeeze(np.load(file_train, allow_pickle=True))
        X_train = np.squeeze(np.loadtxt(file_train))
        df_labels = pd.read_csv(file_lbl_train)
        Y_train, encoder = encode_labels(df_labels.label.values, list_labels)

        return X_train, Y_train.ravel(), file_train
    else:
        raise ValueError("'{}' is not a supported feature representation, please enter 'ivecs' or 'fisher'.".format(feat_type))


