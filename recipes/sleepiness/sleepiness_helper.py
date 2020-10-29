import os
from itertools import zip_longest

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut

from recipes.utils_recipes.utils_recipe import encode_labels

work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/data/'  # ubuntu machine


# E.g.: (4, 'mask', 'fisher') or 'ivecs'
# example: train/fisher-23mf-0del-2g-train.fisher
# loads data (files' name-format that were generated by this SW) existing in the folders "train", "dev" and "test".
# Format of the data labels required and file with the following headers:
#    'file_name     label'  Example: 'file_name     label'
#    'recording.wav label'. Example: 'train_0001.wav True', 'train_0002.wav 2 False', ...
def load_data_full(gauss, task, feat_type, n_feats):
    list_datasets = ['train', 'dev', 'test']  # names for the datasets
    dict_data = {}
    if (feat_type[0] == 'fisher') or (feat_type[0] == 'ivecs') or (feat_type[0] == 'xvecs'):
        # Load train, dev, test
        for item in list_datasets:
            # Set data directories
            file_dataset = work_dir + '{0}/{1}/{2}-{3}{4}-{5}del-{6}-{7}.{8}'.format(task, item, feat_type[0], n_feats, feat_type[1],
                                                                            feat_type[2], str(gauss), item, feat_type[0])
            # Load datasets
            dict_data['x_'+item] = np.loadtxt(file_dataset)
            # Load labels
            file_lbl_train = work_dir + '{}/labels/labels.csv'.format(task) # set data dir
            df = pd.read_csv(file_lbl_train)
            df_labels = df[df['file_name'].str.match(item)]
            dict_data['y_'+item] = df_labels.label.values
        return dict_data['x_train'], dict_data['x_dev'], dict_data['x_test'], dict_data['y_train'], dict_data['y_dev'], dict_data['y_test'], file_dataset
    else:
        raise ValueError("'{}' is not a supported feature representation, please enter 'ivecs' or 'fisher'.".format(feat_type[0]))


def linear_trans_preds(y_train, preds_dev, preds_test_orig):
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    mean_preds_dev = np.mean(preds_dev)
    std_preds_dev = np.std(preds_dev)

    preds_dev_new = (preds_dev - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_dev_new = np.round(preds_dev_new)
    preds_dev_new[preds_dev_new < 1] = 1
    preds_dev_new[preds_dev_new > 9] = 9

    preds_test_new = (preds_test_orig - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_test_new = np.round(preds_test_new)
    preds_test_new[preds_test_new < 1] = 1
    preds_test_new[preds_test_new > 9] = 9

    return preds_dev_new, preds_test_new


def linear_trans_preds_dev(y_train, preds_dev):
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    mean_preds_dev = np.mean(preds_dev)
    std_preds_dev = np.std(preds_dev)

    preds_dev_new = (preds_dev - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_dev_new = np.round(preds_dev_new)
    preds_dev_new[preds_dev_new < 1] = 1
    preds_dev_new[preds_dev_new > 9] = 9

    return preds_dev_new


def linear_trans_preds_test(y_train, preds_dev, preds_test):
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    mean_preds_dev = np.mean(preds_dev)
    std_preds_dev = np.std(preds_dev)

    preds_test_new = (preds_test - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_test_new = np.round(preds_test_new)
    preds_test_new[preds_test_new < 1] = 1
    preds_test_new[preds_test_new > 9] = 9

    return preds_test_new