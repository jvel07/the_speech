import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

from recipes.demencia94B.demencia94B_helper import load_data_demecia94b, nested_cv, join_speakers_feats, \
    group_speakers_feats

task = 'dementia'
feat_type = ['fisher', 'mfcc', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's
gaussians = [2, 4, 8, 16, 32, 64]
# gaussians = [2]

for ga in gaussians:
    x_train, y_train = load_data_demecia94b(
                                            # gauss='512dim-demencia94B-nUBM',
                                            gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type[0], frame_lev_type=feat_type[1],
                                            n_feats=20, n_deltas=feat_type[2], list_labels=[1,2,3])
    y_train[y_train == 2] = 1  # turning labels into binary
