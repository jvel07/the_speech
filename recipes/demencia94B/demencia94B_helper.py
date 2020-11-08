import os
from itertools import zip_longest

import pandas as pd
import numpy as np


# loading only specific audios. Takes source file with 'id,wavs' specified, takes a list of the original (total) audios.
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold, cross_validate
from sklearn.svm import SVC

from recipes.utils_recipes.utils_recipe import encode_labels

# load the list of wavs (names) from the file that contains their labels
# e.g. for the label "006,3" (file_name,label) in 'source_file' which belongs to the spk 006, get its corresponding
# wav names, i.e., 006A_feher.wav 006B_feher.wav 006C_feher.wav
def load_specific(source_file, list_original_audios):
    array = np.squeeze(np.vstack(list_original_audios))  # converting the list to array
    for idx, ele in enumerate(array):  # iterating the array
        array[idx] = os.path.basename(ele)  # updating the array from full path name to the name of the wav only

    df = pd.read_csv(source_file, dtype=str)
    list_wavs = df.file_name.values.tolist()
    list_selected_audios = []
    for ele in array:
        for ele2 in list_wavs:
            print(ele, ele2)
            if ele2 == ele[0:3]:
                list_selected_audios.append(os.path.dirname(list_original_audios[0]) + '/' + ele)

    return list_selected_audios

work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/data/'  # ubuntu machine

# loads the data given the number of gaussians, the name of the task and the type of feature.
# Used for small datasets; loads single file containing training features.
# example: train/fisher-23mf-0del-2g-train.fisher
def load_data_demetia_new8k(gauss, task, feat_type, frame_lev_type, n_feats, n_deltas, list_labels):
    if (feat_type == 'fisher') or (feat_type == 'ivecs') or (feat_type == 'xvecs'):
        # Set data directories
        file_train = work_dir + '{0}/{0}/{1}/{1}-{2}{3}-{4}del-{5}-{0}.{6}'.format(task, feat_type, n_feats, frame_lev_type, n_deltas, gauss, feat_type)
        file_lbl_train = work_dir + '{}/labels/labels.csv'.format(task)

        # Load data
        X_train = np.loadtxt(file_train)
        df_labels = pd.read_csv(file_lbl_train)
        Y_train, encoder = encode_labels(df_labels.label.values, list_labels)

        return X_train, Y_train.ravel(), file_train
    else:
        raise ValueError("'{}' is not a supported feature representation, please enter 'ivecs' or 'fisher'.".format(feat_type))


def nested_cv(X, Y, num_trials, params):
    from thundersvm import SVC as thunder
    svm = thunder(gpu_id=0)
    # Arrays to store scores
    non_nested_scores = np.zeros(num_trials)
    nested_scores = np.zeros(num_trials)

    # Loop for each trial
    for i in range(num_trials):
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=svm, param_grid=params, cv=inner_cv, n_jobs=8)
        clf.fit(X, Y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_validate(clf, X=X, y=Y, cv=outer_cv, scoring=['accuracy'])
        nested_scores[i] = nested_score.mean()

    score_difference = non_nested_scores - nested_scores

    print("Average difference of {:6f} with std. dev. of {:6f}.".format(score_difference.mean(), score_difference.std()))
    print("Scores:\n", non_nested_scores, nested_scores)


def join_speakers_feats(list_grouped_features):
    x = []
    for element in list_grouped_features:
        for a, b, c in zip_longest(*[iter(element)] * 3):  # iterate over the sublist of the list
            array = np.concatenate((a, b, c))  # concatenating arrays (every 3)
            x.append(array)
    print("Speakers' wavs concatenated!")
    return x


def group_speakers_feats(iterable, n):  # iterate every n element within a list
    #   "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    lista = []
    for x, y, z in zip_longest(*[iter(iterable)] * n):
        lista.append(list((x, y, z)))
    return lista
