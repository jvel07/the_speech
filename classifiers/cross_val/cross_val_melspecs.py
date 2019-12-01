import csv
import os
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PowerTransformer

from common import util, data_proc_tools as tools
from classifiers.cross_val.cv_helper import *


# Loading data (if k=0, loads from txt; loads from pickle otherwise)
def load_data(_x, _y, load_mode):
    x = None
    if load_mode == 'txt':
        x = np.loadtxt(_x)
    elif load_mode == 'pickle':
        x = util.read_pickle(_x)

    y = pd.read_csv(_y, header=None)
    y.columns = ['patient_id', 'diagnosis']
    y.diagnosis = pd.Categorical(y.diagnosis)
    y['diag_code'] = y.diagnosis.cat.codes

    return x, y


# Encoding labels to numbers
def encode_labels_alz(_y):
    le = preprocessing.LabelEncoder()
    le.fit(["k", "e", "a"])
    y = le.transform(_y)
    y = y.reshape(-1, 1)
    return y


# Concatenate speakers wavs (3) in one single array
# Returns list of concatenated arrays
def join_speakers_wavs(list_group_wavs):
    x = []
    for element in list_group_wavs:
        for a, b, c in zip_longest(*[iter(element)] * 3):  # iterate over the sublist of the list
            #a, b, c = a.transpose(), b.transpose(), c.transpose()
            array = np.concatenate((a, b, c), axis=1)  # concatenating arrays (every 3)
            x.append(array)
    #print("Speakers' wavs concatenated!")
    return x#np.hstack(x)


if __name__ == '__main__':
    # Loading data
    work_dir = '/Users/jose/PycharmProjects/the_speech'
    file_x = work_dir + '/data/melspecs/melspec_dem_256'
    #file_y = work_dir+ '/data/ids_labels_375.txt'

    # Load data for 75 spk
    x = util.pickle_load_big(file_x)
    y = np.load('labels_75.npy')
    y_train = encode_labels_alz(y)  # Encoding labels

    # Load augmented data (for 300 spk)
    #x, y_df = load_data(file_x, file_y, load_mode='txt')
    #y_train = y_df.diag_code.values
    #groups = np.array(y_df.patient_id.values)

    # (For Alzheimer's) Each speaker has 3 samples, group every 3 samples
    x_train_grouped = util.group_wavs_speakers(x, 3)  # for original data
    #x_train_grouped = util.group_per_audio_type(x, st=5)  # for augmented data
    # Concatenate 3 wavs per/spk into 1 wav per/spk
    x_train = join_speakers_wavs(x_train_grouped)

    scl = PowerTransformer()
    #scl.fit(x_train)
    #x_train = scl.transform(x_train)
    #x_train = tools.standardize_data(x_train)

    skf = StratifiedKFold(n_splits=5)
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                          learning_rate='adaptive', max_iter=500)
    accuracy = []
    for train, test in skf.split(x_train, y_train):
        svc = svm.LinearSVC(C=0.0001, verbose=0, max_iter=965000)
        svc.fit(x_train[train], np.ravel(y_train[train]))
        y_pred = svc.predict(x_train[test])
        accuracy.append(accuracy_score(y_true=y_train[test], y_pred=y_pred))

    print(accuracy)




