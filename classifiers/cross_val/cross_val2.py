import csv
import os
from itertools import zip_longest

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer

from classifiers.cross_val.cv_helper import *
from common import util


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
            array = np.concatenate((a, b, c))  # concatenating arrays (every 3)
            x.append(array)
    #print("Speakers' wavs concatenated!")
    return np.vstack(x)


def metrics(ground_truths, preds):
    accuracy = sk.metrics.accuracy_score(ground_truths, preds)
    #    f1 = sk.metrics.f1_score(ground_truths, preds)
    #   precision = sk.metrics.precision_score(ground_truths, preds)
    #  recall = sk.metrics.recall_score(ground_truths, preds)
    print('acc:', accuracy)
    return accuracy


# Writing results to a csv
def results_to_csv(file_name, g, feat_type, num_filters, deltas, vad, pca, acc):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['Gaussians', 'Feature', 'N_filters', 'VAD', 'PCA', 'Accuracy'])
            file_writer.writerow([g, feat_type, num_filters, deltas, vad, pca, acc])
            print("File " + file_name + " created!")
    else:
        with open(file_name, 'a') as csv_file:
            file_writer = csv.writer(csv_file)
            file_writer.writerow([g, feat_type, num_filters, deltas, vad, pca, acc])
            print("File " + file_name + " updated!")


if __name__ == '__main__':

    work_dir = '/home/egasj/PycharmProjects/the_speech'
    pca_ = 0
    list_num_gauss = [8]
    # obs = 'fbanks_40'
    feat_type = '20mf'
    n_filters = '256i'
    deltas = ''
    vad = ''

    for num_gauss in list_num_gauss:
        # Loading data
        file_x = work_dir +'/data/ivecs/alzheimer/ivecs-{}-{}-{}-{}-{}'.format(num_gauss, feat_type, deltas, vad, n_filters)
        #file_y = work_dir +'/data/ids_labels_300.txt'

        # Load data for 75 spk
        x = np.loadtxt(file_x)
        y = np.load('labels_75.npy')
        y_train = encode_labels_alz(y)  # Encode labels

        # Load augmented data (for 300 or + spk)
        #x, y_df = load_data(file_x, file_y, load_mode='txt')
        #y_train = y_df.diag_code.values
        #groups = np.array(y_df.patient_id.values)

        # (For Alzheimer's) Each speaker has 3 samples, group every 3 samples
        x_train_grouped = util.group_wavs_speakers(x, 3)  # for original data
        #x_train_grouped = util.group_per_audio_type(x, gr=12, st=4)  # for augmented data
        #print(len(x_train_grouped))
        # Concatenate 3 wavs per/spk into 1 wav per/spk
        x_train = join_speakers_wavs(x_train_grouped)

        scl = PowerTransformer()
        #scl.fit(x_train)
        #x_train = scl.transform(x_train)
        #x_train = tools.standardize_data(x_train)

        #c = grid_search(x_train, y_train)
        scores = []
        scores.append(train_model_stratk_group(x_train, y_train, 5, groups, 0.00001))  # training model with augmented
        # scores.append(train_model_cv(x_train, np.ravel(y_train), 5, 0.0001))  # training model with original
        for ii in scores:
            print(np.mean(ii))

        #acc = metrics(ground, pred)
        # print_conf_matrix(ground, pred)
       # results_to_csv('C:/Users/Win10/PycharmProjects/the_speech/data/results_dem.csv',
        #               str(num_gauss), feat_type, n_filters, deltas, str(vad), str(pca_comp), str(acc))
