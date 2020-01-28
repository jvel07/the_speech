from itertools import repeat

import numpy as np
from common import util
import pandas as pd
import os
import csv


# y_labels of dem speaker to pandas Dataframe
def augment_alz_labels():
    lista = np.loadtxt('../classifiers/cross_val/labels-75.txt', delimiter=',', dtype='str').tolist()
    contador = 0
    new = []
    var = [x for item in lista for x in repeat(item, 3)]
    np.savetxt('/opt/project/data/ids_labels_225.txt', var, delimiter=',', fmt='%s')
    print(len(var))


def save_mfccs_txt(in_file, out_file):
    mf = np.load(in_file, allow_pickle=True)
    with open(out_file, mode='w', newline='\n', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')  # , quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for row in mf:
            #row = ["%.7f" % f for f in row]
            writer.writerow(row)


in_list = ['C:/Users/Win10/PycharmProjects/the_speech/data/cold/train/mfccs_cold_13_train_2del.mfcc',
           'C:/Users/Win10/PycharmProjects/the_speech/data/cold/dev/mfccs_cold_13_dev_2del.mfcc',
           'C:/Users/Win10/PycharmProjects/the_speech/data/cold/test/mfccs_cold_13_test_2del.mfcc']

out_list = ['C:/Users/Win10/PycharmProjects/the_speech/data/cold/train/mfccs_cold_13_train_2del.txt',
            'C:/Users/Win10/PycharmProjects/the_speech/data/cold/dev/mfccs_cold_13_dev_2del.txt',
            'C:/Users/Win10/PycharmProjects/the_speech/data/cold/test/mfccs_cold_13_test_2del.txt']


for origen, destino in zip(in_list, out_list):
    save_mfccs_txt(origen, destino)