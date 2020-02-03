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
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_NONE,
                            escapechar='\\')  # , quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for row in mf:
            writer.wrterow(','.join(str(v).lstrip('[').rstrip(']') for v in row))


def take_only_colds(wavs_dir, labels):
    all_wavs = util.read_files_from_dir(wavs_dir)
    labels = np.loadtxt(labels)
    lista = []
    for wav, label in zip(all_wavs, labels):
        if label == 1: lista.append(wav)
    return lista

# a = take_only_colds('../audio/train', '../data/labels/labels.num.train.txt')
