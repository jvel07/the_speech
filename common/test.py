from itertools import repeat

import numpy as np
import sklearn as sk

from common import util
import pandas as pd
import os
import csv
from shutil import copyfile


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




# a = take_only_colds('../audio/train', '../data/labels/labels.num.train.txt')


one = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_1e-07_fisher.txt'
two = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_1_resnet.txt'
def average_post(one, two):
    p1 = np.loadtxt(one)
    p2 = np.loadtxt(two)
    probs = np.mean((p1, p2), axis=0)
    a = np.argmax(probs, axis=1)
    return a


def uar_metric(y_true, y_pred):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar = (one + two) / 2
    return uar

def uar(y_true, y_pred):
    uar_score = uar_metric(y_true, y_pred)
    return 'uar', uar_score, True


one = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_0.1_fisher32plp_linear.txt'
two = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_0.01_resnet.txt'
three = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_0.01_xvecsplp_linear.txt'
def average_post(one, two, three):
    p1 = np.loadtxt(one)
    p2 = np.loadtxt(two)
    p3 = np.loadtxt(three)
    probs = np.mean((p1, p2, p3), axis=0)
    a = np.argmax(probs, axis=1)
    return a
a = average_post(one, two, three)
s = recall_score(y_dev, a, labels=[1, 0], average='macro')
print(s)