import os

import pandas as pd

from common import util
import numpy as np


def create_scp_kaldi():
    path = '/home/egasj/kaldi/egs/cold/audio/test1/'  # path to the kaldi folder
    #work_dir = '/home/egasj/kaldi/egs/cold/audio/wav-bea-diktafon'  # dir of the project

    list_audios = util.read_files_from_dir(path)  # util.just_original_75()
    new_list = []
    for item2 in list_audios:
        new_list.append(item2 + ' ' + path + item2)
    return new_list


# when the labels present the speaker id
def create_utt2spk_kaldi():
    list_audios = util.just_original_75()
    df = pd.read_csv('/opt/project/data/ids_labels_225.txt', header=None, dtype=str)
    df.columns = ['patient_id', 'diagnosis']
    ids = df.patient_id.values
    new_list = []
    for i, j in zip(list_audios, ids):
        # ii = os.path.splitext(i)[0]
        new_list.append(i + ' ' + j)
    return new_list


# when no speaker id nor lables are provided. Output e.g.: 130C_szurke.wav 130
def create_utt2spk_kaldi_2():
    audio_dir = '/home/egasj/kaldi/egs/cold/audio/train/'
    list_audios = util.read_files_from_dir(audio_dir)
    new_list = []
    for i in list_audios:
        ii = os.path.splitext(i)[0]
        new_list.append(i + ' ' + ii[0:10])
    return new_list


# for cold database
def generate_utt2spk():
    df = pd.read_csv("../data/labels/list-map-testlabels.tsv", sep="\t", header=None)
    df.columns = ['id', 'wav', 'nothing']
    utt = df.wav.values
    spk = df.id.values
    n = []
    for i, j in zip(utt, spk):
        n.append(j[:5] + "_" + i + ' ' + j[:5])
    return n
