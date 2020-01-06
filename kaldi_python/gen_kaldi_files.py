import os

import pandas as pd

from common import util


def create_scp_kaldi():
    new_dir = '/home/egasj/kaldi/egs/dementia/data/ubm_audios/'  # path to the kaldi folder
    work_dir = '/home/egasj/kaldi/egs/dementia/wav-bea-diktafon'  # dir of the project

    list_audios = util.read_files_from_dir(work_dir)  # util.just_original_75()
    new_list = []
    for item2 in list_audios:
        new_list.append(item2 + ' ' + new_dir + item2)
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


# when no speaker id nor lables are provided
def create_utt2spk_kaldi_2():
    audio_dir = '/home/egasj/kaldi/egs/dementia/wav-bea-diktafon/'
    list_audios = util.read_files_from_dir(audio_dir)
    new_list = []
    for i in list_audios:
        ii = os.path.splitext(i)[0]
        new_list.append(ii[0:2] + ' ' + audio_dir + i)  # for non-75-ubm
        # new_list.append(ii[0:9]+' '+audio_dir+i)  #  for bea-diktafon
    return new_list
