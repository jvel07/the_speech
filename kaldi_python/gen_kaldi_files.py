import os

import pandas as pd

from common import util
import numpy as np


# generate kaldi scp file
def create_scp_kaldi(list_sets):
    task = 'demencia94B'
    for i in list_sets:
        path = '/media/jose/hk-data/PycharmProjects/the_speech/audio/{}/{}/'.format(task, i)  # path to the kaldi folder
        # work_dir = '/home/egasj/kaldi/egs/cold/audio/wav-bea-diktafon'  # dir of the project
        print(path)

        list_audios = util.read_files_from_dir(path)  # util.just_original_75()
        new_list = []
        for item2 in list_audios:
            new_list.append(item2 + ' ' + path + item2)
        np.savetxt('/home/jose/Documents/kaldi/egs/{}/data/{}/wav.scp'.format(task, i), new_list, fmt="%s", delimiter=' ')
    return new_list


# when the labels are present with the speaker id
def create_utt2spk_kaldi(list_sets):
    task = 'mask'
    for name in list_sets:
        path = '/media/jose/hk-data/PycharmProjects/the_speech/audio/{}/{}/'.format(task, name)  # path to the kaldi folder
        print(path)
        list_audios = util.read_files_from_dir(path) #util.just_original_75()
        df = pd.read_csv('/media/jose/hk-data/PycharmProjects/the_speech/data/{}/labels/labels.csv'.format(task),
                         dtype=str)
        # df.columns = ['id', 'label']
        # ids = df.file_name.values
        df_2 = df[df['file_name'].str.match('test')]
        ids = df_2.file_name.values
        print(ids)
        new_list = []
        for i, j in zip(list_audios, ids):
            ii = os.path.splitext(j)[0]
            new_list.append(i + ' ' + ii)
        np.savetxt('/home/jose/Documents/kaldi/egs/{}/data/{}/utt2spk'.format(task, name), new_list, fmt="%s", delimiter=' ')
        # return new_list


# when no speaker id nor labels are provided. Output e.g.: 130C_szurke.wav 130
def create_utt2spk_kaldi_2(list_sets):
    task = 'demencia94B'
    for i in list_sets:
        path = '/media/jose/hk-data/PycharmProjects/the_speech/audio/{}/{}/'.format(task, i)
        list_audios = util.read_files_from_dir(path)
        new_list = []
        for i in list_audios:
            ii = os.path.splitext(i)[0]
            new_list.append(i + ' ' + ii[0:4])
        np.savetxt('/home/jose/Documents/kaldi/egs/{}/data/train/utt2spk'.format(task, task), new_list, fmt="%s", delimiter=' ')
        return new_list


# for cold database; given in file e.g.: vp010_02_06_butter_009.wav	train_0001.wav
def generate_utt2spk():
    df = pd.read_csv("../data/labels/list-map-testlabels_bk.tsv", sep="\t", header=None)
    df.columns = ['id', 'wav', 'nothing']
    utt = df.wav.values
    spk = df.id.values
    n = []
    for i, j in zip(utt, spk):
        n.append(j[:5] + "_" + i + ' ' + j[:5])
    return n


# order wavs according to spk id for generating a new wav.scp (cold db)
def order_wavs():
    _set = 'test1'
    df = pd.read_csv("wav_c{}.scp".format(_set), sep=" ", header=None)
    df2 = pd.read_csv("utt2spk_c{}".format(_set), sep=" ", header=None)
    df.columns = ['wav', 'path']
    df2.columns = ['wav', 'id']
    ordered_wavs = df.wav.values  # (ascending order: 001, 002, etc)
    ordered_paths = df.path.values
    ordered_ids = df2.id.values
    # new = []
    # for i, j, k in zip(ordered_wavs, ordered_paths, ordered_ids):
    #   new.append(i + " " + j + " " + k)
    df3 = pd.DataFrame(list(zip(ordered_wavs, ordered_paths, ordered_ids)))  # (better than for loop)
    df3.columns = ['wav', 'path', 'spkid']
    df4 = df3.sort_values(by=['spkid', 'wav'])  # sorting wavs by spkid
    df4.columns = ['wav', 'path', 'spkid']
    new_order_wavs = df4.wav.values
    new_order_paths = df4.path.values
    new_order_ids = df4.spkid.values
    n = []  # new list of wavs and paths (kaldi's ordering)
    # for i, j in zip(new_order_wavs, new_order_paths):
    #   n.append(i + ' ' + j)
    # np.savetxt('test.txt'.format(_set), n, fmt="%s", delimiter=' ')
    # rename wavs in kaldi's ordering (renaming needed for kaldi's need)
    for wav, id in zip(new_order_wavs, new_order_ids):
        new_name = id + '_' + wav
        os.rename("/home/egasj/kaldi/egs/cold/audio/{}/{}".format(_set, wav),
                  "/home/egasj/kaldi/egs/cold/audio/{}/{}".format(_set, new_name))

    return n

# create_scp_kaldi()


# generate labels to match the kaldi's order (cold database)
def generate_new_order_labels():
    sets = ['train', 'dev', 'test1']
    for item in sets:
        labels = np.loadtxt('/home/egasj/PycharmProjects/the_speech/data/labels/labels.num.{}.txt'.format(item))
        labels[labels == 2] = 0
        df = pd.read_csv("wav_c{}.scp".format(item), sep=" ", header=None)
        df2 = pd.read_csv("utt2spk_c{}".format(item), sep=" ", header=None)
        df.columns = ['wav', 'path']
        df2.columns = ['wav', 'id']
        ordered_wavs = df.wav.values  # (ascending order: 001, 002, etc)
        ordered_ids = df2.id.values
        df_asc_order = pd.DataFrame(list(zip(ordered_wavs, ordered_ids, labels)), dtype=int)  # (better than for loop)
        df_asc_order.columns = ['wav', 'id', 'label']
        df_new_order= df_asc_order.sort_values(by=['id', 'wav', 'label'])  # sorting wavs by spkid (according to kaldi's order)
        df_new_order[['label']].to_csv('/home/egasj/PycharmProjects/the_speech/data/labels/new_order_{}lbl.csv'.format(item), index=False,
                          header=False)

