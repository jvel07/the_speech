import os

import pandas as pd

from common import util
import numpy as np
import recipes.dementia_new8k.dementia_new8k_helper as ah


def sel_spec_wavs():
    recipe = 'dementia_new8k'
    work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
    audio_dir = work_dir + 'audio/dementia_new8k'

    source_file = work_dir + 'data/{}/labels/labels.csv'.format(recipe)
    list_of_wavs = util.traverse_dir(audio_dir, file_type='.wav')
    list_of_wavs.sort()
    list_specific_wavs = ah.load_specific(source_file=source_file, list_original_audios=list_of_wavs)
    list_specific_wavs.sort()
    wav_names_only = []
    for wav in list_specific_wavs:
        wav_names_only.append(os.path.basename(wav))

    return wav_names_only


# generate kaldi scp file
def create_scp_kaldi(list_sets):
    task = 'multiple_sclerosis'
    audio_folder = 'spontan_munka'
    dest_kaldi_folder = 'multiple_sclerosis'
    for i in list_sets:
        path = '/media/jvel/data/audio/MS/{0}/'.format(i)  # path to the audio
        # path = '/media/jvel/data/audio/MS/{0}/{1}/'.format(audio_folder, i)  # path to the audio when it has train, dev, test partitions
        # path = '/media/jose/hk-data/PycharmProjects/the_speech/audio/dementia_new8k/'
        # work_dir = '/home/egasj/kaldi/egs/cold/audio/wav-bea-diktafon'  # dir of the project
        print(path)

        # list_audios = np.genfromtxt('/media/jose/hk-data/PycharmProjects/the_speech/recipes/demencia94ABC/filt_UBMbea_lthan4secs.txt', dtype=str, delimiter='\n')
        # sel_spec_wavs()
        list_audios = os.listdir(path)
        list_audios.sort()
        new_list = []
        for item2 in list_audios:
            new_list.append(item2 + ' ' + path + item2)
        # np.savetxt('/home/jose/Documents/kaldi/egs/{}/data/{}/wav.scp'.format(task, dest_kaldi_folder), new_list, fmt="%s", delimiter=' ')
        np.savetxt('/home/jvel/Documents/kaldi/egs/{0}/data/{1}/wav.scp'.format(task, i), new_list, fmt="%s", delimiter=' ')

# create_scp_kaldi(['class1_gen', 'class2_gen', 'class3_gen', 'class4_gen', 'class5_gen'])
create_scp_kaldi(['fonetika', 'igefluencia', 'narrative_recall', 'sgap', 'spontan_hobbi', 'velemeny'])#, 'dev', 'test'])


# when the labels are present with the speaker id and are stored in one single file
def create_utt2spk_kaldi(list_sets):
    task = 'multiple_sclerosis'
    audio_folder = 'spontan_munka'
    for name in list_sets:
        # path = '/media/jose/hk-data/PycharmProjects/the_speech/audio/{0}/{1}/'.format(audio_folder, name)  # path to the audio folder
        path = '/media/jvel/data/audio/MS/{0}/'.format(name)  # path to the audio folder
        print(path)
        list_audios = os.listdir(path) #util.just_original_75()
        list_audios.sort()
        # df = pd.read_csv('../data/{}/labels/{}.csv'.format(task, name), dtype=str)
        # df.columns = ['id', 'label']
        # ids = df.file_name.values
        # df_2 = df[df['filename'].str.match(name)]
        # ids = df_2.file_name.values
        # print(ids)
        new_list = []
        for i in list_audios:
            file_name = os.path.basename(i)
            ii = os.path.splitext(file_name)[0]
            # ii = ii[0:7]
            new_list.append(file_name + ' ' + ii)  # spkid is the same as the wav_name
        np.savetxt('/home/jvel/Documents/kaldi/egs/{0}/data/{1}/utt2spk'.format(task, name), new_list, fmt="%s", delimiter=' ')
        # return new_list

create_utt2spk_kaldi(['fonetika', 'igefluencia', 'narrative_recall', 'sgap', 'spontan_hobbi', 'velemeny'])#, 'dev', 'test'])

# when the labels are present with the speaker id and are stored in separated files
def create_utt2spk_kaldi_sep(list_sets):
    task = 'primates'
    audio_folder = 'primates'
    for name in list_sets:
        path = '../audio/{0}/{1}/'.format(audio_folder, name)  # path to the kaldi folder
        print(path)
        list_audios = os.listdir(path) #util.just_original_75()
        list_audios.sort()
        df = pd.read_csv('../data/{}/labels/{}.csv'.format(task, name), dtype=str)
        df['label'] = df['label'].astype('category')
        df['label'] = df['label'].cat.codes
        ids = df.label.values
        # utts = df.filename.values
        print(ids)
        new_list = []

        for i in list_audios:
            value = i + ' ' + i[0:3]
            new_list.append(value)

        # for i, j in zip(list_audios, ids):
        #     # kk = os.path.splitext(k)[0]
        #     value = '{}_{}'.format(str(j), i) + ' 00' + str(j)
        #     new_utt_name = '00{}_{}'.format(str(j), i)
        #     new_list.append(value)
        #     # new_list.sort()
        #     os.rename('{0}{1}'.format(path, i), '{0}{1}'.format(path, new_utt_name))
        np.savetxt('/home/jvel/Documents/kaldi/egs/{0}/data/{1}/utt2spk'.format(task, name), new_list, fmt="%s", delimiter=' ')
        # return new_list


# create_utt2spk_kaldi_sep(['train', 'dev', 'test'])

# when no speaker id nor labels are provided. Output e.g.: 130C_szurke.wav 130
def create_utt2spk_kaldi_2(list_sets):
    task = 'emotion_aibo'
    audio_folder = 'emotion_aibo_gen_classes'
    dest_kaldi_folder = 'emotion_aibo_gen'
    for i1 in list_sets:
        path = '../../audio/{}/{}/'.format(audio_folder, i1)  # when there's train, dev, test folders
        # path = '/media/jose/hk-data/PycharmProjects/the_speech/audio/{}/'.format(audio_folder) # when ther's just one folder
        # list_audios = np.genfromtxt('/media/jose/hk-data/PycharmProjects/the_speech/recipes/demencia94ABC/filt_UBMbea_lthan4secs.txt', dtype=str,
        # delimiter='\n')
        # sel_spec_wavs()
        list_audios = os.listdir(path)
        list_audios.sort()
        print(path)
        new_list = []
        for i in list_audios:
            ii = os.path.splitext(i)[0]
            # new_list.append( i + ' ' + ii[0:6])
            new_list.append(i + ' ' + ii)
            new_list.sort()
        np.savetxt('/home/jvel/Documents/kaldi/egs/{}/data/{}/utt2spk'.format(task, i1), new_list, fmt="%s", delimiter=' ')
        # np.savetxt('/home/jose/Documents/kaldi/egs/{}/data/{}/utt2spk'.format(task, dest_kaldi_folder), new_list, fmt="%s", delimiter=' ')

# create_utt2spk_kaldi_2(['class1_gen', 'class2_gen', 'class3_gen', 'class4_gen', 'class5_gen'])

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

