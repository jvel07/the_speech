import bob.io.audio
import bob.io.base.test_utils
import python_speech_features
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

from common import util


def compute_fbanks(dir, audio_list):
    lista_fbanks = []
    print("Calculating fbanks from: " + dir)
    for item2 in audio_list:
        data2 = bob.io.audio.reader(dir + item2)
        fbank = python_speech_features.fbank(signal=data2.load()[0], samplerate=data2.rate, nfilt=40)
        lista_fbanks.append(fbank)

    print("Fbanks list length:", len(lista_fbanks))
    return lista_fbanks


# For using them as arrays (x, num_filters+ 1 energy coef.)
def concatenate_fbanks(fbanks):
    list_fbanks = []
    for item1, item2 in fbanks:
        item2 = item2.reshape(-1, 1)
        conc = np.concatenate((item1, item2), axis=1)
        list_fbanks.append(np.float32(conc))
    # util.save_pickle(file_name, list_fbanks)
    return list_fbanks


if __name__ == '__main__':

    y = np.load('/home/egasj/PycharmProjects/iVectorsBob/learning/cross_val/labels_75.npy')


    dir_wav_ubm = '/home/egasj/PycharmProjects/iVectorsBob/audio/wav-bea-diktafon/'
    dir_wav_dem_all = '/home/egasj/PycharmProjects/iVectorsBob/audio/wav-demencia-all/'
    dir_wav_anon_75 = '/home/egasj/PycharmProjects/iVectorsBob/audio/wav_anon_75_225/'

    audio_list_wav_ubm = util.read_files_from_dir(dir_wav_ubm)
    audio_list_wav_dem = util.read_files_from_dir(dir_wav_anon_75)
    # audio_list_non_75 = util.read_files_from_dir(dir_wav_non_75)
    # audio_list_bea_and_non75 = audio_list_wav_ubm + audio_list_non_75

    observation = '40'
    file_fbanks = '/home/egasj/PycharmProjects/iVectorsBob/data/fbanks/fbanks_dem_{}'.format(observation)
    file_fbanks_bea = '/home/egasj/PycharmProjects/iVectorsBob/data/fbanks/fbanks_bea_dem_{}'.format(observation)

    lines = open("/home/egasj/PycharmProjects/iVectorsBob/data/wavlista-anon-75-225.txt").read().splitlines()
    wavlista_anon_75_225 = []
    for it in lines:
        wav_file = '{}.wav'.format(it)
        wavlista_anon_75_225.append(wav_file)

    # Computing and concatenating fbanks
    fbanks_dem = compute_fbanks(dir_wav_dem_all, wavlista_anon_75_225)
    list_fbanks_dem = concatenate_fbanks(fbanks_dem)

    fbanks_bea = compute_fbanks(dir_wav_ubm, audio_list_wav_ubm)
    arr_fbanks_bea = np.vstack(concatenate_fbanks(fbanks_bea))

    # Saving fbanks
    util.save_pickle(file_fbanks_bea, arr_fbanks_bea)
    util.save_pickle(file_fbanks, list_fbanks_dem)
