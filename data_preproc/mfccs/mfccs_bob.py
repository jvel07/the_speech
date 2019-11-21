import pkg_resources
import numpy as np
import bob.io.audio
import bob.io.base.test_utils
import gc
from common import util
import pickle
import math

import bob.kaldi

'''
lines = open("../data/wavlista-anon-75-225.txt").read().splitlines()
wavlista_anon_75_225 = []
for it in lines:
    wav_file = '{}.wav'.format(it)
    wavlista_anon_75_225.append(wav_file)

lines2 = open("../data/wavlista-anon-non75-szurke.txt").read().splitlines()
wavlista_non75_szur = []
for it2 in lines2:
    wav_file2 = '{}.wav'.format(it2)
    wavlista_non75_szur.append(wav_file2)
'''

##Getting MFCCs from wavs
"""
for item in audio_list_bea_and_non75:
    if os.path.isfile(dir_wav_ubm + item):
        sample = pkg_resources.resource_filename(__name__, dir_wav_ubm + item)
        data = bob.io.audio.reader(sample)
        # MFCC
        array_mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=True)
    else:
        sample = pkg_resources.resource_filename(__name__, dir_wav_non_75 + item)
        data = bob.io.audio.reader(sample)
        # MFCC
        array_mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=True)

    lista_mfccs_ubm.append(array_mfcc)

mfccs_wav_ubm = np.vstack(lista_mfccs_ubm)
print("MFCC features shape:", mfccs_wav_ubm.shape)
"""

#MFCCs for i-vectors extraction
def mfcc_ivec_extr(rows):
    print("Calculating MFCCs for i-vecs extractor in: " + dir_wav_ubm + " with " + str(rows) + "wavs")
    for item2 in audio_list_wav_ubm[-rows:]:
        sample2 = pkg_resources.resource_filename(__name__, dir_wav_ubm + item2)
        data2 = bob.io.audio.reader(sample2)
        # MFCC
        array_mfcc2 = bob.kaldi.mfcc(data2.load()[0], data2.rate, normalization=True, num_ceps=20)
        lista_mfccs_ivecs.append(array_mfcc2)

    #mfccs_wav_ivec = np.vstack(lista_mfccs_ivecs)
    print("MFCCs list length:", len(lista_mfccs_ivecs))
    return lista_mfccs_ivecs


#MFCCs for UBM training
def mfcc_ubm(rows):
    print("Calculating MFCCs for UBM in:" + dir_wav_ubm + " with " + str(rows) + " wavs")
    for item in audio_list_wav_ubm[:rows]:
        sample = pkg_resources.resource_filename(__name__, dir_wav_ubm + item)
        data = bob.io.audio.reader(sample)
        # MFCC
        array_mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=True, num_ceps=20)
        lista_mfccs_ubm.append(array_mfcc)

    mfccs_wav_ubm = np.vstack(lista_mfccs_ubm)
    print("MFCC features shape:", mfccs_wav_ubm.shape)
    return mfccs_wav_ubm

##Saving MFCCs to file
def save_mfccs(file, lista):
    with open(file, 'wb') as fp:
        pickle.dump(lista, fp)
    print("MFCCs saved to:", file)


if __name__ == '__main__':

    set_ = 'test'
    observation = ''
    dir_wav_ubm = '../audio/cold/cold.{}/'.format(set_)  # '../audio/wav-non-75-ubm/'
    # dir_wav_non_75 = '../audio/cold/wav-non-75-ubm/'
    # dir_wav_dem_all = '../audio/wav-demencia-all/'
    audio_list_wav_ubm = util.read_files_from_dir(dir_wav_ubm)
    # audio_list_wav_dem = util.read_files_from_dir(dir_wav_dem)
    # audio_list_non_75 = util.read_files_from_dir(dir_wav_non_75)
    # audio_list_bea_and_non75 = audio_list_wav_ubm + audio_list_non_75
    lista_mfccs_ubm = []
    lista_mfccs_ivecs = []
    mfccs_file_ivecs = '../data/mfccs/cold/mfccs_cold_{}_20_{}'.format(set_, observation)
    mfccs_file_ubm = '../data/mfccs/cold/mfccs_ubm_cold_{}_20_{}'.format(set_, observation)

    # Calculating MFCCs
    if set_ == 'traini':
        # Setting the percentage of training data to be used on the UBM
        perc = 20
        rows_wavs_ubm = int(math.modf((len(audio_list_wav_ubm) * perc) / 100)[1])
        rows_wavs_ivecs = int(len(audio_list_wav_ubm) - rows_wavs_ubm)
        # Saving MFCCs for ivecs extraction
        l_mfccs_ivecs = mfcc_ivec_extr(rows_wavs_ivecs)
        save_mfccs(mfccs_file_ivecs, l_mfccs_ivecs)
        gc.collect()
        # Saving MFCCs for UBM
        a_mfccs_ubm = mfcc_ubm(rows_wavs_ubm)
        save_mfccs(mfccs_file_ubm, a_mfccs_ubm)
        gc.collect()

    else:   # (in the case of dev and test sets)
        perc = 100
        rows_wavs_ivecs = int(math.modf((len(audio_list_wav_ubm) * perc) / 100)[1])
        # Saving MFCCs for ivecs extraction
        save_mfccs(mfccs_file_ivecs, mfcc_ivec_extr(rows_wavs_ivecs))
        gc.collect()
