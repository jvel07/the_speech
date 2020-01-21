import bob.kaldi
import pkg_resources
import numpy as np
import bob.io.audio
import bob.io.base.test_utils
import gc
from common2 import util
import pickle
import math


def read_75_speakers():
    lines = open("../data/wavlista-anon-75-225.txt").read().splitlines()
    wavlista_anon_75_225 = []
    for it in lines:
        wav_file = '{}.wav'.format(it)
        wavlista_anon_75_225.append(wav_file)
    return wavlista_anon_75_225


def read_non75_szurke():
    lines2 = open("../data/wavlista-anon-non75-szurke.txt").read().splitlines()
    wavlista_non75_szur = []
    for it2 in lines2:
        wav_file2 = '{}.wav'.format(it2)
        wavlista_non75_szur.append(wav_file2)
    return wavlista_non75_szur


# VAD for i-vectors extraction
def vad_ivec_extr(rows):
    print("Calculating VAD...")
    wavlista_anon_75_225 = read_75_speakers()
    for item2 in wavlista_anon_75_225[-rows:]:
        sample2 = pkg_resources.resource_filename(__name__, dir_wav_ubm + item2)
        data2 = bob.io.audio.reader(sample2)
        # MFCC
        array_mfcc2 = bob.kaldi.compute_dnn_vad(data2.load()[0],
                                                data2.rate)  # bob.kaldi.mfcc(data2.load()[0], data2.rate, normalization=False, num_ceps=20)
        lista_vad_ivecs.append(array_mfcc2)

    # mfccs_wav_ivec = np.vstack(lista_mfccs_ivecs)
    # print("MFCCs list length:", len(lista_vad_ivecs))
    return lista_vad_ivecs


# VAD for UBM training
def vad_ubm(rows):
    print("Calculating VAD for UBM in:" + dir_wav_ubm + " with " + str(rows) + "wavs")
    for item in audio_list_wav_ubm[:rows]:
        sample = pkg_resources.resource_filename(__name__, dir_wav_ubm + item)
        data = bob.io.audio.reader(sample)
        # MFCC
        array_mfcc = bob.kaldi.compute_dnn_vad(data.load()[0], data.rate)
        lista_vad_ubm.append(array_mfcc)

    # vad_wav_ubm = np.vstack(lista_vad_ubm)
    # print("MFCC features shape:", vad_wav_ubm.shape)
    return lista_vad_ubm


##Saving MFCCs to file
def save_vads(file, lista):
    with open(file, 'wb') as fp:
        pickle.dump(lista, fp)
    print("VAD saved to:", file)


##  Activity detection
#  Activity detection (requires hc and vads in the form of nparrays)
def remove_non_active_segments(mfccs, vads, file):
    new = []
    for m1 in mfccs:
        for v1 in vads:
            if len(m1) == len(v1):
                m2 = m1[np.ravel(v1) == 1]
                new.append(m2)
    util.save_pickle(file, new)
    return new


def remove_non_active_segments2(mfccs, vads, file):
    new = []
    for num, m1 in enumerate(mfccs, start=0):
        vad = vads[num]
        m2 = m1[np.ravel(vad) == 1]
        new.append(m2)
    # util.save_pickle(file, new)
    util.save_pickle(file, new)
    return new


if __name__ == '__main__':

    set_ = 'traind'
    observation = 'whole'
    dir_wav_ubm = '../audio/wav-bea-diktafon/'
    dir_wav_non_75 = '../audio/wav-non-75-ubm/'
    # dir_wav_dem_all = '../audio/wav-demencia-all/'
    audio_list_wav_ubm = util.read_files_from_dir(dir_wav_ubm)
    # audio_list_wav_dem = util.read_files_from_dir(dir_wav_dem)
    # audio_list_non_75 = util.read_files_from_dir(dir_wav_non_75)
    # audio_list_bea_and_non75 = audio_list_wav_ubm + audio_list_non_75
    lista_vad_ubm = []
    lista_vad_ivecs = []
    vad_file_ivecs = '../data/vad/vad_anon-75-225_{}_{}'.format(set_, observation)
    vad_file_ubm = '../data/vad/vad_ubm_dem_{}_{}'.format(set_, observation)

    # Calculating VAD
    if set_ == 'train':
        # Setting the percentage of training data to be used on the UBM
        perc = 20
        rows_wavs_ubm = int(math.modf((len(audio_list_wav_ubm) * perc) / 100)[1])
        rows_wavs_ivecs = int(len(audio_list_wav_ubm) - rows_wavs_ubm)
        # Saving VAD for ivecs extraction
        l_vad_ivecs = vad_ivec_extr(rows_wavs_ivecs)
        save_vads(vad_file_ivecs, l_vad_ivecs)
        gc.collect()
        # Saving VAD for UBM
        a_vad_ubm = vad_ubm(rows_wavs_ubm)
        save_vads(vad_file_ubm, a_vad_ubm)
        gc.collect()

    else:  # (in the case of dev and test sets)
        perc = 100
        rows_wavs_ubm = int(math.modf((len(audio_list_wav_ubm) * perc) / 100)[1])
        rows_wavs_ivecs = int(math.modf((len(read_75_speakers()) * perc) / 100)[1])
        # Saving MFCCs for ivecs extraction
        save_vads(vad_file_ubm, vad_ubm(rows_wavs_ubm))
        gc.collect()
