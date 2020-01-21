import pkg_resources
import numpy as np
import bob.io.audio
import bob.io.base.test_utils
import gc
from common import util
import pickle
import math

import bob.kaldi



"""
lines2 = open("../data/wavlista-anon-non75-szurke.txt").read().splitlines()
wavlista_non75_szur = []
for it2 in lines2:
    wav_file2 = '{}.wav'.format(it2)
    wavlista_non75_szur.append(wav_file2)
"""

#  Getting MFCCs from wavs


#MFCCs for i-vectors extraction (alzheimer's)
def mfcc_ivec_extr(rows):
    print("Calculating MFCCs for i-vecs extractor in: " + dir_wav_dem_all + " with " + str(rows) + "wavs")
    for item2 in wavlista_anon_75_225[-rows:]:
       # sample2 = pkg_resources.resource_filename(__name__, dir_wav_dem_all + item2)
        data2 = bob.io.audio.reader(dir_wav_dem_all + item2)
        # MFCC
        array_mfcc2 = bob.kaldi.cepstral(data2.load()[0], cepstral_type="mfcc", delta_order=1, rate=data2.rate,
                                         normalization=False, num_ceps=13)
        lista_mfccs_ivecs.append(array_mfcc2)

    #mfccs_wav_ivec = np.vstack(lista_mfccs_ivecs)
    print("MFCCs list length:", len(lista_mfccs_ivecs))
    return lista_mfccs_ivecs


#MFCCs for UBM training
def mfcc_ubm(rows):
    print("Calculating MFCCs for UBM in:" + dir_wav_ubm + " with " + str(rows) + "wavs")
    for item in audio_list_wav_ubm[:rows]:
      #  sample = pkg_resources.resource_filename(__name__, dir_wav_ubm + item)
        data = bob.io.audio.reader(dir_wav_ubm + item)
        # MFCC
        array_mfcc = bob.kaldi.cepstral(data.load()[0], cepstral_type="mfcc", delta_order=1, rate=data.rate,
                                        normalization=False, num_ceps=13)
        lista_mfccs_ubm.append(array_mfcc)

    mfccs_wav_ubm = np.vstack(lista_mfccs_ubm)
    print("MFCC features shape:", mfccs_wav_ubm.shape)
    return lista_mfccs_ubm


# Saving MFCCs to file
def save_mfccs(file, lista):
    with open(file, 'wb') as fp:
        pickle.dump(lista, fp)
    print("MFCCs saved to:", file)


if __name__ == '__main__':

    set_ = 'train'
    observation = 'one_delta'
    dir_wav_ubm = '/home/egasj/PycharmProjects/iVectorsBob/audio/wav-bea-diktafon/'
    dir_wav_dem_all = '/home/egasj/PycharmProjects/iVectorsBob/audio/wav-demencia-all/'
    audio_list_wav_ubm = util.read_files_from_dir(dir_wav_ubm)
    # audio_list_wav_dem = util.read_files_from_dir(dir_wav_dem)
    # audio_list_non_75 = util.read_files_from_dir(dir_wav_non_75)
    # audio_list_bea_and_non75 = audio_list_wav_ubm + audio_list_non_75
    lista_mfccs_ubm = []
    lista_mfccs_ivecs = []
    mfccs_file_ivecs = '/home/egasj/PycharmProjects/iVectorsBob/data/hc/mfccs_dem_{}_13'.format(observation)
    mfccs_file_ubm = '/home/egasj/PycharmProjects/iVectorsBob/data/hc/mfccs_ubm_dem_{}_13'.format(observation)

    lines = open("/home/egasj/PycharmProjects/iVectorsBob/data/wavlista-anon-75-225.txt").read().splitlines()
    wavlista_anon_75_225 = []
    for it in lines:
        wav_file = '{}.wav'.format(it)
        wavlista_anon_75_225.append(wav_file)


    # Calculating MFCCs
    if set_ == 'traini':
        # Setting the percentage of training data to be used on the UBM
        perc = 100
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
        rows_wavs_ivecs = int(math.modf((len(wavlista_anon_75_225) * perc) / 100)[1])
        # Saving MFCCs for ivecs/fisher extraction
        save_mfccs(mfccs_file_ivecs, mfcc_ivec_extr(rows_wavs_ivecs))
        gc.collect()
        rows_wavs_ubm = int(math.modf((len(audio_list_wav_ubm) * perc) / 100)[1])
        # Saving MFCCs for UBM
        save_mfccs(mfccs_file_ubm, mfcc_ubm(rows_wavs_ubm))


'''
perc = 100
rows_wavs_ubm = int(math.modf((len(audio_list_wav_ubm) * perc) / 100)[1])
# Saving MFCCs for ivecs extraction
save_mfccs(mfccs_file_ubm, mfcc_ubm(rows_wavs_ubm))
'''