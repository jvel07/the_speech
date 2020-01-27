import os

import bob.io.audio
import bob.kaldi
import librosa
from python_speech_features import mfcc as p_mfcc
import scipy.io.wavfile as wav

from common import util
import numpy as np
from data_preproc.mfccs import psf_deltas_proc


#  Getting MFCCs from wavs
def mfccs_librosa(path, audio_list, num_feats):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        # y, sr = librosa.load(path + item, sr=16000)
        sr, y = wav.read(path + item)
        data = librosa.feature.mfcc(y=np.float32(y), sr=16000, n_mfcc=num_feats)
        # print(item, data.shape)
        list_mfccs.append(data)
    return list_mfccs


def mfccs_bkaldi(signal, num_feats):
    # print('Computing MFCCs on:', path, '\nNumber of files to process:', len(audio_list))
    data = bob.io.audio.reader(signal)
    mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=True, num_ceps=num_feats, snip_edges=True)
    # mfcc = bob.kaldi.cepstral(data.load()[0], cepstral_type="mfcc", delta_order=0, rate=data.rate, normalization=False, num_ceps=20)
    return mfcc


# With python speech features
def mfccs_psf(path, audio_list, num_feats):
    list_mfccs = []
    print('Computing MFCCs on:', path, '\nNumber of files to process:', len(audio_list))
    for item in audio_list:
        (rate, y) = wav.read(path + item)
        # y, sr = librosa.load(, sr=16000)
        mfcc = p_mfcc(y, 16000, numcep=num_feats)
        list_mfccs.append(mfcc)
    return list_mfccs


def compute_mfccs(list_wavs, out_dir, num_mfccs, recipe, folder_name):
    print("Extracting MFFCs for {} wavs in: {}".format(len(list_wavs), folder_name))
    # Output details
    observation = '2del'
    num_mfccs = num_mfccs

    # ---Calculating and saving MFCCs---
    list_mfccs = []
    for wav in list_wavs:
        mfcc = mfccs_bkaldi(wav, num_mfccs)
        list_mfccs.append(mfcc)
        # parent_dir = os.path.basename(os.path.dirname(list_wavs[0]))
        if not os.path.isdir(out_dir + recipe + '/' + folder_name):
            os.mkdir(out_dir + recipe + '/' + folder_name)

    file_mfccs_cold = out_dir + recipe + '/' + folder_name + '/mfccs_{}_{}_{}_{}.mfcc'.format(recipe,
                                                                                              num_mfccs,
                                                                                              folder_name,
                                                                                              observation)
    print("Extracted {} mfccs from {} utterances".format(len(list_mfccs), len(list_wavs)))
    util.save_pickle(file_mfccs_cold, list_mfccs)
