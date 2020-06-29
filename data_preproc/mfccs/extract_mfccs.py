import os

import bob.io.audio
import bob.kaldi
# import librosa
# from python_speech_features import mfcc as p_mfcc
import scipy.io.wavfile as wav

from common import util
import numpy as np




#  Getting MFCCs from wavs
def mfccs_librosa(path, audio_list, num_feats):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        # y, sr = librosa.load(path + item, sr=16000)
        sr, y = wav.read(path + item)
        # data = librosa.feature.mfcc(y=np.float32(y), sr=16000, n_mfcc=num_feats)
        # print(item, data.shape)
        list_mfccs.append(1)
    return list_mfccs


def cepstral_bkaldi(signal, num_feats, num_deltas, cepstral_type="mfcc", raw_energy=False, num_mel_bins=23, low_freq=20,
                    high_freq=0):
    # print('Computing MFCCs on:', path, '\nNumber of files to process:', len(audio_list))
    data = bob.io.audio.reader(signal)
    # mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=True, num_ceps=num_feats, snip_edges=True)
    mfcc = bob.kaldi.cepstral(data.load()[0], cepstral_type=cepstral_type, delta_order=num_deltas, rate=data.rate, normalization=True,
                              num_ceps=num_feats, snip_edges=True, raw_energy=False, num_mel_bins=num_mel_bins,
                              low_freq=low_freq, high_freq=high_freq)
    return mfcc


# With python speech features
def mfccs_psf(path, audio_list, num_feats):
    list_mfccs = []
    print('Computing MFCCs on:', path, '\nNumber of files to process:', len(audio_list))
    for item in audio_list:
        (rate, y) = wav.read(path + item)
        # y, sr = librosa.load(, sr=16000)
        # mfcc = p_mfcc(y, 16000, numcep=num_feats)
        list_mfccs.append(1)
    return list_mfccs


def compute_flevel_feats(list_wavs, out_dir, obs, num_feats, num_deltas, recipe, folder_name, cepstral_type="mfcc",
                         raw_energy=True, num_mel_bins=23, low_freq=20, high_freq=0):
    print("Extracting {} for {} wavs in: {}".format(cepstral_type, len(list_wavs), folder_name))
    # Output details
    observation = '{}del{}'.format(num_deltas, obs)

    # parent_dir = os.path.basename(os.path.dirname(list_wavs[0]))
    if not os.path.isdir(out_dir + recipe):
        os.mkdir(out_dir + recipe)
    if not os.path.isdir(out_dir + recipe + '/' + folder_name):
        os.mkdir(out_dir + recipe + '/' + folder_name)

    # ---Calculating and saving MFCCs---
    list_mfccs = []
    for wav in list_wavs:
        mfcc = cepstral_bkaldi(wav, num_feats, num_deltas, cepstral_type=cepstral_type,
                               raw_energy=raw_energy, num_mel_bins=num_mel_bins, low_freq=low_freq, high_freq=high_freq)
        list_mfccs.append(mfcc)
    file_mfccs_cold = out_dir + recipe + '/' + folder_name + '/{}_{}_{}_{}_{}.{}'.format(cepstral_type, recipe, num_feats,
                                                                                           folder_name, observation,
                                                                                           cepstral_type)
    print("Extracted {} {} from {} utterances".format(len(list_mfccs), cepstral_type, len(list_wavs)))
    util.save_pickle(file_mfccs_cold, list_mfccs)
    #np.savetxt(file_mfccs_cold, list_mfccs, fmt='%.7f')
