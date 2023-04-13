import os

import librosa
# import bob.io.audio
# import bob.kaldi
# import librosa
# from python_speech_features import mfcc as p_mfcc
import scipy.io.wavfile as wav
from speechbrain.lobes.features import Fbank, MFCC
from speechbrain.dataio.dataio import read_audio
from tqdm import tqdm

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
    mfcc = bob.kaldi.cepstral(data.load()[0], cepstral_type=cepstral_type, delta_order=num_deltas, rate=data.rate,
                              normalization=True,
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
    file_mfccs = out_dir + recipe + '/' + folder_name + '/flevel/{}_{}_{}_{}_{}.{}'.format(cepstral_type, recipe,
                                                                                           num_feats,
                                                                                           folder_name, observation,
                                                                                           cepstral_type)
    print("Extracted {} {} from {} utterances".format(len(list_mfccs), cepstral_type, len(list_wavs)))
    util.save_pickle(file_mfccs, list_mfccs)
    # np.savetxt(file_mfccs_cold, list_mfccs, fmt='%.7f')


def execute_extraction_function(feat_type, **params):
    """Switcher to select a specific feature extraction function
    Args:
        feat_type (string): Type of the frame-level feature to extract from the utterances.
                                Choose from: 'mfcc', 'fbanks', 'spectrogram'.
        waveform (Tensor): Tensor object containing the waveform.
        **params: Parameters belonging to the corresponding feature extraction function.
    """
    switcher = {
        'mfcc': lambda: MFCC(**params),
        'fbank': lambda: Fbank(**params),
    }
    return switcher.get(feat_type, lambda: "Error, feature extraction function {} not supported!".format(feat_type))()


def speechbrain_flevel(list_wavs, out_dir, recipe, dataset_folder, cepstral_type, **params):
    print("Extracting {} for {} wavs in: {}".format(cepstral_type, len(list_wavs), dataset_folder))
    # Output details
    observation = 'Deltas{}'.format(str(params['deltas']))

    # parent_dir = os.path.basename(os.path.dirname(list_wavs[0]))
    # if not os.path.isdir(out_dir + recipe):
    #     os.mkdir(out_dir + recipe)
    # if not os.path.isdir(out_dir + recipe + '/' + dataset_folder):
    #     os.mkdir(out_dir + recipe + '/' + dataset_folder)
    if not os.path.isdir(out_dir + recipe + '/' + dataset_folder + '/' + cepstral_type + '/' + observation):
        os.makedirs(out_dir + recipe + '/' + dataset_folder + '/' + cepstral_type + '/' + observation)

    # ---Calculating and saving flevel feats---
    feature_maker = execute_extraction_function(feat_type=cepstral_type, **params)
    list_feats = []
    for wav in (pbar := tqdm(list_wavs, desc="Extracting {} features".format(cepstral_type), position=0)):
        basename = os.path.basename(wav).split('.')[0]
        pbar.set_description("Processing utterance {}...".format(basename))
        signal = read_audio(wav).unsqueeze(0)
        feats = feature_maker(signal)
        # list_feats.append(feats)
        np_feats = feats.squeeze(0).numpy()
        if cepstral_type == 'mfcc':
            file_feats = out_dir + recipe + '/' + dataset_folder + '/{1}/{2}/{0}_{1}_{3}.{1}'.format(str(params['n_mfcc']),
                                                                                                 cepstral_type,
                                                                                                 observation, basename)
        elif cepstral_type == 'fbank':
            file_feats = out_dir + recipe + '/' + dataset_folder + '/{1}/{2}/{0}{1}_{3}.{1}'.format('',
                                                                                                 cepstral_type,
                                                                                                 observation, basename)
        util.save_pickle(file_feats, np_feats)
        # pbar.set_description("Data pickled to file: {}. With shape: {}".format(file_feats, np_feats.shape))

        # np.savetxt(file_mfccs_cold, list_mfccs, fmt='%.7f')
