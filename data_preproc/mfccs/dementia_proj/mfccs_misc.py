import pickle

import bob.io.audio
import bob.kaldi
#import librosa
from python_speech_features import mfcc as p_mfcc
from common import util


#  Getting MFCCs from wavs
def compute_mfccs_librosa(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        y, sr = librosa.load(path + item, sr=16000)
        data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # print(item, data.shape)
        list_mfccs.append(data)
    return list_mfccs


def compute_mfccs_bkaldi(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        data = bob.io.audio.reader(path + item)
        mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False, num_ceps=13) #bob.kaldi.cepstral(data.load()[0], cepstral_type="mfcc", delta_order=0, rate=data.rate, normalization=False, num_ceps=20)
        list_mfccs.append(mfcc)
    return list_mfccs


# With python speech features
def compute_mfccs_psf(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        y, sr = librosa.load(path + item, sr=16000)
        mfcc = p_mfcc(y, sr, nfilt=13)
        list_mfccs.append(mfcc)
    return list_mfccs


# Read just original 75 speakers
def just_original_75():
    lines = open("../data/wavlista-anon-75-225.txt").read().splitlines()
    wavlista_anon_75_225 = []
    for it in lines:
        wav_file = '{}.wav'.format(it)
        wavlista_anon_75_225.append(wav_file)
    return wavlista_anon_75_225


def main():
    work_dir = '/home/jose/PycharmProjects/the_speech'
    audio_dir = 'C:/Users/Win10/Documents/'

    # Input files
    dir_wav_ubm ='../audio/wav-bea-diktafon'
    dir_anon_75 = '../audio/wav_anon_75_225'
    audio_list_ubm = util.read_files_from_dir(dir_wav_ubm)  # Reading BEA files
    audio_list_dementia_aug = util.read_files_from_dir(dir_anon_75)  # Reading augmented dementia files

    # Output files
    observation = '2del'
    file_mfccs_dem = '../data/mfccs/alzheimer/mfccs_dem_13_{}'.format(observation)
    file_mfccs_dem_aug = '../data/mfccs/alzheimer/mfccs_dem_13_aug_{}'.format(observation)
    file_mfccs_ubm = '../data/mfccs/alzheimer/mfccs_ubm_dem_13_{}'.format(observation)

    # ---Calculating and saving MFCCs---
    # for original audios
    util.save_pickle(file_mfccs_dem_aug, compute_mfccs_bkaldi(dir_anon_75, audio_list_dementia_aug))
    # for augmented audios
    util.save_pickle(file_mfccs_dem, compute_mfccs_bkaldi(dir_anon_75, just_original_75()))
    # for UBM (BEA diktafon)
    util.save_pickle(file_mfccs_ubm, compute_mfccs_bkaldi(dir_wav_ubm, audio_list_ubm))

#if __name__ == '__main__':
#    main()
