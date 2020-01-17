import bob.io.audio
import bob.kaldi
import librosa
from python_speech_features import mfcc as p_mfcc
import scipy.io.wavfile as wav

from common import util
import numpy as np
from data_preproc.mfccs import psf_deltas_proc


#  Getting MFCCs from wavs
def compute_mfccs_librosa(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        # y, sr = librosa.load(path + item, sr=16000)
        sr, y = wav.read(path + item)
        data = librosa.feature.mfcc(y=np.float32(y), sr=16000, n_mfcc=20)
        # print(item, data.shape)
        list_mfccs.append(data)
    return list_mfccs


def compute_mfccs_bkaldi(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path, '\nNumber of files to process:', len(audio_list))
    for item in audio_list:
        data = bob.io.audio.reader(path + item)
        mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False,
                              num_ceps=20)  # bob.kaldi.cepstral(data.load()[0], cepstral_type="mfcc", delta_order=0, rate=data.rate, normalization=False, num_ceps=20)
        list_mfccs.append(mfcc)
    return list_mfccs


# With python speech features
def compute_mfccs_psf(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path, '\nNumber of files to process:', len(audio_list))
    for item in audio_list:
        (rate, y) = wav.read(path + item)
        # y, sr = librosa.load(, sr=16000)
        mfcc = p_mfcc(y, 16000, numcep=20)
        list_mfccs.append(mfcc)
    return list_mfccs


# def main():
if __name__ == '__main__':

    work_dir = '/opt/project'  # C:/Users/Win10/PycharmProjects/the_speech'
    list_sets = ["train", "dev", "test"]

    # Input files
    # dir_wav_ubm = work_dir + '/audio/-bea-diktafon/'
    # audio_list_ubm = util.read_files_from_dir(dir_wav_ubm)  # Reading BEA files

    # Output files
    observation = '2del'
    num_mfccs = 20
    # file_mfccs_ubm = work_dir + '/data/mfccs/cold/mfccs_ubm_dem_{}_{}'.format(num_mfccs, observation)

    # ---Calculating and saving MFCCs---
    # for original audios
    for name in list_sets:
        dir_wavs = work_dir + '/audio/{}/'.format(name)
        audio_list = util.read_files_from_dir(dir_wavs)  # Reading wav files
        file_mfccs_cold = work_dir + '/data/mfccs/cold/mfccs_cold_{}_{}_{}'.format(num_mfccs, name, observation)
        util.save_pickle(file_mfccs_cold, compute_mfccs_bkaldi(dir_wavs, audio_list))

    # for UBM
    # util.save_pickle(file_mfccs_ubm, compute_mfccs_bkaldi())

# if __name__ == '__main__':
#    main()
