import pickle

import bob.io.audio
import bob.kaldi
import librosa
from python_speech_features import mfcc as p_mfcc
from common import util


#  Getting MFCCs from wavs
def compute_mfccs_librosa(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        y, sr = librosa.load(path + item, sr=16000)
        data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        #print(item, data.shape)
        list_mfccs.append(data)
    return list_mfccs


def compute_mfccs_bkaldi(path, audio_list):
    list_mfccs = []
    print('Computing MFCCs on:', path)
    for item in audio_list:
        data = bob.io.audio.reader(path + item)
        mfcc = bob.kaldi.cepstral(data.load()[0], cepstral_type="mfcc", delta_order=2, rate=data.rate,
                                  normalization=False, num_ceps=13)
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


def main():
    work_dir = '/home/jose/PycharmProjects/the_speech'
    audio_dir = 'D:/VHD/audio'

    # Input files
    dir_wav_ubm = work_dir + '/audio/wav-bea-diktafon/'
    dir_anon_75 = work_dir + '/audio/wav_anon_75_225/'
    audio_list_ubm = util.read_files_from_dir(dir_wav_ubm)
    audio_list_dementia = util.read_files_from_dir(dir_anon_75)

    # Output files
    observation = 'aug_2del'
    file_mfccs_dem = work_dir + '/data/mfccs/dem/mfccs_dem_13_{}'.format(observation)
    file_mfccs_ubm = work_dir + '/data/mfccs/dem/mfccs_ubm_dem_13_{}'.format(observation)

    # Calculating and saving MFCCs
    util.save_pickle(file_mfccs_dem, compute_mfccs_bkaldi(dir_anon_75, audio_list_dementia))
    util.save_pickle(file_mfccs_ubm, compute_mfccs_bkaldi(dir_wav_ubm, audio_list_ubm))


if __name__ == '__main__':
    main()
