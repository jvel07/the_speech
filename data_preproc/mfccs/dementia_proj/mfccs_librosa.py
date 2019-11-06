import pickle

import bob.io.audio
import bob.kaldi
import librosa
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
        mfcc = bob.kaldi.cepstral(data.load()[0], cepstral_type="mfcc", delta_order=1, rate=data.rate,
                                  normalization=False, num_ceps=13)
        list_mfccs.append(mfcc)
    return list_mfccs


# Saving MFCCs to file
def save_mfccs(file, lista):
    with open(file, 'wb') as fp:
        pickle.dump(lista, fp)
    print("MFCCs saved to:", file)


if __name__ == '__main__':

    work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'
    audio_dir = 'C:/Users/Win10/PycharmProjects/the_speech/audio'

    # Input files
    dir_wav_ubm = audio_dir + '/wav-bea-diktafon/'
    dir_anon_75 = audio_dir + '/wav_anon_75_225/'
    audio_list_ubm = util.read_files_from_dir(dir_wav_ubm)
    audio_list_dementia = util.read_files_from_dir(dir_anon_75)

    # Output files
    observation = 'aug'
    file_mfccs_dem = work_dir + '/data/mfccs/mfccs_dem_13_{}'.format(observation)
    file_mfccs_ubm = work_dir + '/data/mfccs/mfccs_ubm_dem_13_{}'.format(observation)

    # Calculating and saving MFCCs
    save_mfccs(file_mfccs_dem, compute_mfccs_bkaldi(dir_anon_75, audio_list_dementia))
    save_mfccs(file_mfccs_ubm, compute_mfccs_bkaldi(dir_wav_ubm, audio_list_ubm))
