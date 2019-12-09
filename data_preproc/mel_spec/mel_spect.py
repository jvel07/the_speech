import re

import librosa
import numpy as np

from common import util


#  Getting Mel-Spectrogram from wavs
def compute_mspect_librosa(path, audio_list):
    list_melspecs = []
    print('Computing Mel-Specs on:', path, '\nNumber of items to compute:', len(audio_list))
    for item in audio_list:
        y, sr = librosa.load(path + item, sr=16000)
        data = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=256, hop_length=128,
                                              fmax=8000)
        list_melspecs.append(data)
    return list_melspecs


def preprocess_audio_mel_T(audio, sample_rate=16000, window_size=20,  # log_specgram
                           step_size=10, eps=1e-10):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=320)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40

    return mel_db.T


input_length = 16000*7
def load_audio_file(file_path, audio_list, input_length=input_length):
    print('Computing Mel-Specs on:', file_path, '\nNumber of items to compute:', len(audio_list))
    list_melspecs = []
    for item in audio_list:
        data = librosa.core.load(file_path+item, sr=16000)[0]
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0

            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        list_melspecs.append(preprocess_audio_mel_T(data))

    return list_melspecs


if __name__ == '__main__':
    work_dir = '/opt/project'
    audio_dir = '/opt/project'

    # Input files
    dir_wav_ubm = audio_dir + '/audio/wav-bea-diktafon/'
    dir_anon_75 = audio_dir + '/audio/wav_anon_75_225/'

    audio_list_original_dem = util.just_original_75()  # Reading Original dementia files
    audio_list_ubm = util.read_files_from_dir(dir_wav_ubm)  # Reading BEA files
    audio_list_augmented = util.read_files_from_dir(dir_anon_75)  # Reading augmented files

    # Output files
    observation = ''
    n_mels = '256'
    file_melspec_dem = work_dir + '/data/melspecs/melspec_dem_{}{}'.format(n_mels, observation)
    #file_melspec_ubm = work_dir + '/data/melspecs/melspec_ubm_dem_{}{}'.format(n_mels, observation)
    file_melspec_augmented = work_dir + '/data/melspecs/melspec_ubm_dem_{}_aug{}'.format(n_mels, observation)

    lista = load_audio_file(dir_anon_75, util.just_original_75(), input_length)
    util.save_pickle(work_dir+'/data/melnmffc_dem', lista)

    # ---Calculating and saving Mel-Specs---
    # for original audios
    #specs_dem = compute_mspect_librosa(dir_anon_75, util.just_original_75())
    #util.pickle_dump_big(specs_dem, file_melspec_dem)
    # for augmented audios
    #specs_augmented = compute_mspect_librosa(dir_anon_75, audio_list_augmented)
    #util.pickle_dump_big(specs_augmented, file_melspec_augmented)
    # for BEA-diktafon (UBM) audios
    #specs_ubm = compute_mspect_librosa(dir_wav_ubm, audio_list_ubm)
    #util.save_pickle(file_melspec_ubm, specs_ubm)
