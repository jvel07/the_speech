import librosa

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


if __name__ == '__main__':
    work_dir = '/Users/jose/PycharmProjects/the_speech'
    audio_dir = '/Users/jose/PycharmProjects/iVectorsBob'

    # Input files
    dir_wav_ubm = audio_dir + '/audio/wav-bea-diktafon/'
    dir_anon_75 = audio_dir + '/audio/wav-demencia-all/'

    audio_list_original_dem = util.just_original_75()  # Reading Original dementia files
    audio_list_ubm = util.read_files_from_dir(dir_wav_ubm)  # Reading BEA files

    # Output files
    observation = ''
    n_mels = '256'

    file_melspec_dem = work_dir + '/data/melspecs/melspec_dem_{}{}'.format(n_mels, observation)
    dataset_dem = 'melspecs_dem_{}'.format(n_mels)
    file_melspec_ubm = work_dir + '/data/melspecs/melspec_ubm_dem_{}{}'.format(n_mels, observation)
    dataset_ubm = 'melspecs_ubm_{}'.format(n_mels)

    # ---Calculating and saving Mel-Specs---
    # for original audios
    specs_dem = compute_mspect_librosa(dir_anon_75, audio_list_original_dem)
    util.pickle_dump(specs_dem, file_melspec_dem)
    #util.save_as_hdf5(file_melspec_dem, dataset_dem, specs_dem)
    # for BEA-diktafon (UBM) audios
    specs_ubm = compute_mspect_librosa(dir_wav_ubm, audio_list_ubm)
    util.save_pickle(file_melspec_ubm, specs_ubm)
