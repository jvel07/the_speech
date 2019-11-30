import librosa
from common import util


#  Getting MFCCs from wavs
def compute_mspect_librosa(path, audio_list):
    list_mfccs = []
    print('Computing Mel-Specs on:', path, '\nNumber of files here:', len(audio_list))
    for item in audio_list:
        y, sr = librosa.load(path + item, sr=16000)
        data = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=256, hop_length=128,
                                              fmax=8000, fmin=-80)
        list_mfccs.append(data)
    return list_mfccs


def main():
    work_dir = '/Users/jose/PycharmProjects/the_speech'
    audio_dir = '/Users/jose/PycharmProjects/iVectorsBob'

    # Input files
    dir_wav_ubm = audio_dir + '/audio/wav-bea-diktafon'
    dir_anon_75 = audio_dir + '/audio/wav-dementia-all'

    audio_list_original_dem = util.just_original_75()  # Reading Original dementia files
    audio_list_ubm = util.read_files_from_dir(dir_wav_ubm)  # Reading BEA files

    # Output files
    observation = ''
    n_mels = '256'

    file_melspec_dem = '../data/melspecs/melspec_dem_{}_{}'.format(n_mels, observation)
    file_melspec_ubm = '../data/melspecs/melspec_ubm_dem_{}_{}'.format(n_mels, observation)

    # ---Calculating and saving MFCCs---
    # for original audios
    util.save_pickle(file_melspec_dem, compute_mspect_librosa(dir_anon_75, audio_list_original_dem))
    # for UBM (BEA-diktafon)
    util.save_pickle(file_melspec_ubm, compute_mspect_librosa(dir_wav_ubm, audio_list_ubm))


if __name__ == '__main__':
    main()