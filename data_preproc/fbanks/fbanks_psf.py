#import bob.io.audio
#import bob.io.base.test_utils
import librosa
import numpy as np
import python_speech_features
import scipy.io.wavfile as wav

from common import util


def compute_fbanks(dir_, audio_list):
    print('Computing fbanks on:', dir_, '\nNumber of items to compute:', len(audio_list))
    lista_fbanks = []
    for item2 in audio_list:
        sr, data2 = wav.read(dir_+item2)#librosa.load(dir_+item2, sr=16000)
        fbank = python_speech_features.fbank(signal=data2, samplerate=sr, nfilt=40)
        lista_fbanks.append(fbank)

    print("Fbanks list length:", len(lista_fbanks))
    return lista_fbanks


# For using them as arrays (x, num_filters+ 1 energy coef.)
def concatenate_fbanks(fbanks):
    print("Concatenating fbanks...")
    list_fbanks = []
    for item1, item2 in fbanks:
        item2 = item2.reshape(-1, 1)
        conc = np.concatenate((item1, item2), axis=1)
        list_fbanks.append(np.float32(conc))
    # util.save_pickle(file_name, list_fbanks)
    return list_fbanks


#def main():
if __name__ == '__main__':
    work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'

    dir_wav_ubm = work_dir + '/audio/wav-bea-diktafon/'
    dir_wav_dem_all = work_dir + '/audio/wav-demencia-all/'
    dir_wav_anon_75 = work_dir + '/audio/wav_anon_75_225/'

    audio_list_wav_ubm = util.read_files_from_dir(dir_wav_ubm)
    audio_list_wav_dem = util.read_files_from_dir(dir_wav_anon_75)
    # audio_list_non_75 = util.read_files_from_dir(dir_wav_non_75)
    # audio_list_bea_and_non75 = audio_list_wav_ubm + audio_list_non_75

    observation = '40_aug'
    file_fbanks = work_dir+'/data/fbanks/fbanks_dem_{}.npy'.format(observation)
    file_fbanks_bea = work_dir+'/data/fbanks/fbanks_bea_dem_{}'.format(observation)

    # Computing and concatenating fbanks
    #fbanks_dem = compute_fbanks(dir_wav_anon_75, audio_list_wav_dem)
    #list_fbanks_dem = concatenate_fbanks(fbanks_dem)

    fbanks_bea = compute_fbanks(dir_wav_ubm, audio_list_wav_ubm)
    arr_fbanks_bea = np.vstack(concatenate_fbanks(fbanks_bea))

    # Saving fbanks
    #util.save_pickle(file_fbanks_bea, arr_fbanks_bea)
    #util.pickle_dump_big(list_fbanks_dem, file_fbanks)


#if __name__ == '__main__':
 #   main()
