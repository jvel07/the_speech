#import bob.io.audio
import os
import random

import librosa
import numpy as np
import scipy


# Audio data augmentation


def load_audio_file(file_path):
    data, sr = librosa.load(file_path, sr=16000)
    return data


def load_audio_bob(file):
    data = bob.io.audio.reader(file)
    return data.load()[0]


def save_wav(file_name, rate, data):  # /home/egasj/PycharmProjects/iVectorsBob/audio/wav-demencia-all/001A_szurke.wav'
    scipy.io.wavfile.write(file_name, rate, data)


working_dir= 'C:/Users/Win10/PycharmProjects/the_speech'


def reading_anon75():
    # Reading list of anon 75-225 wav files
    lines = open(working_dir + "/data/wavlista-anon-75-225.txt").read().splitlines()
    wavlista_anon_75_225 = []
    for it in lines:
        wav_file = '{}.wav'.format(it)
        wavlista_anon_75_225.append(wav_file)
    return wavlista_anon_75_225


def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


# File must have a list of wav names (specific to the case of dementia)
def add_noise_to_anon75(noise_factor=0.017):
    # Reading list of anon 75-225 wav files
    dir_ = working_dir + '/audio/wav_anon_75_225/'
    list_audios = reading_anon75()
    for item2 in list_audios:
        data2 = load_audio_file(dir_ + item2)
        aug = add_noise(data2, noise_factor=noise_factor)
        scipy.io.wavfile.write(dir_ + os.path.splitext(os.path.basename(dir_+item2))[0] + '_noised017.wav', 16000, aug)
    print("Augmented with noise!")


add_noise_to_anon75()


def change_pitch_anon75():
    list_audios = reading_anon75()
    dir_ = working_dir + '/audio/wav_anon_75_225/'
    for item2 in list_audios:
        #data2 = bob.io.audio.reader(dir_ + item2)
        data2 = load_audio_file(dir_ + item2)
        aug = librosa.effects.pitch_shift(data2, 16000, )
        #aug = pyrubberband.pyrb.pitch_shift(data2, 16000, 4) #random.uniform(-4, 4))
        scipy.io.wavfile.write(dir_ + os.path.splitext(os.path.basename(dir_+item2))[0] + '_pitched_p4.wav', 16000, aug)
    print("Augmented with pitch!")


change_pitch_anon75()


def change_speed_anon75():
    list_audios = reading_anon75()
    dir_ = working_dir + '/audio/wav_anon_75_225/'
    for item2 in list_audios:
        data2 = load_audio_file(dir_ + item2)
        aug = librosa.effects.time_stretch(data2, random.uniform(0, 1))
        #aug = pyrubberband.pyrb.time_stretch(data2, 2.0)
        scipy.io.wavfile.write(dir_ + os.path.splitext(os.path.basename(dir_+item2))[0] + '_stretched.wav', 16000, aug)
    print("Augmented with stretch!")


change_speed_anon75()


def shift_time(data):
    start_ = int(np.random.uniform(-4800, 4800))
    if start_ >= 0:
        wav_time_shift = np.r_[data[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), data[:start_]]
    return wav_time_shift


def shifting_anon75():
    list_audios = reading_anon75()
    dir_ = working_dir + '/audio/wav_anon_75_225/'
    for item2 in list_audios:
        data2 = load_audio_file(dir_ + item2)
        aug = shift_time(data2)
        scipy.io.wavfile.write(dir_ + os.path.splitext(os.path.basename(dir_+item2))[0] + '_shifted.wav', 16000, aug)
    print("Augmented with shift!")




