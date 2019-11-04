#import bob.io.audio
import os
import random

import librosa
import numpy as np
import scipy

# Audio data augmentation
from common import util


def load_audio_file(file_path):
    data, sr = librosa.load(file_path, sr=16000)
    return data


def load_audio_bob(file):
    #data = bob.io.audio.reader(file)
    return 0#data.load()[0]


def save_wav(file_name, rate, data):  # /home/egasj/PycharmProjects/iVectorsBob/audio/wav-demencia-all/001A_szurke.wav'
    scipy.io.wavfile.write(file_name, rate, data)


def reading_anon75():
    # Reading list of anon 75-225 wav files
    lines = open("C:/Users/Win10/PycharmProjects/the_speech/data/wavlista-anon-75-225.txt").read().splitlines()
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
def add_noise_to_anon75(noise_factor):
    # Reading list of anon 75-225 wav files
    dir_ = 'C:/Users/Win10/Documents/audio/audio/wav_anon_75_225/'
    list_audios = reading_anon75()
    for item2 in list_audios:
        data2 = load_audio_file(dir_ + item2)
        aug = add_noise(data2, noise_factor=noise_factor)
        scipy.io.wavfile.write(dir_ + os.path.splitext(os.path.basename(dir_+item2))[0] + '_noised.wav', 16000, aug)


def change_pitch_anon75():
    list_audios = reading_anon75()
    dir_ = 'C:/Users/Win10/Documents/audio/audio/wav_anon_75_225/'
    for item2 in list_audios:
        #data2 = bob.io.audio.reader(dir_ + item2)
        data2 = load_audio_file(dir_ + item2)
        aug = librosa.effects.pitch_shift(data2, 16000, random.randint(-4, 4))
        scipy.io.wavfile.write(dir_ + os.path.splitext(os.path.basename(dir_+item2))[0] + '_pitched.wav', 16000, aug)


def change_speed_anon75():
    list_audios = reading_anon75()
    dir_ = 'C:/Users/Win10/Documents/audio/audio/wav_anon_75_225/'
    for item2 in list_audios:
        data2 = load_audio_file(dir_ + item2)
        aug = librosa.effects.time_stretch(data2, random.randint(1, 3))
        scipy.io.wavfile.write(dir_ + os.path.splitext(os.path.basename(dir_+item2))[0] + '_stretched.wav', 16000, aug)


def shift_time(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def shifting_anon75():
    list_audios = reading_anon75()
    dir_ = 'C:/Users/Win10/Documents/audio/audio/wav_anon_75_225/'
    for item2 in list_audios:
        data2 = load_audio_file(dir_ + item2)
        aug = shift_time(data2, 16000, 16, 'both')
        scipy.io.wavfile.write(dir_ + 'shifted_' + item2, int(data2.rate), aug)
