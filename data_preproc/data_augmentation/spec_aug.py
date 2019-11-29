import argparse
import librosa
import scipy
from specAugment import spec_augment_tensorflow
import os, sys
import numpy as np
import copy





audio, sampling_rate = librosa.load('/home/egasj/PycharmProjects/the_speech/audio/wav_anon_75_225/001A_szurke.wav')
mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                 sr=16000,
                                                 n_mels=256,
                                                 hop_length=128,
                                                 fmax=8000)

warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram)


a = librosa.feature.inverse.mel_to_audio(warped_masked_spectrogram)

scipy.io.wavfile.write('test.wav', 16000, a)