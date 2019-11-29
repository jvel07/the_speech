import argparse
import librosa
from specAugment import spec_augment_tensorflow
import os, sys
import numpy as np

audio, sampling_rate = librosa.load('C:/Users/Win10/PycharmProjects/the_speech/data_preproc/data_augmentation/61-70968-0002.wav')
mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                 sr=16000,
                                                 n_mels=256,
                                                 hop_length=128,
                                                 fmax=8000)

# reshape spectrogram shape to [batch_size, time, frequency, 1]
shape = mel_spectrogram.shape
mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

# Show Raw mel-spectrogram
spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
                                                  title="Raw Mel Spectrogram")