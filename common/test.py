import matplotlib.pyplot as plt
from itertools import repeat

from librosa import display
import librosa

import numpy as np
from scipy import signal
from scipy.io import wavfile
import argparse
import librosa
from specAugment import spec_augment_tensorflow
import os, sys
import numpy as np
import matplotlib.pyplot as plt


# y_labels of dem speaker to pandas Dataframe 
def augment_alz_labels():
    lista = np.loadtxt('../classifiers/cross_val/labels-75.txt', delimiter=',', dtype='str').tolist()
    contador = 0
    new = []
    var = [x for item in lista for x in repeat(item, 5)]
    np.savetxt('/opt/project/data/ids_labels_375.txt', var, delimiter=',', fmt='%s')


# augment_alz_labels()


audio, sampling_rate = librosa.load('../audio/wav_anon_75_225/001A_szurke.wav')
mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                 sr=sampling_rate,
                                                 n_mels=256,
                                                 hop_length=128,
                                                 fmax=8000)
# reshape spectrogram shape to [batch_size, time, frequency, 1]
shape = mel_spectrogram.shape
mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

# Show Raw mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :, 0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.title("title1")
plt.tight_layout()
#plt.show()

