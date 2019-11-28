import matplotlib.pyplot as plt
from itertools import repeat

from librosa import display
import librosa

import numpy as np
from scipy import signal
from scipy.io import wavfile


# y_labels of dem speaker to pandas Dataframe 
def augment_alz_labels():
    lista = np.loadtxt('../classifiers/cross_val/labels-75.txt', delimiter=',', dtype='str').tolist()
    contador = 0
    new = []
    var = [x for item in lista for x in repeat(item, 5)]
    np.savetxt('/opt/project/data/ids_labels_375.txt', var, delimiter=',', fmt='%s')


# augment_alz_labels()

def aud_to_spec(file):
    x, sr = librosa.load(file, sr=16000)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


aud_to_spec('../audio/wav_anon_75_225/001A_szurke.wav')

