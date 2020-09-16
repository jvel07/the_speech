import numpy as np
from common import util
import shutil


# Aiming to read the aibo-labels and group the wavs label-wise (for unsupervised wavegan training)
def group_wavs_into_class_folders():
    # reading the files in train
    list_wavs = util.read_files_from_dir('E:/emotion-aibo/wav16k/aibo.wav.train/')
    # reading the folder containing the files with the labels
    list_files_labels = util.read_files_from_dir_reg('E:/emotion-aibo', '.txt')

    list_wavs_labels = []
    dict_classes = dict.fromkeys({'class1', 'class2', 'class3', 'class4', 'class5'})
    i = 1
    for file in list_files_labels:
        list_wavs = np.loadtxt('E:/emotion-aibo/' + file, delimiter='\n', dtype='str').tolist()
        dict_classes['class{}'.format(i)] = list_wavs
        i += 1

    for key in sorted(dict_classes):
        lista = dict_classes[key]
        for item in lista:
            shutil.copy('E:/emotion-aibo/wav16k/aibo.wav.train/{}.wav'.format(item),
                        'E:/emotion-aibo/{}/'.format(key))
