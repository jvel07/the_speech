import numpy as np
import pandas as pd
import os
from common import util


from sklearn import preprocessing


# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/data/pcgita/' # windows machine
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/data/pcgita/'  # ubuntu machine
# work_dir2 = 'D:/VHD'


# Encoding labels to numbers
def encode_labels(_y):
    le = preprocessing.LabelEncoder()
    le.fit(["pd", "hc"])
    y = le.transform(_y)
    y = y.reshape(-1, 1)
    return y


# loads the data given the number of gaussians, the name of the task and the type of feature.
# E.g.: (4, 'monologue', 'fisher') or 'ivecs'
def load_data(gauss, task, feat_type):
    if (feat_type == 'fisher') or (feat_type == 'ivecs'):
        # Set data directories
        file_train = work_dir + '{}/{}-20mf-2del-{}g-{}.{}'.format(task, feat_type, gauss, task, feat_type)
        file_lbl_train = work_dir + 'labels/labels_{}.txt'.format(task)

        # Load data
        X_train = np.loadtxt(file_train)
        df_labels = pd.read_csv(file_lbl_train, delimiter=' ', header=None)
        df_labels.columns = ['wav', 'label']
        Y_train = encode_labels(df_labels.label.values)

        return X_train, Y_train.ravel()
    else:
        raise ValueError("'{}' is not a supported feature representation, please enter 'ivecs' or 'fisher'.".format(feat_type))

def load_data_alternate(gauss, task):
    # Set data directories
    file_train = work_dir + 'alternate/vlfeats.mfccs.{}.all.{}.cv.txt'.format(task, gauss)
    file_lbl_train = work_dir + 'labels/labels_{}.txt'.format(task)

    # Load data
    X_train = np.loadtxt(file_train, delimiter=',')
    df_labels = pd.read_csv(file_lbl_train, delimiter=' ', header=None)
    df_labels.columns = ['wav', 'label']
    Y_train = encode_labels(df_labels.label.values)

    return X_train, Y_train.ravel()



##### UTILS FOR CREATING LABELS #####

# gets the 'subtask' folder name of the task
# e.g.: /home/egasj/PycharmProjects/the_speech/audio/sentences2/1_viste/non-normalized/hc/AVPEPUDEAC0001_viste.wav
# gets "1_viste", so that features can be saved separately from each 'subtask'

def get_parent_level_2(path_file):
    dir_of_file = os.path.dirname(path_file)
    parent_dir_of_file = os.path.dirname(dir_of_file)
    return os.path.dirname(parent_dir_of_file)


# gets parent (directory) one more upper level
def get_parent_level_3(path_file):
    return os.path.dirname(get_parent_level_2(path_file))


# make labels according to wav directory: sentences2/1_viste/non-normalized/hc/AVPEPUDEAC0001_viste.wav
# where 'hc' is the label of the wav...
def make_labels(path_file):
    label = os.path.basename(os.path.dirname(path_file)).lower()
    wav = os.path.basename(os.path.splitext(path_file)[0])
    task = os.path.basename(get_parent_level_3(path_file))
    return wav, label, task


# save the labels. list of sets/tasks (NAME of the folders containing the audios), dir to the audios, output dir
def save_labels(list_sets, audio_dir, out_dir):
    for task in list_sets:
        list_of_wavs = util.traverse_dir(audio_dir + task, '.wav')
        labels_task = []
        for wav in list_of_wavs:
            w, label, task_name = make_labels(wav)
            labels_task.append(w + ' ' + label)
            # labels_task.sort()
        np.savetxt(out_dir + "labels_{}.txt".format(task), labels_task, delimiter=',', fmt='%s')