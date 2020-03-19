import numpy as np
import pandas as pd
import os
from common import util


from sklearn import preprocessing


# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/data/pcgita/' # windows machine
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/data/'  # ubuntu machine
# work_dir2 = 'D:/VHD'


# Encoding labels to numbers
def encode_labels(_y, label_1, label_0):
    le = preprocessing.LabelEncoder()
    le.fit([label_1, label_0])
    y = le.transform(_y)
    y = y.reshape(-1, 1)
    return y


# loads the data given the number of gaussians, the name of the task and the type of feature.
# Used for small datasets; loads single file containing training and test features.
# E.g.: (4, 'monologue', 'fisher') or 'ivecs'
# example: train/fisher-23mf-0del-2g-train.fisher
def load_data_single(gauss, task, feat_type, n_feats, n_deltas):
    if (feat_type == 'fisher') or (feat_type == 'ivecs'):
        # Set data directories
        file_train = work_dir + '{}/{}-{}mf-{}del-{}g-{}.{}'.format(task, feat_type, n_feats, n_deltas, gauss, task, feat_type)
        file_lbl_train = work_dir + 'labels/labels_{}.txt'.format(task)

        # Load data
        X_train = np.loadtxt(file_train)
        df_labels = pd.read_csv(file_lbl_train, delimiter=' ', header=None)
        df_labels.columns = ['wav', 'label']
        Y_train = encode_labels(df_labels.label.values)

        return X_train, Y_train.ravel()
    else:
        raise ValueError("'{}' is not a supported feature representation, please enter 'ivecs' or 'fisher'.".format(feat_type))


# E.g.: (4, 'monologue', 'fisher') or 'ivecs'
# example: train/fisher-23mf-0del-2g-train.fisher
# loads data existing in the folders "train", "dev" and "test"
# format of the data labels required and file with the following headers:
#    'file_name     label'           'file_name     label'
#    'recording.wav label'. Example: 'train_0001.wav True', 'train_0002.wav 2 False', ...
def load_data_full(gauss, task, feat_type, n_feats, n_deltas, label_1, label_0):
    list_datasets = ['train', 'dev', 'test']  # names for the datasets
    list_labels = ['y_train', 'y_dev']  # names for the labels
    dict_data = {}
    if (feat_type == 'fisher') or (feat_type == 'ivecs'):
        # Load train, dev, test
        for item in list_datasets:
            # Set data directories
            file_dataset = work_dir + '{}/{}/{}-{}mf-{}del-{}g-{}.{}'.format(task, item, feat_type, n_feats, n_deltas, gauss, item, feat_type)
            # Load datasets
            dict_data['x_'+item] = np.loadtxt(file_dataset)
            # Load labels
            file_lbl_train = work_dir + '{}/labels/labels.csv'.format(task) # set data dir
            df = pd.read_csv(file_lbl_train)
            df_labels = df[df['file_name'].str.match(item)]
            df_labels = df_labels.label.replace('?', label_1)
            dict_data['y_'+item] = encode_labels(df_labels.values, label_1, label_0)
        return dict_data['x_train'], dict_data['x_dev'], dict_data['x_test'], dict_data['y_train'], dict_data['y_dev']
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