import numpy as np
import pandas as pd

from sklearn import preprocessing


# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech' # windows machine
work_dir = '/home/egasj/PycharmProjects/the_speech/data/pcgita/'  # ubuntu machine
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
    # Set data directories
    file_train = work_dir + '{}/{}-20mf-2del-{}g-{}.fisher'.format(task, feat_type, gauss,task)
    file_lbl_train = work_dir + 'labels/labels_{}.txt'.format(task)

    # Load data
    X_train = np.loadtxt(file_train)
    df_labels = pd.read_csv(file_lbl_train, delimiter=' ', header=None)
    df_labels.columns = ['wav', 'label']
    Y_train = encode_labels(df_labels.label.values)

    return X_train, Y_train.ravel()