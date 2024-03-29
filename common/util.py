# import bob
#import h5py
import math

import h5py
import numpy as np
import os, fnmatch
import re
import csv
from itertools import zip_longest
import pickle
from shutil import copy
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


# Read pickle
def read_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        print("Pickle loaded from:", file_name, "With lenght:", len(data), "First ele. shape:", data[0].shape)
    return data


# save as hdf5
def save_as_hdf5(file_name, dataset_name, data):
    hf = h5py.File(file_name, 'w')
    hf.create_dataset(dataset_name, data=data)
    hf.close()
    print("HDF5 to file:", file_name)


# FOR PICKLING FILES MORE THAN 4GB
class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump_big(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)
        print("Data pickled to file:", file_name, "With lenght:", len(obj), "First ele. shape:", obj[0].shape)


def pickle_load_big(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(MacOSFile(f))
        print("Pickle loaded from:", file_path, "With lenght:", len(data), "First ele. shape:", data[0].shape)
        return data

# save pickle
def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        try:
            pickle.dump(data, f, protocol=4)
        except:
            print("Error when saving pickle file:", file_name)


# Read training, dev and test features
def read_txt_file(filename, delimiter=','):
    with open(filename, 'r') as data_file:
        data = np.loadtxt(data_file, delimiter=",")
    print(filename + ": File successfully loaded with:", data.shape)
    return data


# Read training, dev and test features as lists
def read_txt_file_as_list(filename):
    with open(filename, 'r') as data_file:
        np_data = np.loadtxt(data_file, delimiter=",")
        list_data = np_data.tolist()
    print(filename + ": File successfully loaded!")
    return list_data


# Read files separated by newlines
def read_file_new_line_sep(filename):
    list_of_lines = open(filename).read().splitlines()
    return list_of_lines


# Reading a list of files from a directory regex e.g. '.wav' or '.mffc'
def read_files_from_dir_reg(dir_name, regex):
    list_of_files = os.listdir(dir_name)
    pattern = re.compile(r'{}'.format(regex))
    file_list = []
    for entry in sorted(list_of_files):
        if re.match(pattern, entry):  # fnmatch.fnmatch(entry, pattern):
            file_list.append(entry)
    return file_list

"""
import glob, os
os.chdir("/mydir")
for file in glob.glob("*.txt"):
    print(file)
"""


# extracting numbers from strings. E.g. from '/home/egasj/PycharmProjects/the_speech/data/pcgita/UBMs/64/ubm/final.ubm'
# extract 64
def extract_numbers_from_str(string):
    temp1 = re.findall(r'\d+', string)  # through regular expression
    res2 = list(map(int, temp1))
    return res2

# for files in os.walk("../wav_audios/."):
# for filename in files:
#   print(filename)


# Reading a list of files from a directory
def read_files_from_dir(dir_name):
    list_of_files = os.listdir(dir_name)
    return list_of_files.sort()


"""
# Reading single MFC HTK-generated file. With https://github.com/danijel3/PyHTK
def read_mfc_htk_file(filename):  # save_as):
    htk = HTKFile()
    htk.load(filename)
    result = np.array(htk.data)
    #  np.savetxt(save_as, result, fmt='%.4f')
    print(filename + " MFC file loaded successfully with:", result.shape)
    #  print('\n')
    return result


# Processing set of HTK files, receives as parameter regex e.g. r'^train'
# if the file starts with train, with r'^dev' if it starts with development, etc.
# returns a list of processed and trasnposed hc ready to be fed to Fishers
def process_htk_files_for_fishers_trans(path_to_mfccs, regex):
    list_processed = []
    list_of_files = read_files_from_dir(path_to_mfccs, regex)
    for item in list_of_files:
        mfcc_feat_train = read_mfc_htk_file(path_to_mfccs + item)
        mfcc_feat_train_26 = mfcc_feat_train[:, :26]
        mfcc_as_array = np.asarray(mfcc_feat_train_26)
        mfcc_transposed_26 = mfcc_as_array.transpose()
        print("New shape of features", mfcc_transposed_26.shape)
        list_processed.append(mfcc_transposed_26)

    return list_processed


def process_htk_files_for_fishers_normal(path_to_mfccs, regex):
    list_processed = []
    list_of_files = read_files_from_dir(path_to_mfccs, regex)
    for item in list_of_files:
        mfcc_feat = read_mfc_htk_file(path_to_mfccs + item)
        mfcc_feat_26 = mfcc_feat[:, :26]
        mfcc_as_array = np.asarray(mfcc_feat_26)
        print("New shape of features", mfcc_as_array.shape)
        list_processed.append(mfcc_as_array)

    return list_processed
"""


# def results_to_csv(file_name, g, feat_type, num_filters, deltas, vad, pca, acc):
#     if not os.path.isfile(file_name):
#         with open(file_name, mode='w') as csv_file:
#             file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             file_writer.writerow(['Gaussians', 'Feature', 'N_filters', 'VAD', 'PCA', 'Accuracy'])
#             file_writer.writerow([g, feat_type, num_filters, deltas, vad, pca, acc])
#             print("File " + file_name + " created!")
#     else:
#         with open(file_name, 'a') as csv_file:
#             file_writer = csv.writer(csv_file)
#             file_writer.writerow([g, feat_type, num_filters, deltas, vad, pca, acc])
#             print("File " + file_name + " updated!")
#

# write results to csv
def results_to_csv(file_name, list_columns, list_values):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(list_columns)
            file_writer.writerow(list_values)
            print("File " + file_name + " created!")
    else:
        with open(file_name, 'a') as csv_file:
            file_writer = csv.writer(csv_file)
            file_writer.writerow(list_values)
            # print("File " + file_name + " updated!")


# from a list of files and labels, take only the set of files that have a specified label value.
# e.g. from the dataset parkinson's, take the file-names with PD label only.
def take_only_specfic_label(wavs_dir, list_labels, lbl_value):
    all_wavs = read_files_from_dir(wavs_dir)
    # list_labels = np.loadtxt(list_labels)
    lista = []
    for wav, label in zip(all_wavs, list_labels):
        if label == lbl_value: lista.append(wav)
    return lista


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title("Confusion Matrix")

    seaborn.set(font_scale=1.2)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", fmt='.2f', cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_confusion_matrix_2(data, labels, output_filename, cmap, title):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title(title)

    seaborn.set(font_scale=0.95)

    group_counts = ["{0: 0.0f}".format(value) for value in data.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in data.flatten() / np.sum(data)]
    labels_plot = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]

    labels_plot = np.asarray(labels_plot).reshape(data.shape[0], data.shape[1])

    ax = seaborn.heatmap(data, annot=labels_plot, cmap=cmap, fmt='', cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=500)
    plt.close()


def plot_figure(data, labels, legend_x, legend_y, output_filename):
    plt.figure(1, figsize=(9, 6))
    seaborn.set_theme(style='whitegrid', font_scale=1.5)
    ax = seaborn.lineplot(data, labels)
    ax.set(xlabel=legend_x, ylabel=legend_y)
    plt.savefig(output_filename, bbox_inches='tight', dpi=1000, format='eps')
    plt.close()


def plot_multiple_histograms(df):
    fig, axs = plt.subplots(2, 1, figsize=(15, 7))
    fig.subplots_adjust(hspace=0.5)
    sns.set_theme(style='whitegrid', font_scale=1.5)
    # sns.barplot(ax=axs[0, 0], x=df['UAR'], y=df['dims'])
    sns.lineplot(ax=axs[0], y=df['AUC'], x=df['dims'])
    sns.lineplot(ax=axs[1], y=df['PEARSON'], x=df['dims'])
    # sns.barplot(ax=axs[1, 0], x=df['RMSE'], y=df['dims'])
    plt.savefig('exp_results/metrics2', bbox_inches='tight', dpi=1000)
    plt.close(fig)

# plot_multiple_histograms(df1)

# copies specific files contained within a python list to a path (directory)
def copy_from_list_to_dir(list_of_files, orig_path, dest_path):
    for file in list_of_files:
        copy(orig_path+file, dest_path)


"""FOR THE DEMENTIA DATASET PREPROCESSING --- START"""

# Read just original 75 speakers
def just_original_75():
    work_dir = 'C:/Users/Win10/PycharmProjects/the_speech' #/opt/project/
    lines = open(work_dir + "/data/wavlista-anon-75-225.txt").read().splitlines()
    wavlista_anon_75_225 = []
    for it in lines:
        wav_file = '{}.wav'.format(it)
        wavlista_anon_75_225.append(wav_file)
    return wavlista_anon_75_225


# Select between szurke or feher
def select_type_recording(id_type):
    regex = r"(?<=[_])(.+)$"  ##matches everything after "_" e.g. szurke.wav in 001A_szurke.wav
    pattern = re.compile(regex)


#   For adding the labels from each speaker to each speaker's wav (3 wavs per speaker) dementia recipe
def putting_labels(labels_file, wavs_list_file):
    labels = []
    with open(labels_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(row)
    wavs = sorted(open(wavs_list_file).read().splitlines())
    labels_anon75 = []
    for x1, x2 in sorted(labels):
        for x in sorted(wavs):
            if x1 == x[:3]:
                labels_anon75.append(x2)
    return labels_anon75


# Grouping wavs per 'n' speakers
def group_wavs_speakers(iterable, n):  # iterate every n element within a list
    #   "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    lista = []
    for x, y, z in zip_longest(*[iter(iterable)] * n):
        lista.append(list((x, y, z)))

    return lista


# group speakers per type of audio (normal, noisy, stretched, pitched)
def group_per_audio_type(features, gr, st):
    """group speakers per type of audio (normal, noisy, stretched, pitched)"""
    # features = read_pickle(file)
    length = len(features)
    number_of_rows = st
    number_of_group = gr #12, 15
    return [list(features[i:i + number_of_group][j::number_of_rows]) for i in range(0, length, number_of_group) for j in
            range(number_of_rows)]


def group_per_audio_type_2(features, st):
    # features = read_pickle(file)
    length = 3
    step = st  # grouping audio features every n...
    size = step * length
    return list(np.array(features).reshape(len(features) // size, length, step).transpose(0, 2, 1).reshape(
        len(features) // length, length))


# Concatenate speakers wavs (3) in one single array
# Returns list of concatenated arrays
def join_speakers_wavs(list_group_features):
    x = []
    for element in list_group_features:
        for a, b, c in zip_longest(*[iter(element)] * 3):  # iterate over the sublist of the list
            array = np.concatenate((a, b, c))  # concatenating arrays (every 3)
            x.append(array)
    print("Speakers' wavs concatenated!")
    return x


# Traverse directories and pull specific type of file (".WAV", etc...)
def traverse_dir(path, file_type):
    if os.path.isdir(path):
        lista = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(file_type):
                    lista.append(os.path.join(root, file))
        return lista
    else:
        print("\nERROR: Path '{}' does not exist!".format(path))
        raise FileNotFoundError


# Traverse directories and pull specific type of file (".WAV", etc...) same as 1 but using fnmatch which allows specifiying
# more details of the filename
def traverse_dir_2(path, file_type):
    if os.path.isdir(path):
        lista = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if fnmatch.fnmatch(file, file_type):
                    lista.append(os.path.join(root, file))
        return lista
    else:
        print("\nERROR: Path '{}' does not exist!".format(path))
        raise FileNotFoundError


"""FOR THE DEMENTIA DATASET PREPROCESSING  ---  END"""




# Open MFCCs from file
def read_mfcc(file_name):
    with open(file_name, 'rb') as f:
        mfccs = pickle.load(f)
        print("MFCCs loaded from:", file_name)
    return mfccs


def put_commas_to(_string):
    return re.sub("\s+", ",", _string.strip())

