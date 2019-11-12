#import bob
import numpy as np
import os, fnmatch
import re
import csv
from itertools import zip_longest
import pickle


# Read pickle
def read_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        print("Pickle loaded from:", file_name)
    return data


# save pickle
def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
        print("Data pickled to file:", file_name)


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


# Reading a list of files from a directory regex e.g. '*.wav' or '*.mfc'
def read_files_from_dir_reg(dir_name, regex):
    list_of_files = os.listdir(dir_name)
    pattern = re.compile(regex)
    file_list = []
    for entry in sorted(list_of_files):
        if re.match(pattern, entry):  # fnmatch.fnmatch(entry, pattern):
            file_list.append(entry)
    return file_list


# for files in os.walk("../wav_audios/."):
# for filename in files:
#   print(filename)


# Reading a list of files from a directory
def read_files_from_dir(dir_name):
    list_of_files = os.listdir(dir_name)
    file_list = []
    for entry in sorted(list_of_files):
        file_list.append(entry)
    return file_list


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
# returns a list of processed and trasnposed mfccs ready to be fed to Fishers
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

"""FOR THE DEMENTIA DATASET PREPROCESSING --- START"""


# Select between szurke or feher
def select_type_recording(id_type):
    regex = r"(?<=[_])(.+)$"  ##matches everything after "_" e.g. szurke.wav in 001A_szurke.wav
    pattern = re.compile(regex)


#   For adding the labels from each speaker to each speaker's wav (3 wavs per speaker)
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
def group_per_audio_type(features):
    """group speakers per type of audio (normal, noisy, stretched, pitched)"""
    #features = read_pickle(file)
    length = len(features)
    number_of_rows = 4
    number_of_group = 12
    return [list(features[i:i+number_of_group][j::number_of_rows]) for i in range(0, length, number_of_group) for j in range(number_of_rows)]


def group_per_audio_type_2(features):
   # features = read_pickle(file)
    length = 3
    step = 4
    size = step * length
    return np.array(features).reshape(len(features) // size, length, step).transpose(0, 2, 1).reshape(len(features) // length, length)


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
def traverse_dir(dir, file_type):
    lista = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(file_type):
                lista.append(os.path.join(root, file))
    return lista
"""FOR THE DEMENTIA DATASET PREPROCESSING  ---  END"""

# Open MFCCs from file
def read_mfcc(file_name):
    with open(file_name, 'rb') as f:
        mfccs = pickle.load(f)
        print("MFCCs loaded from:", file_name)
    return mfccs


"""
# Save bob machines
def save_bob_machine(hdf5_file_name, machine):
    hdf5_file = bob.io.base.HDF5File(hdf5_file_name, 'w')
    machine.save(hdf5_file)
    del hdf5_file
    print("Machine saved to:", hdf5_file_name)


# Load bob machines
def load_bob_machine(hdf5_file_name):
    hdf5_file = bob.io.base.HDF5File(hdf5_file_name)
    return hdf5_file
"""