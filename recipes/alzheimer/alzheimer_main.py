# from data_preproc.mfccs import extract_mfccs
from data_preproc.fisher import extract_fishers
# from data_preproc.ivecs import extract_ivecs
import numpy as np
import os
from common import util

# Name of the task/recipe/dataset/etc.
import recipes.alzheimer.alzheimer_helper as ah

recipe = 'alzheimer'
folder_audios = 'demencia94B'

# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/'  # for titan x machine (normal)
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
# in and out dirs
audio_dir = work_dir + 'audio/'
out_dir = work_dir + 'data/'

# List of audio-sets (folders containing audio samples)
list_sets = ['demencia94B']

# List of number of clusters wanted to use
# list_n_clusters = [2, 8, 32, 128]
list_n_clusters = [4, 16, 64, 256]


# Computes mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    print("=======MFCC extraction phase========")
    for folder_name in list_sets:
        print("\nReading dir:", folder_name)

        # Loading id-wavs specified in the labels file
        source_file = '/data/alzheimer/labels/labels.csv'
        list_of_wavs = util.traverse_dir(audio_dir + folder_name, '.wav')
        list_of_wavs.sort()
        list_specific_wavs = ah.load_specific(source_file=source_file, list_original_audios=list_of_wavs)
        list_specific_wavs.sort()

        for num_deltas in [0, 1, 2]:
            print("Extracting with {} deltas".format(num_deltas))
            extract_mfccs.compute_mfccs(list_wavs=list_specific_wavs, out_dir=out_dir, num_mfccs=23, recipe=recipe,
                                        folder_name=folder_name, num_deltas=num_deltas)


def do_fishers():
    print("=======fisher-vector extraction phase========")
    mfcc_info = [23, 0]  # info of the mfccs (n_features, deltas)
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    list_files_ubm = [work_dir + 'data/{}/{}/mfccs_{}_23_{}_{}del.mfcc'.format(recipe, folder_audios, recipe,
                                                                               folder_audios, mfcc_info[1])]  # Format is: "featureType_recipeName_nMFCCs_nDeltas.mfcc"

    for folder_name in list_sets:
        print("\nReading dir:", mfccs_dir + folder_name)
        list_mfcc_files = util.traverse_dir_2(mfccs_dir + folder_name, '*{}del.mfcc'.format(mfcc_info[1]))
        print(list_mfcc_files)
        extract_fishers.compute_fishers(list_n_clusters, list_mfcc_files, out_dir, mfcc_info=mfcc_info,
                                        list_files_ubm=list_files_ubm, recipe=recipe, folder_name=folder_name)


def do_ivecs():
    print("=======i-vector extraction phase========")
    mfcc_info = [23, 2]  # info of the mfccs (n_features, deltas)
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    list_files_ubm = [work_dir + 'data/{}/{}/mfccs_{}_23_{}_{}del.mfcc'.format(recipe, folder_audios, recipe,
                                                                               folder_audios, mfcc_info[1])] # Format is: "featureType_recipeName_nMFCCs_nDeltas.mfcc"

    for folder_name in list_sets:
        print("\nReading dir:", mfccs_dir + folder_name)
        list_mfcc_files = util.traverse_dir_2(mfccs_dir + folder_name, '*{}del.mfcc'.format(mfcc_info[1]))
        extract_ivecs.compute_ivecs(list_n_gauss=list_n_clusters, list_mfcc_files=list_mfcc_files, out_dir=out_dir,
                                    list_files_ubm=list_files_ubm, recipe=recipe, mfcc_info=mfcc_info,
                                    folder_name=folder_name)


def do_svm():
    print("svm")


def steps(i):
    switcher = {
        0: do_mfccs,
        1: do_fishers,
        2: do_ivecs,
        3: do_svm
    }
    func = switcher.get(i)
    return func()


# steps(0)
steps(1)
# steps(2)
