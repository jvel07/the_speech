# from data_preproc.mfccs import extract_mfccs
from data_preproc.fisher import extract_fishers
# from data_preproc.ivecs import extract_ivecs
import numpy as np
import os
from common import util

# Name of the task/recipe/dataset/etc.
import recipes.demencia94B.demencia94B_helper as ah

recipe = 'demencia94ABC'

# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/'  # for titan x machine (normal)
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
# in and out dirs
audio_dir = work_dir + 'audio/'
out_dir = work_dir + 'data/'

# List of audio-sets (folder(s) containing audio samples)
list_sets = ['demencia94ABC']

# List of number of clusters wanted to use
list_n_clusters = [2, 8, 16, 32, 64, 128]
# list_n_clusters = [4, 256]


# Computes mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    print("=======MFCC extraction phase========")
    for folder_name in list_sets:
        cepstral_type = "mfcc"  # choose between "mfcc" or "plp"

        # Loading id-wavs specified in the labels file
        print("\nReading dir:", folder_name)
        source_file = '../../data/{}/labels/labels.csv'.format(recipe)
        list_of_wavs = util.traverse_dir(audio_dir + folder_name, '.wav')
        list_of_wavs.sort()
        list_specific_wavs = ah.load_specific(source_file=source_file, list_original_audios=list_of_wavs)
        list_specific_wavs.sort()  # best 75 of 94 wavs

        for deltas in [0, 1, 2]:
            print("Extracting with {} deltas".format(deltas))
            extract_mfccs.compute_flevel_feats(list_of_wavs, out_dir, cepstral_type=cepstral_type, num_feats=13,
                                               recipe=recipe, folder_name=folder_name, num_deltas=deltas, obs='')


def do_fishers():
    print("=======fisher-vector extraction phase========")
    feature_dir = work_dir + '/data/{}/'.format(recipe)

    for deltas in [0, 1, 2]:
        # info-purpose parameters from the frame-level extracted features #
        feats_info = [13, deltas, 'mfcc']  # info of the features (n_features/dimension, deltas, cepstral_type=choose between mfcc or plp)
        obs = ''
        # Format is: "featureType_recipeName_nMFCCs_nDeltas.mfcc"
        list_files_ubm = [work_dir + 'data/demencia94ABC/beadiktafon/mfcc_demencia94ABC_13_beadiktafon_{}del.mfcc'.format(deltas)]
        for folder_name in list_sets:
            list_mfcc_files = util.traverse_dir_2(feature_dir + folder_name, '*{}_{}_{}del.{}'.format(feats_info[0],
                                                                                                    folder_name,
                                                                                                    feats_info[1],
                                                                                                    feats_info[2]))
            print(list_mfcc_files)
            extract_fishers.compute_fishers(list_n_clusters, list_mfcc_files, out_dir, feats_info=feats_info,
                                                list_files_ubm=list_files_ubm, recipe=recipe, folder_name=folder_name)


def do_ivecs():
    print("=======i-vector extraction phase========")
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)

    for deltas in [2]:
        feats_info = [23, deltas, 'mfcc']  # info of the mfccs (n_features, deltas)
        list_files_ubm = [work_dir + 'data/demencia94ABC/beadiktafon/mfcc_demencia94ABC_23_beadiktafon_{}del.mfcc'.format(deltas)]
        for folder_name in list_sets:
            print("\nReading dir:", mfccs_dir + folder_name)
            list_mfcc_files = util.traverse_dir_2(mfccs_dir + folder_name, '*{}_{}_{}del.{}'.format(feats_info[0],
                                                                                                   folder_name,
                                                                                                   feats_info[1], feats_info[2]))
            extract_ivecs.compute_ivecs(list_n_gauss=list_n_clusters, list_mfcc_files=list_mfcc_files, out_dir=out_dir,
                                        list_files_ubm=list_files_ubm, recipe=recipe, mfcc_info=feats_info,
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
