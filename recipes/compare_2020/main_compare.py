from data_preproc.mfccs import extract_mfccs
# from data_preproc.fisher import extract_fishers
# from data_preproc.ivecs import extract_ivecs
import numpy as np
import os
from common import util

# Name of the task/recipe/dataset/etc.
recipe = 'mask'

# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/'  # for titan x machine (normal)
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
# in and out dirs
audio_dir = work_dir + 'audio/' + recipe + '/'
out_dir = work_dir + 'data/'

# List of audio-sets (folders containing audio samples)
list_sets = ['train', 'dev', 'test']

# List of number of clusters wanted to use
# list_n_clusters = [2, 8, 32, 128]
list_n_clusters = [4, 16, 64, 256]


# Computes mfccs from wavs existing in the directories provided by the user
def do_frame_level():
    print("=======Frame-level extraction phase========")

    cepstral_type = "mfcc"  # choose between "mfcc" or "plp"
    for folder_name in list_sets:
        print("\nReading dir:", folder_name)
        list_of_wavs = util.traverse_dir(audio_dir + folder_name, '.wav')
        list_of_wavs.sort()
        # print(list_of_wavs)
        extract_mfccs.compute_mfccs(list_of_wavs, out_dir, cepstral_type=cepstral_type, num_feats=40, recipe=recipe,
                                    folder_name=folder_name, num_deltas=1, raw_energy=False, num_mel_bins=40,
                                    low_freq=20, high_freq=-400, obs='hires')


def do_fishers():
    print("=======fisher-vector extraction phase========")
    # info-purpose parameters #
    cepstral_type = "mfcc"  # choose between mfcc or plp
    feats_info = [40, 0]  # info of the features (n_features/dimension, deltas)
    obs = '_hires'  # observations of the features

    feature_dir = work_dir + '/data/{}/'.format(recipe)
    list_files_ubm = [work_dir + '/data/{}/train/{}_{}_{}_train_{}del{}.{}'.format(recipe, cepstral_type,
                                                                                 recipe,  feats_info[0], feats_info[1],
                                                                                 obs, cepstral_type),
                      work_dir + '/data/{}/train/{}_{}_{}_train_{}del{}.{}'.format(recipe, cepstral_type,
                                                                                 recipe,  feats_info[0], feats_info[1],
                                                                                 obs, cepstral_type)]  # Format is: "featureType_recipeName_nMFCCs_nDeltas.mfcc"

    for folder_name in list_sets:
        print("\nReading dir:", feature_dir + folder_name)
        list_mfcc_files = util.traverse_dir_2(feature_dir + folder_name, '*{}del{}.{}'.format(feats_info[1], obs,
                                                                                             cepstral_type))
        print(list_mfcc_files)
        extract_fishers.compute_fishers(list_n_clusters, list_mfcc_files, out_dir, feats_info=feats_info,
                                        list_files_ubm=list_files_ubm, recipe=recipe, folder_name=folder_name)


def do_ivecs():
    print("=======i-vector extraction phase========")
    mfccs_dir = work_dir + '/data/{}/'.format(recipe)
    file_ubm = work_dir + '/data/compare_2020/train/mfccs_mask_23_train_1del.mfcc'  # Format is: "featureType_recipeName_numberOfDeltas.mfcc"

    for folder_name in list_sets:
        print("\nReading dir:", mfccs_dir + folder_name)
        list_mfcc_files = util.traverse_dir(mfccs_dir + folder_name, '.mfcc')
        extract_ivecs.compute_ivecs(list_mfcc_files, out_dir, info_num_feats_got=13, ivec_dims=256,
                                    file_ubm_feats=file_ubm, recipe=recipe, folder_name=folder_name)


def do_svm():
    print("svm")


def steps(i):
    switcher = {
        0: do_frame_level,
        1: do_fishers,
        2: do_ivecs,
        3: do_svm
    }
    func = switcher.get(i)
    return func()


steps(0)
# steps(1)
# steps(2)
