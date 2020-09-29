# from data_preproc.mfccs import extract_mfccs
# from data_preproc.fisher import extract_fishers
from data_preproc.ivecs import extract_ivecs
import numpy as np
import os
from common import util

# Name of the task/recipe/dataset/etc.
import recipes.dementia_new8k.dementia_new8k_helper as ah

recipe = 'dementia_new8k'

# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/'  # for titan x machine (normal)
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
# in and out dirs
audio_dir = work_dir + 'audio/'
out_dir = work_dir + 'data/'

# List of audio-sets (folder(s) containing audio samples) i.e. train, dev, test...
list_sets = ['dementia_new8k']

# Name of the folder containing the UBM (not the path)
ubm_folder_name = 'bea_wav8k'

# List of number of clusters wanted to use
# list_n_clusters = [256]
# list_n_clusters = [2, 4, 8, 16, 32, 64, 128, 256]
list_n_clusters = [256]
# list_n_clusters = [64]


# Computes mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    print("=======MFCC extraction phase========")
    for folder_name in list_sets:
        cepstral_type = "mfcc"  # choose between "mfcc" or "plp"

        # Loading id-wavs specified in the labels file
        print("\nReading dir:", folder_name)
        source_file = '../../data/{}/labels/labels.csv'.format(recipe)
        list_of_wavs = util.traverse_dir(audio_dir + folder_name, file_type='.wav')
        list_of_wavs.sort()
        list_specific_wavs = ah.load_specific(source_file=source_file, list_original_audios=list_of_wavs)
        list_specific_wavs.sort()  # best 33 wavs selected manually

        for deltas in [0, 1, 2]:
            print("\n Extracting with {} deltas".format(deltas))
            extract_mfccs.compute_flevel_feats(list_specific_wavs, out_dir, cepstral_type=cepstral_type, num_feats=23,
                                               recipe=recipe, folder_name=folder_name, num_deltas=deltas, obs='')

def do_mfccs_ubm():
    print("=======MFCC extraction phase for UBM========")
    out_dir = work_dir + 'data/'

    cepstral_type = "mfcc"  # choose between "mfcc" or "plp"

    # Loading filtered id-wavs
    arr_filtered_wavs_id = np.genfromtxt(
        '/media/jose/hk-data/PycharmProjects/the_speech/recipes/demencia94B/filt_UBMbea_lthan4secs.txt', dtype=str,
        delimiter='\n')
    list_specific_wavs = []
    for i in arr_filtered_wavs_id:  # adding the parent path to the name of the wav
        list_specific_wavs.append('/media/jose/hk-data/PycharmProjects/the_speech/audio/bea_wav8k/' + i)
    # list_specific_wavs[1]

    print("\nReading dir:", ubm_folder_name)
    for deltas in [0, 1, 2]:
        print("Extracting with {} deltas".format(deltas))
        extract_mfccs.compute_flevel_feats(list_specific_wavs, out_dir, cepstral_type=cepstral_type, num_feats=20,
                                           recipe=recipe, folder_name=ubm_folder_name, num_deltas=deltas, obs='')

# for training the extractor and extracting i-vecs when there exists already pretrained UBMs (diagonal ones)
def do_fishers_pretrained_ubm():
    feature_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'  # Where the computed features will be saved
    ubm_dir = work_dir + 'data/UBMs/' + ubm_folder_name +'/ivec_models/'  # where the diagonal ubms live
    # list_ubm_files = util.traverse_dir(ubm_dir, '.dubm')  #  reading all the files with .mdl or .dubm as format (latter is more reliable)

    list_sets = ['dementia_new8k']
    for g in list_n_clusters:
        for deltas in [0, 1, 2]:
            # info-purpose parameters from the frame-level extracted features #
            feats_info = [20, deltas, 'mfcc']  # info of the features (n_features/dimension, deltas, cepstral_type=choose between mfcc or plp)
            for folder_name in list_sets:  # iterating over the list of sets where the features live
                print("\nReading dir:", feature_dir + folder_name)
                list_ubm_files = util.traverse_dir_2(ubm_dir, '*_{}g_{}{}-{}del_{}.dubm'.format(g, feats_info[0], feats_info[2], feats_info[1],
                                                                                                 ubm_folder_name))
                list_mfcc_files = util.traverse_dir_2(feature_dir + folder_name, '*{}_{}_{}del.{}'.format(feats_info[0],
                                                                                                          folder_name,
                                                                                                          feats_info[1],
                                                                                                          feats_info[2]))
                extract_fishers.compute_fishers_pretr_ubm_2(list_mfcc_files=list_mfcc_files, out_dir=out_dir, feats_info=feats_info,
                                                          list_files_ubm=list_ubm_files, recipe=recipe, folder_name=folder_name)



def do_fishers():
    print("=======fisher-vector extraction phase========")
    feature_dir = work_dir + '/data/{}/'.format(recipe)

    for deltas in [0, 1, 2]:
        # info-purpose parameters from the frame-level extracted features #
        feats_info = [20, deltas, 'mfcc']  # info of the features (n_features/dimension, deltas, cepstral_type=choose between mfcc or plp)
        obs = ''
        # Format is: "featureType_recipeName_nMFCCs_nDeltas.mfcc"
        list_files_ubm = [work_dir + 'data/demencia94ABC/wav16k_split_long/mfcc_demencia94ABC_20_wav16k_split_long_{}del.mfcc'.format(deltas)]
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
    ubm_dir = work_dir + 'data/UBMs/{}'.format(ubm_folder_name)

    for deltas in [2]:
        feats_info = [20, deltas, 'mfcc']  # info of the mfccs (n_features, deltas)
        # the following is a list in the case of the UBM is meant to be trained with training and dev sets
        list_files_ubm = [ubm_dir + '/mfcc_{}_{}_{}del.mfcc'.format(feats_info[0], ubm_folder_name, deltas)]
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
        3: do_svm,
        4: do_mfccs_ubm,
        5: do_fishers_pretrained_ubm
    }
    func = switcher.get(i)
    return func()


steps(2)
# step(4)
# steps(2)
