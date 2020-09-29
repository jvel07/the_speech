# from data_preproc.mfccs import extract_mfccs
from data_preproc.fisher import extract_fishers
# from data_preproc.ivecs import extract_ivecs
import numpy as np
import os
from common import util
from recipes.pc_gita.utils_pcgita import save_labels
# from signal_processing import resample_wav

recipe = 'pcgita'

# List of audio-sets (folders containing audio samples)
list_sets = ['monologue_erlangen', 'readtext_erlangen']


# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/'  # for titan x machine (normal)
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)


# Compute mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    print("=======Frame-level extraction phase========")

    audio_dir = work_dir + 'audio/'
    out_dir = work_dir + 'data/'
    list_sets = ['monologue_erlangen', 'readtext_erlangen']
    cepstral_type = "mfcc"  # choose between "mfcc" or "plp"

    for folder_name in list_sets:
        print("Reading dir:", folder_name)
        list_of_wavs = util.traverse_dir(audio_dir + folder_name, '.wav')
        list_of_wavs.sort()
        # save the labels. list of sets/tasks (NAME of the folders containing the audios), dir to the audios, output dir
        save_labels(list_sets, audio_dir, out_dir + recipe + '/')  # make labels of the wavs
        for deltas in [1,2]:
            extract_mfccs.compute_flevel_feats(list_of_wavs, out_dir, cepstral_type=cepstral_type, num_feats=20,
                                               recipe=recipe, folder_name=folder_name, num_deltas=deltas, obs='')


def do_fishers():
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'
    file_ubm = work_dir + 'data/pcgita/DDK_analysis/mfccs_pcgita_20_DDK_analysis_2del.mfcc'

    for folder_name in list_sets:
        print("Reading dir:", mfccs_dir + folder_name)
        list_mfcc_files = util.traverse_dir(mfccs_dir + folder_name, '.mfcc')
        extract_fishers.compute_fishers(list_n_clusters=[2, 4, 8, 16, 32], list_mfcc_files=list_mfcc_files, out_dir=out_dir,
                                        list_files_ubm =file_ubm, recipe=recipe, folder_name=folder_name)


# for training the extractor and extracting i-vecs when there exists already pretrained UBMs
def do_fishers_pretrained_ubm():
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'  # Where the computed features will live in
    ubm_dir = work_dir + 'data/' + recipe + '/UBMs/'  # where the diagonal ubms live
    list_ubm_files = util.traverse_dir(ubm_dir, '.mdl')  #  reading all the files with .mdl or .dubm as format (latter is more reliable)
    mfcc_n_deltas = 2  # Number of deltas of the mfccs

    list_sets = ['monologue_erlangen', 'readtext_erlangen']

    for folder_name in list_sets:  # iterating over the list of sets where the features live
        print("\nReading dir:", mfccs_dir + folder_name)
        for ubm in list_ubm_files:  # iterating over the pretrained ubms
            list_mfcc_files = util.traverse_dir_2(mfccs_dir + folder_name, '*{}del.mfcc'.format(mfcc_n_deltas))  # reading MFCCs to extracting fishers from
            extract_fishers.compute_fishers_pretr_ubm(list_mfcc_files=list_mfcc_files, out_dir=out_dir,
                                                      file_ubm=ubm, recipe=recipe, folder_name=folder_name)


def do_ivecs():
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'
    file_ubm = work_dir + 'data/pcgita/ubm/.mfcc'  # Naming format is: "featureType_recipeName_numberOfDeltas.mfcc"

    for folder_name in list_sets:
        print("\nReading dir:", mfccs_dir + folder_name)
        list_mfcc_files = util.traverse_dir(mfccs_dir + folder_name, '.mfcc')
        extract_ivecs.compute_ivecs(list_n_gauss=[2, 4, 8, 16, 32, 64], list_mfcc_files=list_mfcc_files, out_dir=out_dir,
                                    file_ubm_feats=file_ubm, recipe=recipe, folder_name=folder_name)


# for training the extractor and extracting i-vecs when there exists already pretrained UBMs
def do_ivecs_pretrained_mdls():
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'
    ubm_dir = work_dir + 'data/' + recipe + '/UBMs/'  # where the ubms live
    list_ubm_files = util.traverse_dir(ubm_dir, '.mdll')  # reading all the files with .mdl format

    list_sets = ['read_text']

    for folder_name in list_sets:  # iterating over the list of sets where the features live
        print("\nReading dir:", mfccs_dir + folder_name)
        for ubm in list_ubm_files:  # iterating over the pretrained ubms
            n_ubm = util.extract_numbers_from_str(ubm)  # getting the number of ubms of the corresponding file
            print("\ni-vecs for {} GMMs".format(n_ubm))
            list_mfcc_files = util.traverse_dir(mfccs_dir + folder_name, '.mfcc')  # reading MFCCs to extracting i-vecs from
            extract_ivecs.compute_ivecs_pretr_ubms(list_mfcc_files, out_dir, #n_ubm=n_ubm,
                                                   file_ubm=ubm, recipe=recipe, folder_name=folder_name)


# resample wavs for final test (it is not referred to the test set)
def do_resample():
    audio_dir = work_dir + 'audio/'
    out_dir = work_dir + 'audio/'
    list_sets = ['readtext_test', 'monologue_test']

    for folder_name in list_sets:
        print("Reading dir:", folder_name)
        list_of_wavs = util.traverse_dir(audio_dir + folder_name, '.wav')
        resample_wav.make_resample(list_of_wavs, out_dir, target_sample_rate=16000, subtype='PCM_16')


def do_svm():
    print("to be implemented soon...")


def steps(i):
    switcher = {
        0: do_mfccs,
        1: do_fishers,
        2: do_ivecs,
        3: do_svm,
        4: do_ivecs_pretrained_mdls,
        5: do_fishers_pretrained_ubm,
        6: do_resample
    }
    func = switcher.get(i)
    return func()


steps(5)
# steps(6)
# steps(5)