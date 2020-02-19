from data_preproc.mfccs import extract_mfccs
from data_preproc.fisher import extract_fishers
from data_preproc.ivecs import extract_ivecs
import numpy as np
import os
from common import util

recipe = 'pcgita'

# List of audio-sets (folders containing audio samples)
list_sets = ['DDK_analysis', 'monologue', 'read_text', 'sentences', 'sentences2']

# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
work_dir = '/home/egasj/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)


# Compute mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    audio_dir = work_dir + 'audio/'
    out_dir = work_dir + 'data/'

    for folder_name in list_sets:
        print("Reading dir:", folder_name)
        list_of_wavs = util.traverse_dir(audio_dir + folder_name, '.wav')
        # print(list_of_wavs[0])
        extract_mfccs.compute_mfccs(list_of_wavs, out_dir, num_mfccs=20, recipe='pcgita', folder_name=folder_name)


def do_fishers():
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'
    file_ubm = work_dir + 'data/pcgita/DDK_analysis/mfccs_pcgita_20_DDK_analysis_2del.mfcc'

    for folder_name in list_sets:
        print("Reading dir:", mfccs_dir + folder_name)
        list_mfcc_files = util.traverse_dir(mfccs_dir + folder_name, '.mfcc')
        extract_fishers.compute_fishers(list_mfcc_files, out_dir, info_num_feats=20,
                                        file_ubm_feats=file_ubm, recipe=recipe, folder_name=folder_name)


def do_ivecs():
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'
    file_ubm = work_dir + 'data/pcgita/monologue/mfccs_pcgita_20_monologue_2del.mfcc'  # Format is: "featureType_recipeName_numberOfDeltas.mfcc"

    for folder_name in list_sets:
        print("\nReading dir:", mfccs_dir + folder_name)
        list_mfcc_files = util.traverse_dir(mfccs_dir + folder_name, '.mfcc')
        extract_ivecs.compute_ivecs(list_mfcc_files, out_dir, info_num_feats_got=20,
                                    file_ubm_feats=file_ubm, ivec_dims=256, recipe=recipe, folder_name=folder_name)


# for training the extractor and extracting i-vecs when there exists already pretrained UBMs
def do_ivecs_pretrained_mdls():
    mfccs_dir = work_dir + 'data/{}/'.format(recipe)
    out_dir = work_dir + 'data/'
    ubm_dir = work_dir + 'data/' + recipe + '/UBMs/'  # where the ubms live
    list_ubm_files = util.traverse_dir(ubm_dir, '.mdl')  # reading all the files with .mdl format

    for folder_name in list_sets:  # iterating over the list of sets (audio sets)
        print("\nReading dir:", mfccs_dir + folder_name)
        for ubm in list_ubm_files:  # iterating over the pretrained ubms
            n_ubm = util.extract_numbers_from_str(ubm)  # getting the number of ubms of the corresponding file
            print("\ni-vecs for {} GMMs".format(n_ubm))
            list_mfcc_files = util.traverse_dir(mfccs_dir + folder_name, '.mfcc')  # reading MFCCs to extracting i-vecs from
            extract_ivecs.compute_ivecs_pretr_ubms(list_mfcc_files, out_dir, info_num_feats_got=20, n_ubm=n_ubm,
                                                   file_ubm=ubm, ivec_dims=256, recipe=recipe, folder_name=folder_name)


def do_svm():
    print("to be implemented soon...")


def steps(i):
    switcher = {
        0: do_mfccs,
        1: do_fishers,
        2: do_ivecs,
        3: do_svm,
        4: do_ivecs_pretrained_mdls,
    }
    func = switcher.get(i)
    return func()


steps(4)
