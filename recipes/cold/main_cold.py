from data_preproc.mfccs import extract_mfccs
from data_preproc.fisher import extract_fishers
#from data_preproc.ivecs impor
import numpy as np
import os
from common import util


# Computes mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    recipe='cold'
    audio_dir = '/opt/project/audio/'
    out_dir = '/opt/project/data/'

    list_sets = ['train', 'dev', 'test']
    for folder_name in list_sets:
        print("Reading dir:", folder_name)
        list_of_wavs = util.traverse_dir(audio_dir+folder_name, '.wav')
        #print(list_of_wavs[0])
        extract_mfccs.compute_mfccs(list_of_wavs, out_dir, num_mfccs=20, recipe=recipe, folder_name=folder_name)


def do_fishers():
    recipe='cold'
    mfccs_dir = '/opt/project/data/{}/'.format(recipe)
    out_dir = '/opt/project/data/'
    file_ubm = '/opt/project/data/cold/train/mfccs_cold_train_2del.mfcc'  # Format is: "featureType_recipeName_numberOfDeltas.mfcc"

    list_sets = ['train', 'dev', 'test']
    for folder_name in list_sets:
        print("Reading dir:", mfccs_dir+folder_name)
        list_mfcc_files = util.traverse_dir(mfccs_dir+folder_name, '.mfcc')
        extract_fishers.compute_fishers(list_mfcc_files, out_dir, num_feats_got_feats=20,
                                        file_ubm_feats=file_ubm, recipe=recipe, folder_name=folder_name)


def do_ivecs():
    print("fish")


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


steps(0)
steps(1)
