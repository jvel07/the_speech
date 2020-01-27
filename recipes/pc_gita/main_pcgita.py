from data_preproc.mfccs import extract_mfccs
from data_preproc.fisher import extract_fishers
#from data_preproc.ivecs impor
import numpy as np
import os
from common import util


# Computes mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    audio_dir = '/opt/project/audio/'
    out_dir = '/opt/project/data/'

    list_sets = ['DDK_analysis', 'monologue', 'read_text', 'sentences', 'sentences2']
    for folder_name in list_sets:
        print("Reading dir:", folder_name)
        list_of_wavs = util.traverse_dir(audio_dir+folder_name, '.wav')
        #print(list_of_wavs[0])
        extract_mfccs.compute_mfccs(list_of_wavs, out_dir, num_mfccs=20, recipe='pcgita', folder_name=folder_name)


def do_fishers():
    recipe='pcgita'
    mfccs_dir = '/opt/project/data/{}/'.format(recipe)
    out_dir = '/opt/project/data/'
    file_ubm = '/opt/project/data/pcgita/DDK_analysis/mfccs_pcgita_20_DDK_analysis_2del.mfcc'

    list_sets = ['DDK_analysis', 'monologue', 'read_text', 'sentences', 'sentences2']
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


steps(1)
