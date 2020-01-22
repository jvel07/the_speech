from data_preproc.mfccs import extract_mfccs
#from data_preproc.fisher import fisher_vecs
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


def do_ivecs():
    print("ivecs")


def do_fishers():
    print("fish")


def indirect(i):
    switcher = {
        0: do_mfccs,
        1: do_ivecs,
        2: do_fishers
    }
    func = switcher.get(i)
    return func()


indirect(0)
