import glob
import os


import yaml
from tqdm import tqdm

import sys
# sys.path.append('/media/jvel/data/PycharmProjects/the_speech')
sys.path.append('/home/user/Documents/code/the_speech/')

from data_preproc.mfccs import extract_mfccs
from data_preproc.fisher import extract_fishers
# from data_preproc.ivecs import extract_ivecs
import numpy as np
from common import util



# Name of the task/recipe/dataset/etc.
import recipes.dementia_new8k.dementia_new8k_helper as ah

recipe = 'requests'

# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/'  # for titan x machine (normal)
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
# in and out dirs
audio_dir = '/media/jvel/data/audio/ComParE2023-HCC/'
out_dir = '../../data/'

# List of audio-sets (folder(s) containing audio samples) i.e. train, dev, test...
list_sets = ['train', 'dev', 'test']

# Name of the folder containing the UBM (not the path)
# ubm_folder_name = 'bea_wav8k'

# List of number of clusters wanted to use
# list_n_clusters = [256]
# list_n_clusters = [2, 4, 8, 16, 32, 64, 128]
list_n_clusters = [32, 64, 128]
# list_n_clusters = [256]
# list_n_clusters = [64]

with open("../../conf/requests_flevel.yml") as f:
    config = yaml.safe_load(f)


# Computes mfccs from wavs existing in the directories provided by the user
def do_mfccs():
    cepstral_type = "fbank"  # choose between "mfcc" or "fbank"
    params = config[cepstral_type]

    print("=======Frame-level extraction phase========")
    for dataset_folder in tqdm(list_sets, desc="Extracting frame level features", position=0):
        # Loading id-wavs specified in the labels file
        print("Extracting from: {0}. Using deltas={1}".format(dataset_folder, params['deltas']))
        source_file = '../../data/{}/labels/labels.csv'.format(recipe)
        list_of_wavs = util.traverse_dir(audio_dir + dataset_folder, file_type='.wav')
        list_of_wavs.sort()

        extract_mfccs.speechbrain_flevel(list_of_wavs, out_dir, recipe, dataset_folder, cepstral_type, **params)


def do_fishers():
    print("=======fisher-vector extraction phase========")
    cepstral_type = "mfcc"  # choose between "mfcc" or "fbank"
    params = config[cepstral_type]
    observation = 'Deltas{}'.format(str(params['deltas']))
    # feature_dir = '/media/jvel/data/features/{}/'.format(recipe)
    feature_dir = '/home/user/data/features/{}/'.format(recipe)

    # Format is: "featureType_recipeName_nMFCCs_nDeltas.mfcc"
    # this is: ../data/recipe_name/train/mfcc/40_mfcc_DeltasTrue
    # this is: ../data/recipe_name/train/fbank/DeltasTrue
    n_feats = params['n_mfcc'] if cepstral_type == 'mfcc' else params['n_mels']
    feats_info = [n_feats, observation, cepstral_type]  # needed termporarily... to be removed
    # if cepstral_type == 'mfcc':
    path_ubm_files = os.path.join(feature_dir, 'train/{0}/{1}_{2}_{3}'.format(cepstral_type, n_feats,
                                                                              cepstral_type, observation))
    # elif cepstral_type == 'fbank':
    #     path_ubm_files = os.path.join(feature_dir, 'train/{0}/{1}'.format(cepstral_type, observation))
    list_files_ubm = glob.glob(path_ubm_files + '/*.{}'.format(cepstral_type))

    # take random 1500 files from the train set to compute the UBM
    size = 1500  # (12.5 hours of speech)
    random_numbers = np.random.choice(len(list_files_ubm), size, replace=False)
    list_array = np.array(list_files_ubm)
    list_files_ubm_selected = list_array[random_numbers]

    for dataset_folder in list_sets:
        print("Extracting from: {0}. Using deltas={1}".format(dataset_folder, params['deltas']))
        path_flevel_files = os.path.join(feature_dir, '{4}/{0}/{1}_{2}_{3}'.format(cepstral_type, n_feats,
                                                                                   cepstral_type, observation,
                                                                                   dataset_folder))
        print(path_flevel_files)
        list_mfcc_files = glob.glob(path_flevel_files + '/*.{}'.format(cepstral_type))
        list_mfcc_files.sort()
        print(len(list_mfcc_files))
        extract_fishers.compute_fishers(list_n_clusters, list_mfcc_files, feature_dir, feats_info=feats_info,
                                        list_files_ubm=list_files_ubm_selected, recipe=recipe,
                                        folder_name=dataset_folder)


def do_svm():
    print("svm")


def steps(i):
    switcher = {
        0: do_mfccs,
        1: do_fishers,
        3: do_svm,
    }
    func = switcher.get(i)
    return func()


# steps(0)
steps(1)
# steps(2)
