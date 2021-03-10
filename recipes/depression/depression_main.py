# from data_preproc.mfccs import extract_mfccs
# from data_preproc.fisher import extract_fishers
import bob
from bob.kaldi import io

from data_preproc.ivecs import extract_ivecs
import numpy as np
import os
from common import util

# Name of the task/recipe/dataset/etc.
import recipes.dementia_new8k.dementia_new8k_helper as ah

recipe = 'depression'

# Working directories
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
# in and out dirs
audio_dir = work_dir + 'audio/'

# List of audio-sets (folder(s) containing audio samples) i.e. train, dev, test...
list_sets = ['depression']

# Name of the folder containing the UBM (not the path)
ubm_folder_name = 'wav16k_split_long'

# List of number of clusters wanted to use
# list_n_clusters = [256]
# list_n_clusters = [2, 4, 8, 16, 32, 64, 128, 256]
list_n_clusters = [256]
# list_n_clusters = [64]


feature_dir = work_dir + 'data/{}/'.format(recipe)
out_dir = work_dir + 'data/'  # Where the computed features will be saved
ubm_dir = work_dir + 'data/UBMs/' + ubm_folder_name +'/ivec_models/'  # where the diagonal ubms live
# list_ubm_files = util.traverse_dir(ubm_dir, '.dubm')  #  reading all the files with .mdl or .dubm as format (latter is more reliable)

for g in list_n_clusters:
    # info-purpose parameters from the frame-level extracted features #
    feats_info_ubm = [20, 0, 'mfcc']  # info of the features (n_features/dimension, deltas, cepstral_type=choose between mfcc or plp)
    feats_info_flevel = [20, 0, 'fbank']  # info of the features (n_features/dimension, deltas, cepstral_type=choose between mfcc or plp)
    for folder_name in list_sets:  # iterating over the list of sets where the features live
        # print("\nReading dir:", feature_dir + folder_name+'/fisher/')
        # here the FULL-diag ubm is used
        list_ubm_files = util.traverse_dir_2(ubm_dir, '*_{}g_{}{}-{}del_{}.fubm'.format(g, feats_info_ubm[0],
                                                                                        feats_info_ubm[2],
                                                                                        feats_info_ubm[1],
                                                                                        ubm_folder_name))
        list_mfcc_files = util.traverse_dir_2(feature_dir + folder_name+'/', '*{}_{}_{}del.{}'.format(
                                                                                                  feats_info_flevel[0],
                                                                                                  folder_name,
                                                                                                  feats_info_flevel[1],
                                                                                                  feats_info_flevel[2]
        ))
        extract_ivecs.extract_ivecs(list_mfcc_files=list_mfcc_files, g=g, list_fubms=list_ubm_files,
                                    mfcc_info=feats_info_ubm, recipe=recipe, out_dir=out_dir, folder_name=folder_name)





fubmfile='/media/jose/hk-data/PycharmProjects/the_speech/data/UBMs/wav16k_split_long/ivec_models/fubm_mdl_256g_20mfcc-0del_wav16k_split_long.fubm'
with io.open_or_fd(fubmfile, mode='r') as fd:
    fubm = fd.read()

ivec_x = '/media/jose/hk-data/PycharmProjects/the_speech/data/UBMs/wav16k_split_long/ivec_models/ivec_mdl_256g_20mfcc-0del_demencia94ABC.ivexc'
with io.open_or_fd(fubmfile, mode='r') as fd:
    ivex = fd.read()

a = np.load('../data/depression/depression/fbank_depression_20_depression_0del.fbank', allow_pickle=True)

ivectors_list=[]
for i2 in a:  # extracting i-vecs
    ivector_array = bob.kaldi.ivector_extract(i2, fubm, ivex, num_gselect=int(np.log2(256)))
    ivectors_list.append(ivector_array)
a_ivectors = np.vstack(ivectors_list)
file = '../data/depression/depression/ivecs/ivecs-20mfcc-0del-256g-depression-2.ivecs'
np.savetxt(file, a_ivectors, fmt='%.7f')

with 0.01: 0.5645135111648696




