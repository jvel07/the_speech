import os

import cyvlfeat as vlf
from common import util
import numpy as np

# Loading Files
list_sets = ['train', 'dev', 'test']
num_feats = 20

file_mfccs_bea = '/home/egasj/PycharmProjects/the_speech/data/mfccs/cold/mfccs_cold_train_{}_'.format(num_feats)
array_mfccs_bea = np.load(file_mfccs_bea, allow_pickle=True)

# convert list of mfccs to array
# array_mfccs = np.vstack(list_mfccs)
#array_mfccs_bea = np.vstack(array_mfccs_bea)

# training GMM
num_clusters = [2, 4, 8, 16, 32, 64, 128]
for name in list_sets:
    file_mfccs = '/home/egasj/PycharmProjects/the_speech/data/mfccs/cold/mfccs_cold_{}_{}_'.format(name, num_feats)
    list_mfccs = np.load(file_mfccs, allow_pickle=True)
    for g in num_clusters:
        means, covs, priors, LL, posteriors = vlf.gmm.gmm(array_mfccs_bea, g)
        # fisher encoding
        list_fisher = []
        for i in list_mfccs:
            # Extracting fisher vecs
            print("Ëxtracting fisher vecs from:", os.path.basename(file_mfccs), "Number of features:", len(list_mfccs))
            enc = vlf.fisher.fisher(i.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
            list_fisher.append(enc)
        fishers = np.vstack(list_fisher)
        # Saving fishers
        obs = '20mf_2d'
        file_fishers = 'C:/Users/Win10/PycharmProjects/the_speech/data/fisher_vecs/fisher-{}-{}'.format(g, obs)
        np.savetxt(file_fishers, fishers)
        print("Fishers saved to:", file_fishers)
