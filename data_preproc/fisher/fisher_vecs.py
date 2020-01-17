import os

import cyvlfeat as vlf

import numpy as np

# Loading Files
list_sets = ['train', 'dev', 'test']
num_feats = 13

file_mfccs_1 = 'C:/Users/Win10/PycharmProjects/the_speech/data/mfccs/cold/mfccs_cold_{}_train_2del'.format(num_feats)
array_mfccs_1 = np.load(file_mfccs_1, allow_pickle=True)
file_mfccs_2 = 'C:/Users/Win10/PycharmProjects/the_speech/data/mfccs/cold/mfccs_cold_{}_dev_2del'.format(num_feats)
array_mfccs_2 = np.load(file_mfccs_2, allow_pickle=True)

# convert list of mfccs to array
array_mfccs_1 = np.vstack(array_mfccs_1)
array_mfccs_2 = np.vstack(array_mfccs_2)
array_mfccs_3 = np.concatenate((array_mfccs_1, array_mfccs_2))
print(array_mfccs_3.shape)

# training GMM
num_clusters = [2,4,8, 16,32,64]
for name in list_sets:
    # Input file (MFCCs)
    file_mfccs = 'C:/Users/Win10/PycharmProjects/the_speech/data/mfccs/cold/mfccs_cold_{}_{}_2del'.format(num_feats, name)
    list_mfccs = np.load(file_mfccs, allow_pickle=True)
    for g in num_clusters:
        means, covs, priors, LL, posteriors = vlf.gmm.gmm(array_mfccs_3, g)
        # fisher encoding
        list_fisher = []
        print("Ëxtracting {}-GMM fisher vecs from:".format(g), os.path.basename(file_mfccs), "Number of features:", len(list_mfccs))
        for i in list_mfccs:
            # Extracting fisher vecs
            enc = vlf.fisher.fisher(i.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
            list_fisher.append(enc)
        fishers = np.vstack(list_fisher)
        # Saving fishers
        obs = '2del_tdubm'
        # Output file (fishers)
        file_fishers = 'D:/fishers/fisher-{}-{}-{}-{}'.format(num_feats, obs, g, name)
        np.savetxt(file_fishers, fishers)
        print("Fishers saved to:", file_fishers)
