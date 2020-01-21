import os

import cyvlfeat as vlf

import numpy as np


def ubm_training_type(i):
    switcher = {
        0: 'array_mfccs_1',  # train UBM with training data only
        1: 'array_mfccs_2',  # train UBM with dev data only
        2: 'array_mfccs_3',  # train UBM with training adn dev data
    }
    return switcher.get(i, "Invalid training type.")


def extract_fishers(list_sets, num_feats_got_mfccs, training_type=0):
    # Loading Files
    list_sets = list_sets
    num_feats = num_feats_got_mfccs

    file_mfccs_1 = 'C:/Users/Win10/PycharmProjects/the_speech/data/hc/cold/mfccs_cold_{}_train_2del'.format(num_feats)
    array_mfccs_1 = np.load(file_mfccs_1, allow_pickle=True)
    file_mfccs_2 = 'C:/Users/Win10/PycharmProjects/the_speech/data/hc/cold/mfccs_cold_{}_dev_2del'.format(num_feats)
    #array_mfccs_2 = np.load(file_mfccs_2, allow_pickle=True)

    # convert list to array
    array_mfccs_1 = np.vstack(array_mfccs_1)
    #array_mfccs_2 = np.vstack(array_mfccs_2)
    #array_mfccs_3 = np.concatenate((array_mfccs_1, array_mfccs_2))
    #print(array_mfccs_3.shape)

    # training GMM
    num_clusters = [2,4,8,16,32,64]
    for name in list_sets:
        # Input file (MFCCs)
        file_mfccs = 'C:/Users/Win10/PycharmProjects/the_speech/data/hc/cold/mfccs_cold_{}_{}_2del'.format(num_feats,
                                                                                                              name)
        list_mfccs = np.load(file_mfccs, allow_pickle=True)
        for g in num_clusters:
            # means, covs, priors, LL, posteriors = vlf.gmm.gmm(array_mfccs_1, n_clusters=g, n_repetitions=2, verbose=0)  # UBM Train only
            # means, covs, priors, LL, posteriors = vlf.gmm.gmm(array_mfccs_2, n_clusters=g, n_repetitions=2, verbose=0)  # UBM Dev only
            means, covs, priors, LL, posteriors = vlf.gmm.gmm(eval(ubm_training_type(training_type)), n_clusters=g, n_repetitions=2, verbose=0)  # UBM Train and Dev
            # fisher encoding
            list_fisher = []
            print("Ëxtracting {}-GMM fisher vecs from:".format(g), os.path.basename(file_mfccs), "Number of files:",
                  len(list_mfccs))
            for i in list_mfccs:
                # Extracting fisher vecs
                enc = vlf.fisher.fisher(i.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
                list_fisher.append(enc)
            fishers = np.vstack(list_fisher)
            # Output file (fishers)
            obs = '2del_tdubm'
            file_fishers = 'D:/fishers/fisher-{}-{}-{}-{}'.format(num_feats, obs, g, name)
            np.savetxt(file_fishers, fishers, fmt='%.7f')
            print("Fishers saved to:", file_fishers)
