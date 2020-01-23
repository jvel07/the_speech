import os

import cyvlfeat as vlf

import numpy as np


def compute_fishers(list_sets, out_dir, num_feats_got_mfccs, file_ubm_mfccs, recipe, folder_name):
    # Loading File for UBM
    print("File of MFCCs for UBM:", file_ubm_mfccs)
    array_mfccs_1 = np.load(file_ubm_mfccs, allow_pickle=True)

    # convert list to array
    array_mfccs_1 = np.vstack(array_mfccs_1)

    # training GMM
    num_clusters = [2,4,8,16,32,64]
    for name in list_sets:
        # Input file (MFCCs)
        list_mfccs = np.load(name, allow_pickle=True)
        for g in num_clusters:
            means, covs, priors, LL, posteriors = vlf.gmm.gmm(array_mfccs_1, n_clusters=g, n_repetitions=2, verbose=0)  # UBM Train and Dev
            # fisher encoding
            list_fisher = []
            print("Ëxtracting {}-GMM fisher vecs from:".format(g), os.path.basename(name))
            for i in list_mfccs:
                # Extracting fisher vecs
                enc = vlf.fisher.fisher(i.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
                list_fisher.append(enc)
            fishers = np.vstack(list_fisher)
            # Output file (fishers)
            obs = '2del'
            file_fishers = out_dir + recipe + '/' + folder_name + 'fisher-{}-{}-{}g-{}.fish'.format(num_feats_got_mfccs, obs, g, name)
            np.savetxt(file_fishers, fishers, fmt='%.7f')
            print("Fishers saved to:", file_fishers)
