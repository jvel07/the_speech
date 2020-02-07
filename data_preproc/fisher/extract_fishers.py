import os

import cyvlfeat as vlf

import numpy as np


def do_gmm(features, num_gaussian):
    print("Training {}-GMM fisher's.".format(num_gaussian))
    means, covs, priors, LL, posteriors = vlf.gmm.gmm(features, n_clusters=num_gaussian, n_repetitions=1, verbose=0)
    return means, covs, priors


def do_fishers(features, means, covs, priors):
    # Extracting fisher vecs
    fish = vlf.fisher.fisher(features.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
    return fish


def compute_fishers(list_n_clusters, list_mfcc_files, out_dir, num_feats_got_feats, file_ubm_feats, recipe, folder_name):
    # Loading File for UBM
    print("File of MFCCs for UBM:", file_ubm_feats)
    array_mfccs_ubm = np.load(file_ubm_feats, allow_pickle=True)
    # convert list to array
    array_mfccs_ubm = np.vstack(array_mfccs_ubm)

    # print(list_feats[0])
    print("Fisher-vecs will be extracted using {} number of Gaussians!".format(list_n_clusters))
    for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
        list_feat = np.load(file_name, allow_pickle=True)  # this list should contain all the mfccs per FILE
        for g in list_n_clusters:
            list_fishers = []
            means, covs, priors = do_gmm(array_mfccs_ubm, g)  # training GMM
            for feat in list_feat:  # iterating over the wavs (mfccs)
                #print("Features shape:", feat.shape)
                fish = vlf.fisher.fisher(feat.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
                list_fishers.append(fish)  # Extracting fishers from features
                # Output file (fishers)
            obs = '2del'
            file_fishers = out_dir + recipe + '/' + folder_name + '/fisher-{}mf-{}-{}g-{}.fish'.format(
                num_feats_got_feats,
                obs, g, folder_name)
            np.savetxt(file_fishers, list_fishers, fmt='%.7f')
            print("{} fishers saved to:".format(len(list_fishers)), file_fishers, "with (1st ele.) shape:", list_fishers[0].shape)
