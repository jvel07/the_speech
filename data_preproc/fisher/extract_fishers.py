import os
import re

import cyvlfeat as vlf

import numpy as np

from kaldi_python.read_kaldi_gmm import get_diag_gmm_params


def do_gmm(features, num_gaussian):
    print("Training {}-GMM fisher's.".format(num_gaussian))
    means, covs, priors, LL, posteriors = vlf.gmm.gmm(features, init_mode='kmeans', n_clusters=num_gaussian, n_repetitions=2, verbose=0)
    return means, covs, priors


def do_fishers(features, means, covs, priors):
    # Extracting fisher vecs
    fish = vlf.fisher.fisher(features.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
    return fish

regex = re.compile(r'\d+')
def compute_fishers(list_n_clusters, list_mfcc_files, out_dir, list_files_ubm, recipe, folder_name):
    # Loading Files for UBM
    list_feats = []
    for file_ubm in list_files_ubm:
        print("File of MFCCs for UBM:", file_ubm)
        array_feats = np.load(file_ubm, allow_pickle=True)
        # convert list to array
        array_feats = np.vstack(array_feats)
        list_feats.append(array_feats)
    array_mfccs_ubm = np.vstack(list_feats)
    print("Shape of the UBM:", array_mfccs_ubm.shape)
    del list_feats, array_feats

    # print(list_feats[0])
    print("Fisher-vecs will be extracted using {} number of Gaussians!".format(list_n_clusters))
    for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
        list_feat = np.load(file_name, allow_pickle=True)  # this list should contain all the mfccs per FILE
        for g in list_n_clusters:
            list_fishers = []
            means, covs, priors = do_gmm(array_mfccs_ubm, g)  # training GMM
            for feat in list_feat:  # iterating over the mfccs
                fish = vlf.fisher.fisher(feat.transpose(), means.transpose(), covs.transpose(), priors, square_root=True,
                                         normalized=True, improved=True)  # Extracting fishers from features
                list_fishers.append(fish)
            # Output file (fishers)
            info_num_feats = regex.findall(file_name)
            obs = '{}del'.format(int(info_num_feats[1]))  # getting number of deltas info
            file_fishers = out_dir + recipe + '/' + folder_name + '/fisher-{}mf-{}-{}g-{}.fisher'.format(
                int(info_num_feats[0]), obs, g, folder_name)
            np.savetxt(file_fishers, list_fishers, fmt='%.7f')
            print("{} fishers saved to:".format(len(list_fishers)), file_fishers, "with (1st ele.) shape:", list_fishers[0].shape)


def compute_fishers_pretr_ubm(list_mfcc_files, out_dir, file_ubm, recipe, folder_name):
    # Loading File for UBM
    print("File for UBM:", file_ubm)
    vars, means, weights, g = get_diag_gmm_params(file_ubm, out_dir)

    # print(list_feats[0])
    print("Fisher-vecs will be extracted using {} number of Gaussians!".format(g))
    for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
        list_feat = np.load(file_name, allow_pickle=True)  #  this list should contain all the mfccs per FILE
        # for g in list_n_clusters:
        list_fishers = []
        # means, covs, priors = do_gmm(array_mfccs_ubm[:2000], g)  # training GMM
        for feat in list_feat:  # iterating over the wavs (mfccs)
            fish = vlf.fisher.fisher(feat.transpose(), means[:,:40].transpose(), vars[:,:40].transpose(), weights, improved=True)
            list_fishers.append(fish)  # Extracting fishers from features
        # Output file (fishers)
        info_num_feats = regex.findall(file_name)
        obs = '{}del'.format(int(info_num_feats[1]))  # getting number of deltas info
        file_fishers = out_dir + recipe + '/' + folder_name + '/fisher-{}mf-{}-{}g-{}.fisher'.format(
            int(info_num_feats[0]), obs, g, folder_name)
        np.savetxt(file_fishers, list_fishers, fmt='%.7f')
        print("{} fishers saved to:".format(len(list_fishers)), file_fishers, "with (1st ele.) shape:", list_fishers[0].shape, "/n")