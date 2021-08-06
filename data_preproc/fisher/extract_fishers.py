import os
import re

import cyvlfeat as vlf

import numpy as np

from common import util
from kaldi_python.read_kaldi_gmm import get_diag_gmm_params


def do_gmm(features, num_gaussian):
    print("Training {}-GMM fishers...".format(num_gaussian))
    means, covs, priors, LL, posteriors = vlf.gmm.gmm(features, n_clusters=num_gaussian, n_repetitions=2, verbose=0)
    return means, covs, priors


def do_fishers(features, means, covs, priors):
    # Extracting fisher vecs
    print("Extracting FV encodings...")
    fish = vlf.fisher.fisher(features.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
    return fish


def compute_fishers(list_n_clusters, list_mfcc_files, out_dir, list_files_ubm, recipe, folder_name, feats_info):
    # Loading Files for UBM
    list_feats = []
    for file_ubm in list_files_ubm:
        print("File of features for the UBM:", file_ubm)
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

            # Files for storing the mean, covs and weights (priors) of the GMM
            file_means = out_dir + recipe + '/UBM/means_gmm_{}{}_{}del_{}g'.format(feats_info[0], feats_info[2],
                                                                                   feats_info[1], str(g))
            file_covs = out_dir + recipe + '/UBM/covs_gmm_{}{}_{}del_{}g'.format(feats_info[0], feats_info[2],
                                                                                 feats_info[1], str(g))
            file_priors = out_dir + recipe + '/UBM/priors_gmm_{}{}_{}del_{}g'.format(feats_info[0], feats_info[2],
                                                                                     feats_info[1], str(g))
            try:
                np.savetxt(file_means, means)
                np.savetxt(file_covs, covs)
                np.savetxt(file_priors, priors)
                print("UBM statistics saved successfully.")
            except:
                print("Error: couldn't save UBM statistics! Check that the path exists and try again.", )

            for feat in list_feat:  # iterating over the mfccs
                fish = vlf.fisher.fisher(feat.transpose(), means.transpose(), covs.transpose(), priors,
                                         square_root=True,
                                         normalized=True, improved=True)  # Extracting fishers from features
                list_fishers.append(fish)
            # Output file (fishers)
            obs = '{}del'.format(int(feats_info[1]))  # getting number of deltas info
            file_fishers = out_dir + recipe + '/' + folder_name + '/fisher/fisher-{}{}-{}-{}g-{}.fisher'.format(
                str(feats_info[0]), feats_info[2], obs, g, folder_name)
            # util.save_pickle(file_fishers, list_fishers)  # save as pickle
            np.savetxt(file_fishers, list_fishers, fmt='%.7f')  # save as txt
            print("{} fishers saved to:".format(len(list_fishers)), file_fishers, "with (1st ele.) shape:",
                  list_fishers[0].shape)
            print()


# When the GMM-diagonals and -variances are already provided.
regex = re.compile(
    r'\d+')  # to find the specific format file of the provided model (usually '.mdl'; per Kaldi's format)


def compute_fishers_pretr_ubm(list_mfcc_files, out_dir, file_ubm, recipe, folder_name):
    # Loading File for UBM
    print("File for UBM:", file_ubm)
    vars, means, weights, g = get_diag_gmm_params(file_diag=file_ubm, out_dir=out_dir)

    # print(list_feats[0])
    print("Fisher-vecs will be extracted using {} number of Gaussians!".format(g))
    for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
        list_feat = np.load(file_name, allow_pickle=True)  # this list should contain all the mfccs per FILE
        # for g in list_n_clusters:
        list_fishers = []
        # means, covs, priors = do_gmm(array_mfccs_ubm[:2000], g)  # training GMM (here not neccesary since we already have it)
        for feat in list_feat:  # iterating over the wavs (mfccs)
            fish = vlf.fisher.fisher(feat.transpose(), means[:, :60].transpose(), vars[:, :60].transpose(), weights,
                                     improved=True)
            list_fishers.append(fish)  # Extracting fishers from features
        # Output file (fishers)
        info_num_feats = regex.findall(file_name)
        obs = '{}del'.format(int(info_num_feats[1]))  # getting number of deltas info
        file_fishers = out_dir + recipe + '/' + folder_name + '/fisher/fisher-{}-{}-{}g-{}.fisher'.format(
            int(info_num_feats[0]), obs, g, folder_name)
        util.save_pickle(file_fishers, list_fishers)  # save as pickle
        # np.savetxt(file_fishers, list_fishers, fmt='%.7f')  # save as txt
        print("{} fishers saved to:".format(len(list_fishers)), file_fishers, "with (1st ele.) shape:",
              list_fishers[0].shape, "\n")


# When the GMM-diagonals and -variances are already provided.
def compute_fishers_pretr_ubm_2(list_mfcc_files, out_dir, list_files_ubm, recipe, folder_name, feats_info):
    for file_ubm in list_files_ubm:
        # Loading File for UBM
        print("File for UBM:", os.path.basename(file_ubm))
        parent_dir_ubm = os.path.basename(os.path.dirname(os.path.dirname(list_files_ubm[0])))
        vars, means, weights, g = get_diag_gmm_params(file_diag=file_ubm,
                                                      out_dir=out_dir + 'UBMs/' + parent_dir_ubm + '/GMM_fishers/')

        print("Fisher-vecs will be extracted using {} number of Gaussians!".format(g))
        for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
            list_feat = np.load(file_name, allow_pickle=True)  # this list should contain all the mfccs per FILE
            list_fishers = []
            for feat in list_feat:  # iterating over the wavs (mfccs)
                # Extracting fishers from features
                fish = vlf.fisher.fisher(feat, means, vars, weights, improved=True)
                list_fishers.append(fish)
            # Output file (fishers)
            # getting info about the number of frame-level feats and the deltas used (for naming the output files)
            # info_num_feats = regex.findall(os.path.basename(file_name))
            file_fishers = out_dir + recipe + '/' + folder_name + '/fisher/fisher-{}{}-{}del-{}g-{}.fisher'.format(
                feats_info[0], feats_info[2],
                feats_info[1], g, folder_name)
            # util.save_pickle(file_fishers, list_fishers)  # save as pickle
            np.savetxt(file_fishers, list_fishers)#, fmt='%.7f')  # save as txt
            # print("{} fishers saved to:".format(len(list_fishers)), file_fishers, "with (1st ele.) shape:", list_fishers[0].shape, "\n")
