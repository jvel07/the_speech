import numpy as np
import pickle
import bob.kaldi
from bob.kaldi import io
import re
import os

from common import util


def train_models(mfccs_ubm, feats_ivexc, diag, full, ivec_mdl, num_gauss, ivector_dim):
    num_iters = 100
    min_post = 0.025
    post_scale = 1
    # Train diagonal GMM
    print("Training " + str(num_gauss) + " diagonal-GMM...")
    dubm = bob.kaldi.ubm_train(mfccs_ubm, diag, num_threads=8,
                               num_gauss=num_gauss, num_iters=num_iters,
                               num_gselect=int(np.log2(num_gauss)))

    # Train full GMM
    print("Training full GMM...")
    fubm = bob.kaldi.ubm_full_train(mfccs_ubm, dubm, full,
                                    num_iters=num_iters, num_gselect=int(np.log2(num_gauss)))

    # Train ivector extractor
    print("Training i-vec extractor with " + str(ivector_dim) + " dimensions...")
    feats = [[]]
    feats = feats_ivexc
    ivector = bob.kaldi.ivector_train(feats, fubm, ivec_mdl,
                                      ivector_dim=ivector_dim,
                                      num_iters=num_iters, min_post=min_post,
                                      posterior_scale=post_scale)

    return dubm, fubm, ivector


def train_ubm_models(mfccs_ubm, diag, full, num_gauss):
    num_iters = 100
    # Train diagonal GMM
    print("Training " + str(num_gauss) + " diagonal-GMM...")
    dubm = bob.kaldi.ubm_train(mfccs_ubm, diag, num_threads=8,
                               num_gauss=num_gauss, num_iters=num_iters,
                               num_gselect=int(np.log2(num_gauss)))

    # Train full GMM
    print("Training full GMM...")
    fubm = bob.kaldi.ubm_full_train(mfccs_ubm, dubm, full,
                                    num_iters=num_iters, num_gselect=int(np.log2(num_gauss)))

    return dubm, fubm

def train_ivec_extractor(ivector_dim, feats_ivexc, fubm, ivec_mdl, ):
    num_iters = 100
    min_post = 0.025
    post_scale = 1
    # Train ivector extractor
    print("Training i-vec extractor with " + str(ivector_dim) + " dimensions...")
    feats = [[]]
    feats = feats_ivexc
    ivector_extractor = bob.kaldi.ivector_train(feats, fubm, ivec_mdl,
                                      ivector_dim=ivector_dim,
                                      num_iters=num_iters, min_post=min_post,
                                      posterior_scale=post_scale)

    return ivector_extractor


# Extracting i-vecs when given the fubm and the ivec_extractor
def extract_ivecs(list_mfcc_files, g, model_fubm, model_ivector, mfcc_info, dest_data_dir_ivecs, folder_name):
    for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
        list_feat = np.load(file_name, allow_pickle=True)  # this list should contain all the mfccs per FILE
        print("Extracting i-vecs from {}".format(file_name))
        ivectors_list = []
        n_gselect = int(np.log2(g))
        print(n_gselect)
        for i2 in list_feat:  # extracting i-vecs
            ivector_array = bob.kaldi.ivector_extract(i2, model_fubm, model_ivector, num_gselect=n_gselect)
            ivectors_list.append(ivector_array)
        a_ivectors = np.vstack(ivectors_list)
        print("i-vectors shape:", a_ivectors.shape)
        # Output file (i-vectors)
        obs = '{}del'.format(mfcc_info[1])  # getting number of deltas info
        # output file for the ivecs
        file_fishers = dest_data_dir_ivecs + '/ivecs-{}{}-{}-{}g-{}.ivecs'.format(str(mfcc_info[0]), mfcc_info[2], obs, g,
                                                                                  folder_name)
        np.savetxt(file_fishers, a_ivectors, fmt='%.7f')
        print("{} ivecs saved to:".format(len(a_ivectors)), file_fishers, "with shape:", a_ivectors.shape, '\n')


regex = re.compile(r'\d+') # to get the number of gaussians when reading the txt models
# Use this when there already exists the fubm trained
def compute_ivecs_pretr_ubms(list_mfcc_files, out_dir, file_ubm, recipe, folder_name):
    num_iters = 100
    min_post = 0.025
    post_scale = 1

    gaussians = int(regex.findall(file_ubm)[0])

    # ---Input Files---
    # Loading File for UBM
    obs_ivec = ''
    print("File of MFCCs for UBM:", file_ubm)
    with io.open_or_fd(file_ubm, mode='r') as fd:
        fubm = fd.read()

    for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
        list_feat = np.load(file_name, allow_pickle=True)  #  this list should contain all the mfcc-features per FILE
        # models for i-vecs
        file_ivec_extractor_model =out_dir + recipe + '/' + folder_name + '/ivec_mdl_{}g_dem_{}'.format(gaussians, obs_ivec)
        ivec_dims = np.log2(gaussians)*(len(list_feat[0][1])) # the ivec dims is given by log2(numgaussians) * mfcc features dim
        # Train ivector extractor
        print("Training i-vec extractor with " + str(ivec_dims) + " dimensions...")
        # feats = [[]]
        # feats = list_feat
        model_ivector = bob.kaldi.ivector_train(list_feat, fubm, file_ivec_extractor_model,
                                              ivector_dim=256,#int(ivec_dims),
                                              num_iters=num_iters, min_post=min_post,
                                              posterior_scale=post_scale)
        # Extract ivectors
        print("Extracting i-vecs...")
        ivectors_list = []
        n_gselect = int(np.log2(gaussians))
        print(n_gselect)
        for i2 in list_feat:  # extracting i-vecs
            ivector_array = bob.kaldi.ivector_extract(i2, fubm, model_ivector)#, num_gselect=5)
            ivectors_list.append(ivector_array)
        a_ivectors = np.vstack(ivectors_list)
        print("i-vectors shape:", a_ivectors.shape)
        # Save i-vectors to a txt file
        obs = '2del'
        file_ivecs = out_dir + recipe + '/' + folder_name + '/ivecs-{}mf-{}-{}g-{}.ivecs'.format(
            len(list_feat[0][1]), obs, gaussians, folder_name)
        np.savetxt(file_ivecs, a_ivectors, fmt='%.7f')
        print("i-vectors saved to:", file_ivecs)


# trains dubm, fubm, ivec_extractor and computes ivectors
def compute_ivecs(list_n_gauss, list_mfcc_files, out_dir, list_files_ubm, recipe, folder_name, mfcc_info):
    # getting name of dirs
    parent_dir_ubm = os.path.basename(os.path.dirname(list_files_ubm[0]))  # ...For naming properly the models' files only
    dest_data_dir_ivecs = out_dir + recipe + '/' + folder_name
    dest_data_dir_models = out_dir + 'UBMs/' + parent_dir_ubm

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

    print("i-vecs will be extracted using {} Gauss".format(list_n_gauss))
    for file_name in list_mfcc_files:  # This list should contain the mfcc FILES within folder_name
        list_feat = np.load(file_name, allow_pickle=True)  # this list should contain all the mfccs per FILE
        for g in list_n_gauss:
            # models for i-vecs
            file_diag_ubm_model = dest_data_dir_models + '/ivec_models/dubm_mdl_{}g_{}{}-{}del_{}.dubm'.format(g, mfcc_info[0], mfcc_info[2], mfcc_info[1], parent_dir_ubm)
            file_full_ubm_model = dest_data_dir_models + '/ivec_models/fubm_mdl_{}g_{}{}-{}del_{}.fubm'.format(g, mfcc_info[0], mfcc_info[2], mfcc_info[1], parent_dir_ubm)
            file_ivec_extractor_model = dest_data_dir_models + '/ivec_models/ivec_mdl_{}g_{}{}-{}del_{}.ivexc'.format(g, mfcc_info[0], mfcc_info[2], mfcc_info[1], parent_dir_ubm)
            if not os.path.isdir(dest_data_dir_models + '/ivec_models/'):
                os.mkdir(dest_data_dir_models + '/ivec_models/')
            # Train models
            ivec_dims = np.log2(g) * (len(list_feat[0][1]))
            model_dubm, model_fubm, model_ivector = train_models(np.vstack(array_mfccs_ubm), file_diag_ubm_model,
                                                                 file_full_ubm_model, file_ivec_extractor_model, g,
                                                                 int(ivec_dims))

            # Extract ivectors
            print("Extracting i-vecs...")
            ivectors_list = []
            n_gselect = int(np.log2(g))
            print(n_gselect)
            for i2 in list_feat: # extracting i-vecs
                ivector_array = bob.kaldi.ivector_extract(i2, model_fubm, model_ivector, num_gselect=n_gselect)
                ivectors_list.append(ivector_array)
            a_ivectors = np.vstack(ivectors_list)
            print("i-vectors shape:", a_ivectors.shape)
            # Output file (i-vectors)
            # info_num_feats = regex.findall(file_name)
            obs = '{}del'.format(mfcc_info[1])  # getting number of deltas info
            # output file for the ivecs
            file_fishers = dest_data_dir_ivecs + '/ivecs-{}{}-{}-{}g-{}.ivecs'.format(str(mfcc_info[0]), mfcc_info[2], obs, g, folder_name)
            np.savetxt(file_fishers, a_ivectors, fmt='%.7f')
            print("{} ivecs saved to:".format(len(a_ivectors)), file_fishers, "with shape:", a_ivectors.shape, '\n')


# trains UBMs only (dubm and fubm)
def train_ubm_only(list_n_gauss, out_dir, list_files_ubm, feats_info):
    # getting name of dirs
    parent_dir_ubm = os.path.basename(os.path.dirname(list_files_ubm[0]))  # ...For properly naming the models' files
    dest_data_dir_models = out_dir + 'UBMs/' + parent_dir_ubm
    if not os.path.isdir(dest_data_dir_models):
        os.mkdir(dest_data_dir_models)

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

    print("UBM will be trained using {} Gauss".format(list_n_gauss))
    for g in list_n_gauss:
        # models for i-vecs
        file_diag_ubm_model = dest_data_dir_models + '/ivec_models/dubm_mdl_{}g_{}{}-{}del_{}.dubm'.format(g,
                                                                                                           feats_info[
                                                                                                               0],
                                                                                                           feats_info[
                                                                                                               2],
                                                                                                           feats_info[
                                                                                                               1],
                                                                                                           parent_dir_ubm)
        file_full_ubm_model = dest_data_dir_models + '/ivec_models/fubm_mdl_{}g_{}{}-{}del_{}.fubm'.format(g,
                                                                                                           feats_info[
                                                                                                               0],
                                                                                                           feats_info[
                                                                                                               2],
                                                                                                           feats_info[
                                                                                                               1],
                                                                                                           parent_dir_ubm)
        if not os.path.isdir(dest_data_dir_models + '/ivec_models/'):
            os.mkdir(dest_data_dir_models + '/ivec_models/')
        # Train models
        train_ubm_models(mfccs_ubm=np.vstack(array_mfccs_ubm), diag=file_diag_ubm_model, full=file_full_ubm_model, num_gauss=g)


# Save models
def save_models(file_diag_ubm_model, diag_mdl,
                file_full_ubm_model, full_mdl,
                file_ivec_extractor_model, ivec_mdl):
    with open(file_diag_ubm_model, 'wb') as fp1:
        pickle.dump(diag_mdl, fp1)
    print("Diagonal model saved to:", file_diag_ubm_model)

    with open(file_full_ubm_model, 'wb') as fp2:
        pickle.dump(full_mdl, fp2)
    print("Full model saved to:", file_full_ubm_model)

    with open(file_ivec_extractor_model, 'wb') as fp3:
        pickle.dump(ivec_mdl, fp3)
    print("i-vector extractor model saved to:", file_ivec_extractor_model)


def load_models(diag_ubm, full_ubm, ivec_extr):
    with open(diag_ubm, 'rb') as f1:
        mdl_diag = pickle.load(f1)
        print(diag_ubm, "loaded successfully.")

    with open(full_ubm, 'rb') as f2:
        mdl_full = pickle.load(f2)
        print(full_ubm, "loaded successfully.")

    with open(ivec_extr, 'rb') as f3:
        mdl_ivec = pickle.load(f3)
        print(ivec_extr, "loaded successfully.")
    return mdl_diag, mdl_full, mdl_ivec