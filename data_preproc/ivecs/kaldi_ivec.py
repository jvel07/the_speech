import numpy as np
import pickle
import bob.kaldi

from common import util


def train_models(mfccs_ubm, list_mfccs_ivecs, diag, full, ivec_mdl, num_gauss, idim):
    num_iters = 200
    ivector_dim = idim
    min_post = 0.025
    post_scale = 1
    # Train diagonal GMM
    print("\nTraining " + str(num_gauss) + " diagonal-GMM")
    dubm = bob.kaldi.ubm_train(mfccs_ubm, diag,
                               num_gauss=num_gauss, num_iters=num_iters,
                               num_gselect=int(np.log2(num_gauss)))

    # Train full GMM
    print("Training full GMM...")
    fubm = bob.kaldi.ubm_full_train(mfccs_ubm, dubm, full,
                                    num_iters=num_iters, num_gselect=int(np.log2(num_gauss)))

    # Train ivector extractor
    print("Training i-vec extractor with " + str(ivector_dim) + " dimensions...")
    feats = [[]]
    feats = list_mfccs_ivecs
    ivector = bob.kaldi.ivector_train(feats, fubm, ivec_mdl,
                                      ivector_dim=ivector_dim,
                                      num_iters=num_iters, min_post=min_post,
                                      posterior_scale=post_scale)

    return dubm, fubm, ivector


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


def main():
    work_dir = '/home/jose/PycharmProjects/the_speech'

    set_ = ''
    set_models = ''
    obs = '1del'
    obs_ivec = '1del'
    num_mfccs = '20'
    ivecs_dim = 256

    # ---Input Files---
    # MFCCs
    file_mfccs_ivec = work_dir + '/data/mfccs/dem/mfccs_dem_{}_{}'.format(num_mfccs, obs)
    file_mfccs_ubm = work_dir + '/data/mfccs/dem/mfccs_ubm_dem_{}_{}'.format(num_mfccs, obs)
    # Load MFCCs for UBM
    mfccs_wav_ubm = np.vstack(util.read_pickle(file_mfccs_ubm))
    # Load MFCCs for i-vectors extraction
    list_mfccs_ivecs = util.read_pickle(file_mfccs_ivec)
    # group per type (original, noised, stretched, pitched) corresponding to each spk.
    # and join (concatenate) 3 wavs per speaker
    # list_mfccs_joint = util.join_speakers_wavs(util.group_per_audio_type(list_mfccs_ivecs))

    num_gauss = [2, 4, 8, 16, 32, 64, 128]
    for g in num_gauss:
        # ---OUTPUT FILES---
        # i-vecs
        ivector_2D_file = work_dir + '/data/ivecs/alzheimer/ivecs-' + str(g) + '-{}mf-100i--{}'.format(num_mfccs, obs_ivec)
        # models for i-vecs
        file_diag_ubm_model = work_dir + '/data/models/dem/dubm_mdl_{}g_dem_{}'.format(g, obs)
        file_full_ubm_model = work_dir + '/data/models/dem/fubm_mdl_{}g_dem_{}'.format(g, obs)
        file_ivec_extractor_model = work_dir + '/data/models/dem/ivec_mdl_{}g_dem_{}'.format(g, obs)
        # Train models
        model_dubm, model_fubm, model_ivector = train_models(mfccs_wav_ubm, list_mfccs_ivecs, file_diag_ubm_model,
                                                             file_full_ubm_model, file_ivec_extractor_model, g, ivecs_dim)

        # Extract ivectors
        print("Extracting i-vecs...")
        ivectors_list = []
        n_gselect = int(np.log2(g))
        print(n_gselect)
        for i2 in list_mfccs_ivecs:
            ivector_array = bob.kaldi.ivector_extract(i2, model_fubm, model_ivector, num_gselect=n_gselect)
            ivectors_list.append(ivector_array)
        a_ivectors = np.vstack(ivectors_list)
        # a_ivectors_3d = np.expand_dims(a_ivectors, axis=1)
        print("i-vectors shape:", a_ivectors.shape)

        # Save i-vectors to a txt file
        np.savetxt(ivector_2D_file, a_ivectors)
        print("i-vectors saved to:", ivector_2D_file)


if __name__ == '__main__':
    main()
