import numpy as np
import pickle
import bob.kaldi
from bob.kaldi import io

#from bob.learn.linear import FisherLDATrainer


# Opening MFCCs from file
def read_mfcc(file_name):
    with open(file_name, 'rb') as f:
        mfccs = pickle.load(f)
        print("MFCCs loaded from:", file_name)
    return mfccs


# Train UBM
def train_models():
    # Train diagonal GMM
    print("Training " + str(num_gauss) + " diagonal-GMM with", mfccs_file_ubm)
    dubm = bob.kaldi.ubm_train(mfccs_wav_ubm, file_diag_ubm_model,
                               num_gauss=num_gauss, num_iters=num_iters,
                               num_gselect=np.log2(num_gauss))

    # Train full GMM
    print("Training full GMM...")
    fubm = bob.kaldi.ubm_full_train(mfccs_wav_ubm, dubm, file_full_ubm_model,
                                    num_iters=num_iters, num_gselect=np.log2(num_gauss))

    # Train ivector extractor
    print("Training i-vec extractor with " + str(ivector_dim) + " dimensions...")
    feats = [[]]
    feats = list_mfccs_ivecs
    ivector = bob.kaldi.ivector_train(feats, fubm, file_ivec_extractor_model,
                                      ivector_dim=ivector_dim,
                                      num_iters=num_iters, min_post=min_post,
                                      posterior_scale=post_scale)

    return dubm, fubm, ivector


def train_ivec_extractor(list_mfccs_ivecs):
    print("Training i-vec extractor with " + str(ivector_dim) + " dimensions...")
    ivector = bob.kaldi.ivector_train(list_mfccs_ivecs, fubm, file_ivec_extractor_model,
                                      ivector_dim=ivector_dim,
                                      num_iters=num_iters, min_post=min_post,
                                      posterior_scale=post_scale)

    return ivector

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


if __name__ == '__main__':

    set_ = 'train'
    set_models = 'train'
    obs = 'vad'
    obs_ivec = 'vad'

    num_gauss = 2
    num_iters = 200
    ivector_dim = 200
    min_post = 0.025
    post_scale = 1
    num_gselect = np.log2(num_gauss)

    #ivector_3D_file = '../data/ivectors3d-train'
    ivector_2D_file = '../data/ivecs/alzheimer/ivecs-' + str(num_gauss) + 'g-100i-{}-{}'.format(set_, obs_ivec)

    mfccs_file_ivec = '../data/mfccs/mfccs_dem_20_vad'#.format(set_)
    mfccs_file_ubm = '../data/mfccs/mfccs_ubm_bea_20_vad'

    file_diag_ubm_model = '../data/models/dem/dubm_mdl_{}g_dem_{}_{}'.format(num_gauss, set_models, obs)
    file_full_ubm_model = '../data/models/dem/fubm_mdl_{}g_dem_{}_{}'.format(num_gauss, set_models, obs)
    file_ivec_extractor_model = '../data/models/dem/ivec_mdl_{}g_dem_{}_{}'.format(num_gauss, set_models, obs)

    # Train models...
    if set_ == 'train':
        # Load MFCCs for UBM
        mfccs_wav_ubm = np.vstack(read_mfcc(mfccs_file_ubm))

        # Load MFCCs for i-vectors extraction
        list_mfccs_ivecs = read_mfcc(mfccs_file_ivec)

        # Train models
        dubm, fubm, ivector = train_models()

        # Save models
        save_models(file_diag_ubm_model, dubm,
                    file_full_ubm_model, fubm,
                    file_ivec_extractor_model, ivector)
    # ...or load models
    else:
        # Load MFCCs for i-vectors extraction
        list_mfccs_ivecs = read_mfcc(mfccs_file_ivec)

        # Load models
        dubm, fubm, ivector = load_models(file_diag_ubm_model, file_full_ubm_model,
                                          file_ivec_extractor_model)

    # Extract ivectors
    print("Extracting i-vecs...")
    ivectors_list = []
    for i2 in list_mfccs_ivecs:
        ivector_array = bob.kaldi.ivector_extract(i2, fubm, ivector, num_gselect=5)
        ivectors_list.append(ivector_array)
    a_ivectors = np.vstack(ivectors_list)
    #a_ivectors_3d = np.expand_dims(a_ivectors, axis=1)
    print("i-vectors shape:", a_ivectors.shape)

    # Save i-vectors to a txt file
    np.savetxt(ivector_2D_file, a_ivectors)
    print("i-vectors saved to:", ivector_2D_file)
