import cyvlfeat as vlf
from common import util
import numpy as np

# Loading Files
file_mfccs = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_dem_40_aug.npy' #'../data/mfccs/mfccs_dem_no_deltas_13_vad'
file_mfccs_bea = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_bea_dem_40_aug.npy'
list_mfccs = np.load(file_mfccs, allow_pickle=True)
array_mfccs_bea = np.load(file_mfccs_bea)

# convert list of mfccs to array
# array_mfccs = np.vstack(list_mfccs)
array_mfccs_bea = np.vstack(array_mfccs_bea)

# training GMM
num_clusters = [2, 4, 8, 16, 32, 64, 128]
for g in num_clusters:
    means, covs, priors, LL, posteriors = vlf.gmm.gmm(array_mfccs_bea, g)
    # fisher encoding
    list_fisher = []
    for i in list_mfccs:
        enc = vlf.fisher.fisher(i.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
        list_fisher.append(enc)
    fishers = np.vstack(list_fisher)
    # Saving fishers
    obs = 'fb-40_aug--'  # 'novad'
    file_fishers = 'C:/Users/Win10/PycharmProjects/the_speech/data/fisher_vecs/fisher-{}-{}'.format(g, obs)
    np.savetxt(file_fishers, fishers)
    print("Fishers saved to:", file_fishers)
