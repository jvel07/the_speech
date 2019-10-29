import cyvlfeat as vlf
from common import util
import numpy as np

# Loading Files
file_mfccs = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_dem_40_2deltas' #'../data/mfccs/mfccs_dem_no_deltas_13_vad'
file_mfccs_bea = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_ubm_dem_40_2deltas'
list_mfccs = util.read_pickle(file_mfccs)
array_mfccs_bea = util.read_pickle(file_mfccs_bea)

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
    obs = 'fbanks-40_imp_2deltas--'  # 'novad'
    file_fishers = 'C:/Users/Win10/PycharmProjects/the_speech/data/fisher_vecs/fisher-{}-{}'.format(g, obs)
    np.savetxt(file_fishers, fishers)
    print("Fishers saved to:", file_fishers)
