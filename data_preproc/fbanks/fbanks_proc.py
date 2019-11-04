import numpy as np
import matplotlib.pyplot as plt
import python_speech_features
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from common import data_proc_tools as tools

from common import util


def group_and_join(list_fbanks_dem):
    fishers_grouped = util.group_wavs_speakers(list_fbanks_dem, 3)
    fishers_joint = util.join_speakers_wavs(fishers_grouped)
    return fishers_joint


def scale_min_max(array_bea_1, list_fbanks_1):
    # Min-max the data
    scaler = MinMaxScaler().fit(array_bea_1)
    array_bea_scaled = scaler.transform(array_bea_1)
    list_fb_scaled = []
    for i in list_fbanks_1:
        fb_scaled = scaler.transform(i)
        list_fb_scaled.append(fb_scaled)

    return array_bea_scaled, list_fb_scaled


def sel_pca_comp_graph(data_rescaled):
    pca = PCA().fit(data_rescaled)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')

    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=14)

    plt.title('Dataset Explained Variance')
    plt.grid(axis='x')
    plt.show()


def run_pca(array_bea_scaled, list_fb_scaled, n_comp):
    # Fitting
    pca = PCA(n_components=n_comp)
    pca.fit(array_bea_scaled)
    # Reducing
    reduced_bea = pca.transform(bea_scaled)
    reduced_list_fbanks = []
    for item in list_fb_scaled:
        reduced = pca.transform(item)
        reduced_list_fbanks.append(reduced)
    return reduced_list_fbanks, reduced_bea


def do_lda(array_bea, list_fbanks, labels, n_comp):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda.fit(array_bea, labels)
    bea_reduced = lda.transform(array_bea)
    list_fbanks_reduced = []
    for i in list_fbanks:
        _ = lda.transform(i)
        list_fbanks_reduced.append(_)

    return bea_reduced, list_fbanks_reduced


def get_var_ratio(X, y):
    lda = LinearDiscriminantAnalysis(n_components=None)
    lda.fit(X, y)
    return lda.explained_variance_ratio_


def best_components(bea_scaled):
    var_ratio = tools.get_var_ratio_pca(bea_scaled)
    comp = tools.sel_pca_comp(var_ratio, 0.95)
    return comp


def compute_deltas(list_fbanks, number_deltas):
    list_deltas = []
    for i in list_fbanks:
        deltas = python_speech_features.base.delta(feat=i, N=number_deltas)  # Fbanks with n deltas
        list_deltas.append(deltas)
    return list_deltas


def concatenate_list_of_deltas(original, deltas):
    list_conc = []
    for item1, delta in zip(original, deltas):
            conc = np.concatenate((item1, delta))
            list_conc.append(np.float32(conc))
    return list_conc


def concatenate_list_of_deltas2(original, deltas1, deltas2):
    list_conc = []
    for item1, delta1, delta2 in zip(original, deltas1, deltas2):
            conc = np.concatenate((item1, delta1, delta2))
            list_conc.append(np.float32(conc))
    return list_conc


if __name__ == '__main__':

    # Loading fbanks
    file_fbanks = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_dem_40'
    file_fbanks_bea = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_ubm_dem_40'

    list_fbanks = util.read_pickle(file_fbanks)
    fbanks_bea = np.vstack(util.read_pickle(file_fbanks_bea))

    # Output files
    file_pca_fbanks = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_dem_40_PCA'
    file_pca_bea = 'C:/Users/Win10/PycharmProjects/the_speech/data/fbanks/fbanks_ubm_dem_40_PCA_SP'

    # Scaling and selecting best number of components
    bea_scaled, list_dem_scaled = scale_min_max(fbanks_bea, list_fbanks)
    c = best_components(bea_scaled)
    # Reducing dimensions PCA
    fbanks_reduced, bea_reduced = run_pca(bea_scaled, list_dem_scaled, c)

    # Computing deltas
   # list_fbanks_deltas = compute_deltas(list_fbanks, 1)
    #bea_deltas = python_speech_features.base.delta(feat=fbanks_bea, N=1)
    # 2nd deltas
    #list_fbanks_deltas2 = compute_deltas(list_fbanks, 2)
   # bea_deltas2 = python_speech_features.base.delta(feat=fbanks_bea, N=2)
    # Concatenating deltas
    #fbanks_deltas_conc = concatenate_list_of_deltas(list_fbanks, list_fbanks_deltas)
    #bea_deltas_conc = np.concatenate((fbanks_bea, bea_deltas))
    print('deltas concatenated!')

    # Saving data
   # util.save_pickle(file_pca_fbanks, fbanks_reduced)
    util.save_pickle(file_pca_bea, bea_reduced)
