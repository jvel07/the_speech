from data_preproc.mfccs import extract_mfccs
from data_preproc.ivecs import extract_ivecs
import numpy as np

# Name of the task/recipe/dataset/etc.

recipe = 'wav16k_split_long'

# Working directories
# work_dir = '/opt/project/'  # for titan x machine (docker bob kaldi)
# work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/'  # for titan x machine (normal)
work_dir = '/media/jose/hk-data/PycharmProjects/the_speech/'  # for ubuntu (native bob kaldi)
# in and out dirs
audio_dir = work_dir + 'audio/'
out_dir = work_dir + 'data/'

# Name of the folder containing the UBM (not the path)
ubm_folder_name = 'wav16k_split_long'

# List of number of clusters wanted to use
# list_n_clusters = [256]
list_n_clusters = [4, 8, 16, 32, 64, 128]
# list_n_clusters = [256]
# list_n_clusters = [64]


# Computes mfccs from wavs existing in the directories provided by the user
def do_mfccs_ubm():
    print("=======MFCC extraction phase for UBM========")
    out_dir = work_dir + 'data/UBMs/'
    cepstral_type = "mfcc"  # choose between "mfcc" or "plp"

    # Loading filtered id-wavs
    arr_filtered_wavs_id = np.genfromtxt(
        '/media/jose/hk-data/PycharmProjects/the_speech/recipes/demencia94B/filt_UBMbea_lthan4secs.txt', dtype=str,
        delimiter='\n')
    list_specific_wavs = []
    for i in arr_filtered_wavs_id:  # adding the parent path to the name of the wav
        list_specific_wavs.append('/media/jose/hk-data/PycharmProjects/the_speech/audio/{0}/'.format(ubm_folder_name) + i)
    # list_specific_wavs[1]

    print("\nReading dir:", ubm_folder_name)
    for deltas in [0, 1, 2]:
        print("Extracting with {} deltas".format(deltas))
        extract_mfccs.compute_flevel_feats(list_specific_wavs, out_dir, cepstral_type=cepstral_type, num_feats=20,
                                           recipe=recipe, folder_name=ubm_folder_name, num_deltas=deltas, obs='')




def do_ubms():
    print("=======i-vector extraction phase========")
    ubm_dir = work_dir + 'data/UBMs/{0}/{0}'.format(ubm_folder_name)

    for deltas in [2]:
        feats_info = [20, deltas, 'mfcc']  # info of the mfccs (n_features, deltas)
        # the following is a list in the case of the UBM is meant to be trained with training and dev sets
        list_files_ubm = [ubm_dir + '/mfcc_{0}_{1}_{0}_{2}del.mfcc'.format(ubm_folder_name, feats_info[0], deltas)]
        extract_ivecs.train_ubm_only(list_n_gauss=list_n_clusters, out_dir=out_dir,
                                     list_files_ubm=list_files_ubm, feats_info=feats_info)



def steps(i):
    switcher = {
        0: do_mfccs_ubm,
        1: do_ubms
    }
    func = switcher.get(i)
    return func()


steps(1)

