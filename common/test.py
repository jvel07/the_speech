from itertools import repeat

import numpy as np
from common import util
import pandas as pd
import os

# y_labels of dem speaker to pandas Dataframe 
def augment_alz_labels():
    lista = np.loadtxt('../classifiers/cross_val/labels-75.txt', delimiter=',', dtype='str').tolist()
    contador = 0
    new = []
    var = [x for item in lista for x in repeat(item, 3)]
    np.savetxt('/opt/project/data/ids_labels_225.txt', var, delimiter=',', fmt='%s')
    print(len(var))


def create_scp_kaldi():
    new_dir = '/home/jose/mydata/kaldi/egs/dementia/wav-bea-diktafon/'  # dir in the kaldi machine
    work_dir = '/opt/project'  # dir in docker
    dir_wav_ubm = work_dir + '/audio/wav-bea-diktafon/'  # dir in the local machine to the audios

    list_audios = util.read_files_from_dir(dir_wav_ubm)  #util.just_original_75()
    new_list = []
    for item2 in list_audios:
        new_list.append(item2+' '+new_dir+item2)
    return new_list


def create_utt2spk_kaldi():
    list_audios = util.just_original_75()
    df = pd.read_csv('/opt/project/data/ids_labels_225.txt', header=None, dtype=str)
    df.columns = ['patient_id', 'diagnosis']
    ids = df.patient_id.values
    new_list = []
    for i, j in zip(list_audios, ids):
        #ii = os.path.splitext(i)[0]
        new_list.append(i+' '+j)
    return new_list

a=create_utt2spk_kaldi()