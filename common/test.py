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




a=create_utt2spk_kaldi()