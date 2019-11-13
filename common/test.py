import numpy as np
from itertools import repeat
from common import util

"""
# y_labels of dem speaker to pandas Dataframe 
lista = np.loadtxt('../classifiers/cross_val/labels-75.txt', delimiter=',', dtype='str').tolist()
contador = 0
new = []
var = [x for item in lista for x in repeat(item, 4)]
np.savetxt('ids_labels_300.txt', var, delimiter=',', fmt='%s')
"""

# group speakers per type of audio (normal, noisy, stretched, pitched)
mfccs = util.read_pickle('../data/mfccs/mfccs_dem_13_aug')
len_of_data = len(mfccs)
number_of_rows = 4
number_of_group = 12
result = [list(mfccs[i:i+number_of_group][j::number_of_rows]) for i in range(0, len_of_data, number_of_group) for j in range(number_of_rows)]
