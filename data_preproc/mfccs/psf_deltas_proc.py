# File for computing deltas of python speech features library
import numpy as np
import python_speech_features


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
            conc = np.concatenate((item1, delta1, delta2), axis=1)
            list_conc.append(np.float32(conc))
    return list_conc