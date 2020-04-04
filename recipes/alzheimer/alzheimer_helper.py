import os

import pandas as pd
import numpy as np


# loading only specific audios. Takes source file with 'id,wavs' specified, takes a list of the original audios.
def load_specific(source_file, list_original_audios):
    array = np.squeeze(np.vstack(list_original_audios))
    for idx, ele in enumerate(array):
        array[idx] = os.path.basename(ele)

    df = pd.read_csv(source_file, dtype=str)
    list_wavs = df.id.values.tolist()
    list_selected_audios = []
    for ele in array:
        for ele2 in list_wavs:
            if ele2 == ele[0:3]:
                list_selected_audios.append(os.path.dirname(list_original_audios[0])+ '/' +
                    ele)


    return list_selected_audios
