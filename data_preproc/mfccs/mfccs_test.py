import pkg_resources
import numpy as np
import bob.io.audio
import bob.io.base.test_utils
import os
from common2 import util
import pickle

import bob.kaldi

# MFCC
data = bob.io.audio.reader('../audio/wav-demencia-all/001A_feher.wav')
array_mfcc = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=True)

print("MFCC features shape:", array_mfcc.shape)

# Train small diagonal GMM
dubm = bob.kaldi.ubm_train(array_mfcc, 'temp_dubm_file4', num_gauss=2,
                           num_gselect=2, num_iters=2)
# Train small full GMM
fubm = bob.kaldi.ubm_full_train(array_mfcc, dubm, 'temp_fubm_file4',
                                num_gselect=2, num_iters=2)
# Train small ivector extractor
feats = [[]]
feats[0] = array_mfcc
ivector = bob.kaldi.ivector_train(feats, fubm, 'temp_ivec_file4',
                                  num_gselect=2, ivector_dim=20, num_iters=2)
# Extract ivector
ivector_array2 = bob.kaldi.ivector_extract(array_mfcc, fubm, ivector,
                                        num_gselect=2)

