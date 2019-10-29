import numpy as np
import pickle
import bob.kaldi
from common2 import util

from bob.learn.linear import FisherLDATrainer

ivector_2D_file = '../data/ivecs/ivecs-'+str(16)+'g-cold-128i-train'
ivecs = np.asarray(np.loadtxt(ivector_2D_file), dtype='float64')

ivecs_grouped = util.group_wavs_speakers(ivecs, 3)

# Whitening iVectors
#whitening_trainer = WhiteningTrainer()
#whitener_machine = bob.learn.linear.Machine(256, 256)
#whitening_trainer.train(a_ivectors, whitener_machine)
#a_ivectors = whitener_machine(a_ivectors)


# LDA ivectors
lda_trainer = FisherLDATrainer()
lda_machine = bob.learn.linear.Machine(ivecs[1].shape, 74)
lda_trainer.train(ivecs_grouped, lda_machine)
for item in ivecs_grouped:
    for subitem in item:
        subitem = lda_machine(subitem)

