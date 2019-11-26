#!/usr/bin/env python
#
# Milos Cernak <milos.cernak@idiap.ch>
# March 1, 2017
#

'''Tests for Kaldi bindings'''

import pkg_resources
import numpy as np
import bob.io.audio
import bob.io.base.test_utils
import os

import bob.kaldi


def test_ivector_train():

    temp_dubm_file = bob.io.base.test_utils.temporary_filename()
    temp_fubm_file = bob.io.base.test_utils.temporary_filename()
    temp_ivec_file = bob.io.base.test_utils.temporary_filename()
    
    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

    data = bob.io.audio.reader(sample)
    # MFCC
    array = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False)
    # Train small diagonal GMM
    dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss=2,
                               num_gselect=2, num_iters=2)
    # Train small full GMM
    fubm = bob.kaldi.ubm_full_train(array, dubm, temp_fubm_file,
                               num_gselect=2, num_iters=2)
    # Train small ivector extractor
    feats=[[]]
    feats[0]=array
    ivector = bob.kaldi.ivector_train(feats, fubm, temp_ivec_file,
                               num_gselect=2, ivector_dim=20, num_iters=2)

    assert ivector.find('IvectorExtractor')


def test_ivector_extract():

    temp_dubm_file = bob.io.base.test_utils.temporary_filename()
    temp_fubm_file = bob.io.base.test_utils.temporary_filename()
    temp_ivec_file = bob.io.base.test_utils.temporary_filename()

    sample = pkg_resources.resource_filename(__name__, '../audio/wav_anon_75_225/001A_szurke.wav')
   # reference = pkg_resources.resource_filename(
    #    __name__, 'data/sample16k.ivector')

    data = bob.io.audio.reader(sample)
    # MFCC
    array = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False)
    # Train small diagonal GMM
    dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss=2,
                               num_gselect=2, num_iters=2)
    # Train small full GMM
    fubm = bob.kaldi.ubm_full_train(array, dubm, temp_fubm_file,
                                   num_gselect=2, num_iters=2)
    # Train small ivector extractor
    feats=[[]]
    feats[0]=array
    ivector = bob.kaldi.ivector_train(feats, fubm, temp_ivec_file,
                               num_gselect=2, ivector_dim=20, num_iters=2)
    # Extract ivector
    ivector_array = bob.kaldi.ivector_extract(array, fubm, ivector,
                                              num_gselect=2)

    print(ivector_array)
    #theirs = np.loadtxt(reference)

    #assert np.allclose(ivector_array, theirs)


def test_plda_train():

    plda_file = bob.io.base.test_utils.temporary_filename()
    mean_file = bob.io.base.test_utils.temporary_filename()
    features = pkg_resources.resource_filename(
        __name__, 'data/feats-mobio.npy')

    feats = np.load(features)

    # Train PLDA
    plda = bob.kaldi.plda_train(feats, plda_file, mean_file)

    assert plda[0].find('Plda')
    assert os.path.exists(mean_file)


def test_plda_enroll():

    plda_file = bob.io.base.test_utils.temporary_filename()
    mean_file = bob.io.base.test_utils.temporary_filename()
    features = pkg_resources.resource_filename(
        __name__, 'data/feats-mobio.npy')

    feats = np.load(features)

    # Train PLDA
    plda = bob.kaldi.plda_train(feats, plda_file, mean_file)

    # Enroll; plda[0] - PLDA model, plda[1] - PLDA global mean
    enrolled = bob.kaldi.plda_enroll(feats[0], plda[1])

    assert enrolled.find('spk36')


def test_plda_score():

    plda_file = bob.io.base.test_utils.temporary_filename()
    mean_file = bob.io.base.test_utils.temporary_filename()
    spk_file = bob.io.base.test_utils.temporary_filename()
    test_file = pkg_resources.resource_filename(
        __name__, 'data/test-mobio.ivector')
    features = pkg_resources.resource_filename(
        __name__, 'data/feats-mobio.npy')

    train_feats = np.load(features)
    test_feats = np.loadtxt(test_file)

    # Train PLDA
    plda = bob.kaldi.plda_train(train_feats, plda_file, mean_file)
    # Enroll PLDA (for the first speaker)
    enrolled = bob.kaldi.plda_enroll(train_feats[0], plda[1])
    # Score PLDA
    score = bob.kaldi.plda_score(test_feats, enrolled, plda[0], plda[1])

    assert np.allclose(score, [-23.9922], 1e-03, 1e-05)


test_ivector_extract()