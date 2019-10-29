import bob.db.iris
import bob.learn.em
import bob.learn.linear
import matplotlib.pyplot as plt
import numpy
import numpy as np
from common2 import util
numpy.random.seed(2)  # FIXING A SEED


def train_ubm(features, n_gaussians):
    """
    Train UBM

     **Parameters**
       features: 2D numpy array with the features

       n_gaussians: Number of Gaussians

    """

    input_size = features.shape[1]

    kmeans_machine = bob.learn.em.KMeansMachine(int(n_gaussians), input_size)
    ubm = bob.learn.em.GMMMachine(int(n_gaussians), input_size)

    # The K-means clustering is firstly used to used to estimate the initial
    # means, the final variances and the final weights for each gaussian
    # component
    kmeans_trainer = bob.learn.em.KMeansTrainer('RANDOM_NO_DUPLICATE')
    bob.learn.em.train(kmeans_trainer, kmeans_machine, features)

    # Getting the means, weights and the variances for each cluster. This is a
    # very good estimator for the ML
    (variances, weights) = kmeans_machine.get_variances_and_weights_for_each_cluster(features)
    means = kmeans_machine.means

    # initialize the UBM with the output of kmeans
    ubm.means = means
    ubm.variances = variances
    ubm.weights = weights

    # Creating the ML Trainer. We will adapt only the means
    trainer = bob.learn.em.ML_GMMTrainer(
        update_means=True, update_variances=False, update_weights=False)
    bob.learn.em.train(trainer, ubm, features)

    return ubm


def ivector_train(features, ubm, dim):
    """
     Trains T matrix

     **Parameters**
       features: List of :py:class:`bob.learn.em.GMMStats`

       n_gaussians: UBM (:py:class:`bob.learn.em.GMMMachine`)

     """

    stats = []
    for user in features:
        s = bob.learn.em.GMMStats(ubm.shape[0], ubm.shape[1])
        for f in user:
            ubm.acc_statistics(f, s)
        stats.append(s)

    subspace_dimension_of_t = dim

    ivector_trainer = bob.learn.em.IVectorTrainer(update_sigma=True)
    ivector_machine = bob.learn.em.IVectorMachine(
        ubm, subspace_dimension_of_t, 10e-5)

    # train IVector model
    bob.learn.em.train(ivector_trainer, ivector_machine, stats, 500)

    return ivector_machine


def acc_stats(data, gmm):
    gmm_stats = []
    for d in data:
        s = bob.learn.em.GMMStats(gmm.shape[0], gmm.shape[1])
        gmm.acc_statistics(d, s)
        gmm_stats.append(s)

    return gmm_stats


def compute_ivectors(gmm_stats, ivector_machine):
    """
    Given :py:class:`bob.learn.em.GMMStats` and an T matrix, get the iVectors.
    """

    ivectors = []
    for g in gmm_stats:
        ivectors.append(ivector_machine(g))

    return numpy.array(ivectors)


def f(x):
    return np.float64(x)


# Parameters
obs_ivec = 'bob'
num_gauss = 8
ivector_dim = 100

sets = ['train', 'dev', 'test']
for i in sets:
    if i == 'train':
        set_ = 'train'
        file_ivectors = '../data/ivecs/cold/ivecs-{}-g-cold-{}i-{}-{}'.format(num_gauss, ivector_dim, set_, obs_ivec)
        file_ubm_model = '../data/models/cold/ubm_mdl_{}g_cold_{}.hdf5'.format(num_gauss, obs_ivec)
        file_ivec_model = '../data/models/cold/ivec_mdl_{}g_cold_{}.hdf5'.format(num_gauss, obs_ivec)
        # Load MFCCs
        mfccs_file_ivec = '../data/mfccs/cold/mfccs_cold_{}_20_scl'.format(set_)
        mfccs_file_ubm = '../data/mfccs/cold/mfccs_cold_ubm_20_fulltmt_scl'
        mfccs_wav_ubm = np.asarray(np.vstack(util.read_pickle(mfccs_file_ubm)), dtype='float64')
        list_mfccs_ivecs = util.read_pickle(mfccs_file_ivec)
        f2 = np.vectorize(f)
        list_mfccs_ivecs = [f2(i) for i in list_mfccs_ivecs]
        # TRAINING THE PRIOR
        print("training ubm...")
        ubm = train_ubm(mfccs_wav_ubm, num_gauss)
        del mfccs_wav_ubm
        # Save ubm machine
        util.save_bob_machine(file_ubm_model, ubm)
        # Building i-vec machine (i-vec extractor model)
        print("building i-vec machine...")
        ivector_machine = ivector_train(list_mfccs_ivecs, ubm, ivector_dim)
        # Save i-vec machine
        util.save_bob_machine(file_ivec_model, ivector_machine)
        # Extract i-vecs
        print("extracting i-vecs...")
        ivecs = compute_ivectors(acc_stats(list_mfccs_ivecs, ubm), ivector_machine)
        del ubm, ivector_machine, list_mfccs_ivecs
        np.savetxt(file_ivectors, ivecs)
        del ivecs
        print("i-vecs saved to:", file_ivectors)
    elif i == 'dev':
        set_ = 'dev'
        file_ivectors = '../data/ivecs/cold/ivecs-{}-g-cold-{}i-{}-{}'.format(num_gauss, ivector_dim, set_, obs_ivec)
        file_ubm_model = '../data/models/cold/ubm_mdl_{}g_cold_{}.hdf5'.format(num_gauss, obs_ivec)
        file_ivec_model = '../data/models/cold/ivec_mdl_{}g_cold_{}.hdf5'.format(num_gauss, obs_ivec)
        # Load MFCCs
        mfccs_file_ivec = '../data/mfccs/cold/mfccs_cold_{}_20_scl'.format(set_)
        list_mfccs_ivecs = util.read_pickle(mfccs_file_ivec)
        f2 = np.vectorize(f)
        list_mfccs_ivecs = [f2(i) for i in list_mfccs_ivecs]  # converting arrays to float64 within the list
        # Load UBM trained with training set
        h5file_ubm = util.load_bob_machine(file_ubm_model)
        ubm = bob.learn.em.GMMMachine(h5file_ubm)
        # Load i-vec extractor trained with training set
        h5file_ivec = util.load_bob_machine(file_ivec_model)
        ivec_machine = bob.learn.em.IVectorMachine(h5file_ivec)
        # Extract i-vecs
        print("extracting i-vecs...")
        ivecs = compute_ivectors(acc_stats(list_mfccs_ivecs, ubm), ivec_machine)
        del ubm, ivec_machine, list_mfccs_ivecs
        np.savetxt(file_ivectors, ivecs)
        print("i-vecs saved to:", file_ivectors)
        del ivecs
    else:
        set_ = 'test'
        file_ivectors = '../data/ivecs/cold/ivecs-{}-g-cold-{}i-{}-{}'.format(num_gauss, ivector_dim, set_, obs_ivec)
        file_ubm_model = '../data/models/cold/ubm_mdl_{}g_cold_{}.hdf5'.format(num_gauss, obs_ivec)
        file_ivec_model = '../data/models/cold/ivec_mdl_{}g_cold_{}.hdf5'.format(num_gauss, obs_ivec)
        # Load MFCCs
        mfccs_file_ivec = '../data/mfccs/cold/mfccs_cold_{}_20_scl'.format(set_)
        list_mfccs_ivecs = util.read_pickle(mfccs_file_ivec)
        f2 = np.vectorize(f)
        list_mfccs_ivecs = [f2(i) for i in list_mfccs_ivecs]  # converting arrays to float64 within the list
        # Load UBM trained with training set
        h5file_ubm = util.load_bob_machine(file_ubm_model)
        ubm = bob.learn.em.GMMMachine(h5file_ubm)
        # Load i-vec extractor trained with training set
        h5file_ivec = util.load_bob_machine(file_ivec_model)
        ivec_machine = bob.learn.em.IVectorMachine(h5file_ivec)
        # Extract i-vecs
        print("extracting i-vecs...")
        ivecs = compute_ivectors(acc_stats(list_mfccs_ivecs, ubm), ivec_machine)
        del ubm, ivec_machine, list_mfccs_ivecs
        np.savetxt(file_ivectors, ivecs)
        del ivecs
        print("i-vecs saved to:", file_ivectors)



# Whitening iVectors
#whitening_trainer = bob.learn.linear.WhiteningTrainer()
#whitener_machine = bob.learn.linear.Machine(
 #   ivecs.shape[1], ivecs.shape[1])
#whitening_trainer.train(numpy.vstack(list_mfccs_ivecs), whitener_machine)
#ivecs = whitener_machine(ivecs)


# LDA ivectors
#print('lda')
#lda_trainer = bob.learn.linear.FisherLDATrainer()
#lda_machine = bob.learn.linear.Machine(
 #   ivecs.shape[1], ivecs.shape[1])
#lda_trainer.train(list_mfccs_ivecs, lda_machine)
#ivecs = lda_machine(ivecs)


# WCCN ivectors
# wccn_trainer = bob.learn.linear.WCCNTrainer()
# wccn_machine = bob.learn.linear.Machine(
#     ivector_setosa.shape[1], ivector_setosa.shape[1])
# wccn_trainer.train([ivector_setosa, ivector_versicolor,
#                     ivector_virginica], wccn_machine)
# ivector_setosa = wccn_machine(ivector_setosa)
# ivector_versicolor = wccn_machine(ivector_versicolor)
# ivector_virginica = wccn_machine(ivector_virginica)
