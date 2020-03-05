import os
import mmap
import re
import numpy as np
import h5py

from kaldi.gmm import FullGmm  # pykaldi
from kaldi.gmm import DiagGmm  # pykaldi
from kaldi.util import io   #  pykaldi


def nnet3read(dnnFilename, outFilename="", write_to_disk=False):
    """ This is a simple, yet fast, routine that reads in Kaldi NNet3 Weight and Bias
        parameters, and converts them into lists of 64-bit floating point numpy arrays
        and optionally dumps the parameters to disk in HDF5 format.

        :param dnnFilename: input DNN file name (it is assumed to be in text format)
        :param outFilename: output hdf5 filename [optional]
        :param write_to_disk: whether the parameters should be dumped to disk [optional]

        :type dnnFilename: string
        :type outFilename: string
        :type write_to_diks: bool

        :return: returns the NN weight and bias parameters (optionally dumps to disk)
        :rtype: tuple (b,W) of list of 64-bit floating point numpy arrays

    """
    # nn_elements = ['LinearParams', 'BiasParams']
    with open('/home/egasj/PycharmProjects/the_speech/data/pcgita/UBMs/16/ubm/final.mdl', 'r') as f:
        pattern = re.compile(rb'<(\bWEIGHTS\b|\bMEANS_INVCOVARS\b|\bINV_COVARS\b)>\s+\[\s+([-?\d\.\de?\s]+)\]')
        with mmap.mmap(f.fileno(), 0,
                       access=mmap.ACCESS_READ) as m:
            b = []
            W = []
            inv_covs = []
            ix = 0
            for arr in pattern.findall(m):
                if arr[0] == b'MEANS_INVCOVARS':
                    b.append(arr[1].split())
                    print("layer{}: [{}x{}]".format(ix, len(b[ix]), len(W[ix]) // len(b[ix])))
                    ix += 1
                elif arr[0] == b'WEIGHTS':
                    W.append(arr[1].split())
                elif arr[0] == b'INV_COVARS':
                    inv_covs.append(arr[1].split())
                else:
                    raise ValueError('Element not in the list.')

    # converting list of strings into lists of 64-bit floating point numpy arrays and reshaping
    weights = [np.array(W[ix], dtype=np.float).reshape(-1, 1) for ix in range(len(W))]
    means_incovars = [np.array(b[ix], dtype=np.float).reshape(len(W[ix]), len(b[ix]) // len(W[ix])) for ix in range(len(b))]
    inv_covars = [np.array(inv_covs[ix], dtype=np.float).reshape(len(W[ix]), len(inv_covs[ix]) // len(W[ix])) for ix in range(len(inv_covs))]



    if write_to_disk:
        # writing the DNN parameters to an HDF5 file
        if not outFilename:
            raise ValueError('Please, enter the output path.')
        filepath = os.path.dirname(outFilename)
        if filepath and not os.path.exists(filepath):
            os.makedirs(filepath)
        with h5py.File(outFilename, 'w') as h5f:
            for ix in range(len(weights)):
                h5f.create_dataset('means' + str(ix), data=means[ix],
                                   dtype='f8', compression='gzip', compression_opts=9)
                h5f.create_dataset('weights' + str(ix), data=weights[ix],
                                   dtype='f8', compression='gzip', compression_opts=9)
    return weights, means_incovars, inv_covars


def get_diag_gmm_params(file_diag, out_dir):
    diag_mdl = io.xopen(file_diag)  # reading .mdl or .ubm file
    gmm = DiagGmm()  # creating DiagGmm object
    gmm.read(diag_mdl.stream(), diag_mdl.binary)  # reading model

    vars = np.asanyarray(gmm.get_vars())
    means = np.asanyarray(gmm.get_means())
    weights = np.asanyarray(gmm.weights())  # priors

    np.savetxt(out_dir + 'variances_dubm_{}'.format(gmm.num_gauss()), vars)
    np.savetxt(out_dir + 'means_dubm_{}'.format(gmm.num_gauss()), means)
    np.savetxt(out_dir + 'weights_dubm_{}'.format(gmm.num_gauss()), weights)
    print("Vars, means and weights saved to:", out_dir)

    return vars, means, weights, gmm.num_gauss()
