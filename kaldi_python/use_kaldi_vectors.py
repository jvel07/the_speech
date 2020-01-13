# Reading ark and scp files that were extracted using Kaldi.
import kaldi_io
import numpy as np


def read_as_dict(file):
    utt2feat = {}
    with open(file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt2feat[line.split()[0]] = line.split()[1]
    return utt2feat


class Dataset(object):
    pass


class SPKID_Dataset(Dataset):
    # Loads i-vectors/x-vectors for the data
    def __init__(self, vector_scp):
        self.feat_list = self.read_scp(vector_scp)

    def read_scp(self, vector_scp):
        utt2scp = read_as_dict(vector_scp)
        feat_list = []
        for utt in utt2scp:
            feat_list.append(utt2scp[utt])
        return feat_list

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
 #       feat = kaldi_io.read_mat(self.feat_list[idx])
        feat = kaldi_io.read_vec_flt(self.feat_list[idx])
        return feat


def get_xvecs():
    dataset = SPKID_Dataset('../kaldi_python/exp/xvectors_test1/xvector.scp')
    xvecs = []
    for i in range(len(dataset)):
        xvecs.append(dataset.__getitem__(i))
    x = np.vstack(xvecs)
    np.savetxt('../data/xvecs/xvecs--23mf---512_ctest2', x)
    print(x.shape)


def get_ivecs():
    num = [1, 2, 3, 4]
    xvecs = []
    for number in num:
        dataset = SPKID_Dataset('../kaldi_python/exp/ivectors_train/ivector.{}.scp'.format(number))
        for i in range(len(dataset)):
            xvecs.append(dataset.__getitem__(i))
    x = np.vstack(xvecs)
    np.savetxt('../data/ivecs/alzheimer/ivecs-8-20mf---256i', x)
    print(x.shape)

get_xvecs()
