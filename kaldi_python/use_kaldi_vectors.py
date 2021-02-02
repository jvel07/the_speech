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
        return feat_list#, list(utt2scp.keys())

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
       feat = kaldi_io.read_mat(self.feat_list[idx])
       # feat = kaldi_io.read_vec_flt(self.feat_list[idx])
       return feat

    def __getutt__(self, idx):
        utt = self.utt_list[idx]
        return utt



def get_frame_level_to_txt(list_features, batch_number):
    for c, feat in enumerate(list_features, 1):
        out_file_name = 'sleepiness/train/vad_train.{}_{}.vad'.format(batch_number, c)
        np.savetxt(out_file_name, feat)
        print('Saved', out_file_name)


def get_frame_level(list_sets, n_batches):
    feats_info = ['sleepiness', 'mfcc', 23, 0]  # [task, feat_type, n_feats, deltas]
    for _set in list_sets:
        feats = []
        for k in range(1, n_batches+1):
            in_file = '/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/mfcc/raw_mfcc_{}.{}.scp'.format(_set, k)
            print(in_file)
            dataset = SPKID_Dataset(in_file)
            for j in range(len(dataset)):
                feats.append(dataset.__getitem__(j))
        out_file_name = '/media/jose/hk-data/PycharmProjects/the_speech/data/{0}/{1}/{2}_{0}_{3}_{1}_{4}del.{2}'.format(feats_info[0], _set, feats_info[1], feats_info[2], feats_info[3])
        np.save(out_file_name, feats, allow_pickle=True)
        print('Saved', out_file_name)

# get_frame_level(['train', 'dev', 'test'], 8)


def get_xvecs(list_sets, dest_task):
    obs = '7234786_fbanks40'
    for i in list_sets:
        # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/exp_20mfcc/xvectors_demencia_94abc_bea16k_special/xvector.scp')
        dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/exp_23mf_train_only_srand_{1}/{0}/xvector.scp'.format(i, obs))
        xvecs = []
        for j in range(len(dataset)):
            xvecs.append(dataset.__getitem__(j))
        x = np.vstack(xvecs)
        np.savetxt('../data/{0}/{1}/xvecs-23mfcc-0del-{2}dim-train_dev-{4}-{3}.xvecs'.format(dest_task, i, x.shape[1], i, obs), x)
        # np.savetxt('../data/{0}/{1}/xvecs/xvecs-23mfcc-0del-{2}dim-pretrained-{3}.xvecs'.format(dest_task, dest_task, x.shape[1], dest_task), x)
        print(x.shape)

# get_xvecs(['train', 'dev', 'test'], 'sleepiness')

def get_ivecs():
    num = [1, 2, 3, 4]
    xvecs = []
    # for number in num:
    dataset = SPKID_Dataset('../kaldi_python/exp/test1/ivector.scp')
    for i in range(len(dataset)):
        xvecs.append(dataset.__getitem__(i))
    x = np.vstack(xvecs)
    np.savetxt('../data/ivecs/ivecs-32-23mf---300_test', x)
    print(x.shape)

