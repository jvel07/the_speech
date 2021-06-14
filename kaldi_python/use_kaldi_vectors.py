# Reading ark and scp files that were extracted using Kaldi.
import kaldi_io
import numpy as np
import pandas


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
        self.feat_list, self.utt_list = self.read_scp(vector_scp)

    def read_scp(self, vector_scp):
        utt2scp = read_as_dict(vector_scp)
        feat_list = []
        for utt in utt2scp:
            feat_list.append(utt2scp[utt])
        return feat_list, list(utt2scp.keys())

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
       # feat = kaldi_io.read_mat(self.feat_list[idx])  # reading MFCCs/fbanks...
       feat = kaldi_io.read_vec_flt(self.feat_list[idx])   # reading xvecs
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
    feats_info = ['depression', 'fbank', 23, 0]  # [task, feat_type, n_feats, deltas]
    for _set in list_sets:
        feats = []
        for k in range(1, n_batches+1):
            in_file = '/home/jose/Documents/kaldi/egs/depression/data/mfcc/raw_{0}_{1}.{2}.scp'.format(feats_info[1],
                                                                                                       _set, k)
            print(in_file)
            dataset = SPKID_Dataset(in_file)
            for j in range(len(dataset)):
                feats.append(dataset.__getitem__(j))
        out_file_name = '/media/jose/hk-data/PycharmProjects/the_speech/data' \
                        '/{0}/{1}/{2}_{0}_{3}_{1}_{4}del.{2}'.format(feats_info[0], _set,
                                                                     feats_info[1], feats_info[2], feats_info[3])
        np.save(out_file_name, feats, allow_pickle=True)
        print('Saved', out_file_name)

# get_frame_level(['train'], 1)


def get_xvecs(list_sets, dest_task):
    obs = 'VAD'
    # obs = ''
    feat = 'spectrogram'
    net = 'coughvidDNN'
    for i in list_sets:
        # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/exp_20mfcc/xvectors_demencia_94abc_bea16k_special/xvector.scp')
        # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/ORIG_exp_{2}_DNN_{3}_'
        #                         '{1}/{0}/xvector.scp'.format(i, obs, feat, net))
        # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/exp_transfer_l/x_vectors'
        #                         '/{0}/xvector.scp'.format(i, obs, feat, net))
        dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/{0}_{3}_{2}_xvecs_{4}/'
                                '{1}/xvector.scp'.format(feat, i, net, dest_task, obs))
        xvecs = []
        for j in range(len(dataset)):
            xvecs.append(dataset.__getitem__(j))
        x = np.vstack(xvecs)
        file_name = '../data/{0}/{1}/xvecs-{5}-0del-{2}dim-{6}_{4}-{3}.xvecs'.format(dest_task, i, x.shape[1], i,
                                                                                         obs, feat, net)
        np.savetxt(file_name, x)
        # np.savetxt('../data/{0}/{1}/xvecs/xvecs-23mfcc-0del-{2}dim-pretrained-{3}.xvecs'.format(dest_task, dest_task, x.shape[1], dest_task), x)
        print(x.shape)
        print(file_name)


get_xvecs(['train', 'dev', 'test'], 'CovidCough')


def get_xvecs_2(list_sets, dest_task):
    # srand_list = ['389743', '564896', '2656842', '2959019', '4336987', '7234786', '9612365', '423877642', '987236753',
    #               '764352323']
    srand_list = ['389743']

    obs = 'vad'
    feat = 'spectro'
    for srand in srand_list:
        for i in list_sets:
            # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/exp_20mfcc/xvectors_demencia_94abc_bea16k_special/xvector.scp')
            # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python'
            #                         '/exp_{0}_train_only_srand_{1}_{2}/{2}/xvector.scp'.format(feat,srand, obs, i))
            dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python'
                                    '/ORIG_exp_{0}_DNN_sre16_pretrained_{1}/{2}/xvector.scp'.format(feat, obs, i))
            xvecs = []
            for j in range(len(dataset)):
                xvecs.append(dataset.__getitem__(j))
            x = np.vstack(xvecs)
            # file_name = '../data/{0}/{1}/xvecs-{6}-0del-{2}dim-train_dev-{5}_{4}-{3}.xvecs'.format(dest_task, i,
            #                                                                                           x.shape[1], i,
            #                                                                                           obs+'-voxcel', srand, feat)
            file_name = '../data/{0}/{1}/xvecs-{5}-0del-{2}dim-sre16_{4}-{3}.xvecs'.format(dest_task, i,
                                                                                                      x.shape[1], i,
                                                                                                      obs, feat)
            np.savetxt(file_name, x)
            # np.savetxt('../data/{0}/{1}/xvecs/xvecs-23mfcc-0del-{2}dim-pretrained-{3}.xvecs'.format(dest_task, dest_task, x.shape[1], dest_task), x)
            print(x.shape)
            print(file_name)
            print()


# get_xvecs_2(['train', 'dev', 'test'], 'depression')

def get_xvecs_as_dataframe(list_sets, dest_task):
    srand = 389743
    obs = 'noAug'
    feat = '40fbanks'
    xvecs = []
    filenames = []
    for i in list_sets:
        # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python/exp_20mfcc/xvectors_demencia_94abc_bea16k_special/xvector.scp')
        # dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python'
        #                         '/exp_{0}_train_only_srand_{1}_{2}/{2}/xvector.scp'.format(feat,srand, obs, i))
        dataset = SPKID_Dataset('/media/jose/hk-data/PycharmProjects/the_speech/kaldi_python'
                                '/exp_{0}_DNN_train_dev_srand_{1}_{2}/{3}/xvector.scp'.format(feat, srand, obs, i))
        for j in range(len(dataset)):
            xvecs.append(dataset.__getitem__(j))
            filenames.append(str(dataset.__getutt__(j)))
        df = pandas.DataFrame(data=xvecs, columns=np.arange(len(xvecs[0])))
        df['filename'] = filenames
        file_name = '../data/{0}/{1}/xvecs-{6}-0del-{2}dim-train_dev-{5}_{4}-{3}.csv'.format(dest_task, i,
                                                                                               df.shape[1]-1, i,
                                                                                               obs, srand, feat)
        df.to_csv(file_name)

# get_xvecs_as_dataframe(['train', 'dev', 'test'], 'primates')


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

