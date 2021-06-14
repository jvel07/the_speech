"""
Created by José Vicente Egas López
on 2021. 04. 02. 11 43

"""
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import recall_score, make_scorer
from sklearn.preprocessing import MinMaxScaler

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

######## scoring #####
def uar_scoring(y_true, y_pred, **kwargs):
    uar = recall_score(y_true, y_pred, labels=[1, 0], average='macro')
    return uar

my_scorer = make_scorer(uar_scoring, greater_is_better=True)


##### scoring #####
def average_post(one, two):
    p1 = np.loadtxt(one)
    p2 = np.loadtxt(two)
    probs = np.mean((p1, p2), axis=0)
    a = np.argmax(probs, axis=1)
    return a

def average_post_3(one, two, three):
    p1 = np.loadtxt(one)
    p2 = np.loadtxt(two)
    p3 = np.loadtxt(three)
    probs = np.mean((p1, p2, p3), axis=0)
    a = np.argmax(probs, axis=1)
    return a

task = 'CovidSpeech'
feat_type = ['xvecs', 'spectrogram', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
gaussians = [512]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
# list_c = [0.0001]

preds_dev = 0

srand_list = ['389743']

dev_preds_dic = {}
obs = 'VAD_SPK'
net = 'coldDNN'

x_train, x_dev, x_test, y_train, y_dev, file_n, enc = rutils.load_data_compare2021(
                                        # gauss='512dim-train_dev-{0}_{1}'.format(srand_list[0], obs),
                                        gauss='512dim-{1}_{0}'.format(obs, net),
                                        # gauss='{}g'.format(ga),
                                        task=task, feat_type=feat_type,
                                        n_feats="", list_labels=['positive', 'negative'])

x_combined = np.concatenate((x_train, x_dev))
y_combined = np.concatenate((y_train, y_dev))

std_flag = False
if std_flag:
    std_scaler = preprocessing.StandardScaler()
    x_train = std_scaler.fit_transform(x_train)
    x_dev = std_scaler.transform(x_dev)

    x_combined = std_scaler.fit_transform(x_combined)
    x_test = std_scaler.transform(x_test)

c = 1e-5
kernel = 'linear'

svc = svm.LinearSVC(C=1e-5, verbose=0, max_iter=100000, class_weight='balanced')
svc.fit(x_combined, y_combined.ravel())
y_prob = svc._predict_proba_lr(x_test)
# y_pred = np.argmax(y_prob, axis=1)
np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_test_{}_128fisher13mfcc_{}.txt'.format(c, kernel), y_prob)

self_probs = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_test_{}_128fisher13mfcc_{}.txt'.format(c, kernel)
external_probs = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/baseline/probs_mask_test_0.01_resnet'

fused_probs = average_post(self_probs, external_probs)

team_name = 'TeamGaborGosztolya'
submission_index = 3
label_file = '/media/jose/hk-data/PycharmProjects/the_speech/data/CovidSpeech/labels/test_orig.csv'
df_labels = pd.read_csv(label_file)
# Write out predictions to csv file (official submission format)
pred_file_name = task + '.' + 'x' +'.test.' + team_name + '_' + str(submission_index) + '.csv'
print('Writing file ' + pred_file_name + '\n')
df = pd.DataFrame(data={'filename': df_labels['filename'][df_labels['filename'].str.startswith('test')].values,
                        'prediction': lencoder.inverse_transform(fused_probs).flatten()},
                  columns=['filename', 'prediction'])
df.to_csv(pred_file_name, index=False)

print('Done.\n')

