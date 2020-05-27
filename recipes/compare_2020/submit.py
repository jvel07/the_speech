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

task = 'mask'
feat_type = ['fisher', 'mfcc']  # provide the types of features and frame-level features to use e.g.: 'fisher', 'mfcc', 'xvecs'
deli = 0
# Loading data: 'fisher' or 'xvecs'
gaussians = [2]
for gauss in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_full(
                                                                             # gauss='512dim',
                                                                             gauss='{}g'.format(gauss),
                                                                             task=task,
                                                                             feat_type=feat_type, n_feats=13,
                                                                             n_deltas=deli, list_labels=['mask','clear'])
    # x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_compare()

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))


    # Scale data
    std_scaler = preprocessing.StandardScaler()
    # x_train = std_scaler.fit_transform(x_train)
    # x_dev = std_scaler.transform(x_dev)

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

    team_name = 'TeamFOSAI'
    submission_index = 3
    label_file = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/labels/labels.csv'
    df_labels = pd.read_csv(label_file)
    # Write out predictions to csv file (official submission format)
    pred_file_name = task + '.' + 'x' +'.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': lencoder.inverse_transform(fused_probs).flatten()},
                      columns=['file_name','prediction'])
    df.to_csv(pred_file_name, index=False)

    print('Done.\n')

