"""
Created by José Vicente Egas López
on 2021. 03. 02. 15 44

"""
import os

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.utils import shuffle

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix


def average_post(one, two):
    p1 = np.loadtxt(one)
    p2 = np.loadtxt(two)
    probs = np.mean((p1, p2), axis=0)
    a = np.argmax(probs, axis=1)
    return a, probs

def average_post_3(one, two, three):
    p1 = np.loadtxt(one)
    p2 = np.loadtxt(two)
    p3 = np.loadtxt(three)
    probs = np.mean((p1, p2, p3), axis=0)
    a = np.argmax(probs, axis=1)
    return a, probs


task = 'CovidSpeech'
feat_type = ['fisher', '23mfcc', 2]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256]
gaussians = [2]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
# list_c = [0.0001]

preds_dev = 0

# srand_list = ['389743', '564896', '2656842', '2959019', '4336987', '7234786', '9612365', '423877642', '987236753',
#               '764352323']
srand_list = ['389743']

dev_preds_dic = {}
obs = 'VAD_SPK'
net = 'coldDNN'

for ga in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, file_n, enc = rutils.load_data_compare2021(
                                            # gauss='512dim-train_dev-{0}_{1}'.format(srand_list[0], obs),
                                            # gauss='512dim-{1}_{0}'.format(obs, net),
                                            gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type,
                                            n_feats="", list_labels=['positive', 'negative'])

    # x_train, x_dev, x_test, y_train, y_dev, le = rutils.load_feats_compare_2021('../../data/CovidSpeech/features/'
    #                                                                                     'opensmile/ComParE_2016', ['positive',
    #                                                                                                                'negative'])

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    std_flag = True
    if std_flag:
        std_scaler = preprocessing.StandardScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)

        x_combined = std_scaler.fit_transform(x_combined)
        x_test = std_scaler.transform(x_test)

    scores = []

    for c in list_c:
        posteriors, clf = svm_fits.train_linearsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, class_weight='balanced')
        # posteriors = svm_fits.train_rbfsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, gamma=0.55555555555555)

        # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_k{}_{}_fisher_{}.txt'.format(folds, c, kernel), posteriors)
        y_pred = np.argmax(posteriors, axis=1)
        uar = recall_score(y_dev, y_pred, average='macro')
        scores.append(uar)
        print("with", c, "-", uar)

    best_c = list_c[np.argmax(scores)]
    best_uar = np.max(scores)
    # util.results_to_csv(file_name='exp_results/results_{}_{}.csv'.format(task, feat_type[0]),
    #                     list_columns=['Exp. Details', 'C', 'UAR', 'STD', 'NET', 'SET'],
    #                     list_values=[os.path.basename(file_n), best_c, best_uar, std_flag, net, 'DEV'])

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = list_c[np.argmax(scores)]
    print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity), best_uar)

    posteriors_test, clf = svm_fits.train_linearsvm_cpu(x_combined, y_combined.ravel(), c=optimum_complexity, X_eval=x_test,
                                                        class_weight='balanced')
    np.savetxt('posteriors/post_{}.txt'.format(feat_type[1]), posteriors_test)


    opem_smile_post = 'posteriors/post_opensmile.txt'
    spectrogram_post = 'posteriors/post_spectrogram.txt'
    fbanks_post = 'posteriors/post_40fbanks.txt'

    preds_fused, fused_probs = average_post_3(opem_smile_post, spectrogram_post, fbanks_post)
    # preds_OSpec, fused_probs_OSpec = average_post(opem_smile_post, spectrogram_post)
    # preds_OFBan, fused_probs_OFBan = average_post(opem_smile_post, fbanks_post)


    # team_name = 'TeamGaborGosztolya'
    # submission_index = 4
    # label_file = '/media/jose/hk-data/PycharmProjects/the_speech/data/CovidSpeech/labels/test_orig.csv'
    # df_labels = pd.read_csv(label_file)
    # # Write out predictions to csv file (official submission format)
    # pred_file_name = task + '.' + 'x2' +'.test.' + team_name + '_' + str(submission_index) + '.csv'
    # print('Writing file ' + pred_file_name + '\n')
    # df = pd.DataFrame(data={'filename': df_labels['filename'][df_labels['filename'].str.startswith('test')].values,
    #                         'prediction': enc.inverse_transform(preds_fused).flatten()},
    #                   columns=['filename', 'prediction'])
    # df.to_csv(pred_file_name, index=False)
    #
    # print('Done.\n')



