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

from common import util
from common.util import plot_confusion_matrix_2
from recipes.sleepiness.sleepiness_helper import load_data_full
from recipes.sleepiness import sleepiness_helper as sh

task = 'CovidSpeech'
feat_type = ['xvecs', 'spectrogram', 0]  # provide the types of features, type of frame-level feats, and deltas to use e.g.: 'fisher', 'mfcc', 0

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
# gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
gaussians = [512]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
# list_c = [0.0001]

preds_dev = 0

# srand_list = ['389743', '564896', '2656842', '2959019', '4336987', '7234786', '9612365', '423877642', '987236753',
#               '764352323']
srand_list = ['389743']

dev_preds_dic = {}
obs = 'VAD_SPK'
net = 'coldDNN_AUG'

for ga in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, file_n, enc = rutils.load_data_compare2021(
                                            # gauss='512dim-train_dev-{0}_{1}'.format(srand_list[0], obs),
                                            gauss='512dim-{1}_{0}'.format(obs, net),
                                            # gauss='{}g'.format(ga),
                                            task=task, feat_type=feat_type,
                                            n_feats="", list_labels=['positive', 'negative'])

    # x_combined = np.concatenate((x_train, x_dev))
    # y_combined = np.concatenate((y_train, y_dev))

    std_flag = False
    if std_flag:
        std_scaler = preprocessing.StandardScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)

        # x_combined = std_scaler.fit_transform(x_combined)
        # x_test = std_scaler.transform(x_test)

    scores = []
    print(file_n)
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
    util.results_to_csv(file_name='exp_results/results_{}_{}.csv'.format(task, feat_type[0]),
                        list_columns=['Exp. Details', 'C', 'UAR', 'STD', 'NET', 'SET'],
                        list_values=[os.path.basename(file_n), best_c, best_uar, std_flag, net, 'DEV'])

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = list_c[np.argmax(scores)]
    print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity), best_uar)

    # Saving best dev posteriors
    # dev_preds = svm_fits.train_svr_gpu(x_train, y_train.ravel(), X_eval=x_dev, c=optimum_complexity, nu=0.5)
    # np.savetxt('preds_{}/best_preds_dev_{}_srand_{}.txt'.format(feat, optimum_complexity, srand), dev_preds)
    #
    # y_pred = svm_fits.train_svr_gpu(x_combined, y_combined.ravel(), X_eval=x_test, c=optimum_complexity, nu=list_nu[0])
    # # y_pred = sh.linear_trans_preds_test(y_train=y_train, preds_dev=preds_orig, preds_test=y_pred)
    # coef_test, p_2 = stats.spearmanr(y_test, y_pred)
    # # coef_test2 = np.corrcoef(y_test, y_pred)
    #
    # print(os.path.basename(file_n), "\nTest results with", optimum_complexity, "- spe:", coef_test)
    # print(20*'-')
    # util.results_to_csv(file_name='exp_results/results_{}_{}_srand_spec.csv'.format(task, feat_type[0]),
    #                     list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
    #                     list_values=[os.path.basename(file_n), ga, feat_type[2], optimum_complexity, coef_test,
    #                                  std_flag, 'TEST', srand])
    # # np.savetxt('preds_{}/preds_test_{}_srand_{}.txt'.format(feat, optimum_complexity, srand), y_pred)
    #
    # a = confusion_matrix(y_test, np.around(y_pred), labels=np.unique(y_train))
    # plot_confusion_matrix_2(a, np.unique(y_train), 'conf.png', cmap='Oranges', title="Spearman CC .365")
