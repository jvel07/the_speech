"""
Created by José Vicente Egas López
on 2021. 02. 10. 11 22

"""
import numpy as np
import pandas as pd
from scipy.stats import stats
from recipes.sleepiness import sleepiness_helper as sh

from common import util

#
srand_list = ['389743', '564896', '2656842', '2959019', '4336987', '7234786', '9612365', '423877642', '987236753',
              '764352323']
# srand_list = ['9612365', '764352323']
# srand_list = ['764352323']

df = pd.read_csv('../../data/sleepiness/labels/labels.csv')

df_labels = df[df['file_name'].str.match('train')]
y_train = df_labels.label.values

df_labels = df[df['file_name'].str.match('dev')]
y_dev = df_labels.label.values

df_labels = df[df['file_name'].str.match('test')]
y_test = df_labels.label.values

feat_type = 'mfcc'

std_flag = False
stand = ''
if std_flag:
    stand = '_std'

c = '0.001'

# selecting n random srands to train the SVR with
for ra in [3, 5, 7, 9, 10]:
    no_srand_randoms = ra  # reset the seed for the same set of numbers to appear every time
    np.random.seed(7)  # for the seed to repeat everytime
    srand_idx = np.random.choice(10, size=no_srand_randoms, replace=False)  # generating list of srands indices
    selected_srands = [srand_list[index] for index in srand_idx]  # picking the corresponding srands
    srands_info = ' '.join(selected_srands)  # for information purposes, convert list to string

    best_preds_dev = []
    best_preds_test = []

    # LOAD DEV PREDICTIONS
    for srand in srand_list:
        preds_dev = np.loadtxt('preds_BEA_AUG_{}/best_preds_dev_0.0001_srand_{}{}.txt'.format(feat_type, srand, stand))
        coef_dev, p_std = stats.spearmanr(y_dev, preds_dev)
        print("SRAND {} - dev".format(srand), coef_dev)

        best_preds_dev.append(preds_dev)

    comb_dev = np.mean((best_preds_dev), axis=0)

    res_dev_t = sh.linear_trans_preds_dev(y_train=y_train, preds_dev=comb_dev)
    tot_coef_dev, p_std = stats.spearmanr(y_dev, res_dev_t)

    csv_name = 'results_srands_BEA_AUG_selected.csv'
    util.results_to_csv(file_name='exp_results_selected/{}'.format(csv_name),
                        list_columns=['Exp. Details', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
                        list_values=['xvecs_srand_{}'.format(feat_type), c, tot_coef_dev, std_flag, 'DEV', srands_info])

    # LOAD TEST PREDICTIONS
    for srand in selected_srands:
        preds_test = np.loadtxt('preds_BEA_AUG_{}/preds_test_0.0001_srand_{}{}.txt'.format(feat_type, srand, stand))
        coef_test, p_std = stats.spearmanr(y_test, preds_test)
        print("SRAND {} - test".format(srand), coef_test)
        print()
        best_preds_test.append(preds_test)

    comb_test = np.mean((best_preds_test), axis=0)

    res_test_t = sh.linear_trans_preds_test(y_train=y_train, preds_dev=comb_dev, preds_test=comb_test)
    tot_coef_test, p_std = stats.spearmanr(y_test, res_test_t)

    util.results_to_csv(file_name='exp_results_selected/{}'.format(csv_name),
                        list_columns=['Exp. Details', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
                        list_values=['xvecs_srand_{}'.format(feat_type), c, tot_coef_test, std_flag, 'TEST', srands_info])
