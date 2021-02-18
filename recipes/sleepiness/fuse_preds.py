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
# srand_list = ['389743', '564896', '2656842', '2959019', '4336987', '7234786', '423877642', '987236753']
# srand_list = ['9612365', '764352323']
srand_list = ['764352323']

df = pd.read_csv('../../data/sleepiness/labels/labels.csv')

df_labels = df[df['file_name'].str.match('train')]
y_train = df_labels.label.values

df_labels = df[df['file_name'].str.match('dev')]
y_dev = df_labels.label.values

df_labels = df[df['file_name'].str.match('test')]
y_test = df_labels.label.values

feat_type = 'spec'
std_flag = False
c = '0.001'

best_preds_dev = []
best_preds_test = []

for srand in srand_list:
    preds_dev = np.loadtxt('preds_{}/best_preds_dev_0.001_srand_{}.txt'.format(feat_type, srand))
    coef_dev, p_std = stats.spearmanr(y_dev, preds_dev)
    print("SRAND {} - dev".format(srand), coef_dev)

    best_preds_dev.append(preds_dev)

comb_dev = np.mean((best_preds_dev), axis=0)

res_dev_t = sh.linear_trans_preds_dev(y_train=y_train, preds_dev=comb_dev)
tot_coef_dev, p_std = stats.spearmanr(y_dev, res_dev_t)

util.results_to_csv(file_name='exp_results/bests_comb_srand_xvecs_{}.csv'.format(feat_type),
                    list_columns=['Exp. Details', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
                    list_values=['xvecs_srand', c, tot_coef_dev, std_flag, 'DEV', "ALL"])

# LOAD TEST PREDICTIONS
for srand in srand_list:
    preds_test = np.loadtxt('preds_{}/preds_test_0.001_srand_{}.txt'.format(feat_type, srand))
    coef_test, p_std = stats.spearmanr(y_test, preds_test)
    print("SRAND {} - test".format(srand), coef_test)
    print()
    best_preds_test.append(preds_test)

comb_test = np.mean((best_preds_test), axis=0)

res_test_t = sh.linear_trans_preds_test(y_train=y_train, preds_dev=comb_dev, preds_test=comb_test)
tot_coef_test, p_std = stats.spearmanr(y_test, res_test_t)

util.results_to_csv(file_name='exp_results/bests_comb_srand_xvecs_{}.csv'.format(feat_type),
                    list_columns=['Exp. Details', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
                    list_values=['xvecs_srand', c, tot_coef_test, std_flag, 'TEST', "ALL"])
