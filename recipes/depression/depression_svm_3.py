import os

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import recall_score, roc_auc_score, f1_score, mean_squared_error

from classifiers.svm_utils import svm_fits
from scipy import stats
import numpy as np

# load data
from common import util, metrics
from common.metrics import calculate_sensitivity_specificity

task = 'depression'
exp_info = ['xvecs', '23fbanks', 'BEA16k_VAD_aug']  # feat_type, frame-level feat, DNN class

file = '/media/jose/hk-data/PycharmProjects/the_speech/data/depression/depression/{0}-{2}-0del-512dim-{1}-train.{0}'\
    .format(exp_info[0], exp_info[2], exp_info[1])
# file = '/media/jose/hk-data/PycharmProjects/the_speech/data/depression/depression/ivecs-20fbanks-0del-256g-depression.ivecs'
df = pd.read_csv(file, delimiter=' ', header=None)

# load labels
label_file = '../../data/depression/labels/labels_2.csv'
df_lbl = pd.read_csv(label_file, delimiter=',')
data = df_lbl.drop(['filename'], axis=1)  # dropping 'fileName' column
data.fillna(0, inplace=True)

data['Sex'] = data['Sex'].astype('category')  # setting the 'sex' column as category
data['Sex'] = data['Sex'].cat.codes  # encoding cat to numbers

gender, age = data.Sex.values, data.Age.values

# x_train['Sex'] = x_train['Sex'].astype('category')  # setting the 'sex' column as category
# x_train['Sex'] = x_train['Sex'].cat.codes  # encoding cat to numbers

# x_train = x_train.drop(['Age', 'Sex'], axis=1)  # dropping sex and age columns

concat_sex = True
if concat_sex:
    df['512'] = gender

concat_age = True
if concat_age:
    df['513'] = age

x_train = df.values
y_train = df_lbl.label.values


# std data
std = True
if std:
    std_scaler = preprocessing.StandardScaler()
    x_train = std_scaler.fit_transform(x_train)

# train SVR
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
keep_feats = [25, 50, 75, 100, 125, 150, 175, 200]
# keep_feats = [150, 514]
# keep_feats = [160]

feats_cc_scores = []
feats_mrsq_scores = []
feats_auc_scores = []
feats_uar_scores = []

for n in keep_feats:
    print("Keep feats: {}".format(n))
    corr_scores = []
    uar_scores = []
    auc_scores = []
    f1_scores = []
    spec_scores = []
    sens_scores = []
    eer_scores = []

    for c in list_c:
        preds, trues = svm_fits.loocv_NuSVR_cpu_pearson(X=x_train, Y=y_train, c=c, kernel='linear', keep_feats=n)
        # preds, trues = svm_fits.loocv_SVR_cpu(X=x_train, Y=y_train, c=c, kernel='linear')
        corr, _ = stats.pearsonr(trues, preds)
        corr_scores.append(corr)

        # binary class
        trues_bin = np.copy(trues)
        trues_bin[trues_bin < 13.5] = 0
        trues_bin[trues_bin >= 13.5] = 1
        preds_bin = np.copy(preds)
        preds_bin[preds_bin < 13.5] = 0
        preds_bin[preds_bin >= 13.5] = 1

        # metrics
        auc = roc_auc_score(trues_bin, preds)
        auc_scores.append(auc)
        uar = recall_score(trues_bin, preds_bin, average='macro')
        uar_scores.append(uar)
        sens_scores.append(recall_score(trues_bin, preds_bin))
        sensitivity, specificity, accuracy = calculate_sensitivity_specificity(trues_bin, preds_bin)
        spec_scores.append(specificity)
        f1 = f1_score(trues_bin, preds_bin)
        f1_scores.append(f1)
        # eer = metrics.calculate_eer(trues_bin, preds_bin)
        eer = mean_squared_error(trues, preds, squared=False)
        eer_scores.append(eer)

        # print("with {}:".format(c), corr)
        print("with {}:".format(c), "corr:", corr, "uar:", uar, "spec:", specificity, "sens:", sensitivity,
              "AUC:", auc, "F1:", f1, "RMSE:", eer)
    print()

    best_c = list_c[np.argmax(corr_scores)]
    best_corr = np.max(corr_scores)
    best_eer = eer_scores[np.argmax(corr_scores)]
    best_uar = uar_scores[np.argmax(corr_scores)]
    best_sens = sens_scores[np.argmax(corr_scores)]
    best_spec = spec_scores[np.argmax(corr_scores)]
    best_f1 = f1_scores[np.argmax(corr_scores)]
    best_auc = auc_scores[np.argmax(corr_scores)]

    feats_cc_scores.append(best_corr)
    feats_auc_scores.append(best_auc)
    feats_mrsq_scores.append(best_eer)
    feats_uar_scores.append(best_uar)

    # util.results_to_csv(file_name='exp_results/results_depression_3.csv',#.format(task, feat_type[0]),
    #                     list_columns=['Exp. Details', 'C', 'STD', 'UAR', 'SPEC', 'SENS', 'AUC', 'f1', 'EER', 'PEARSON',
    #                                   'x-vec model', 'KeepFeats'],
    #                     list_values=[os.path.basename(file), best_c, std, best_uar, best_spec, best_sens, best_auc,
    #                                  best_f1, best_eer, best_corr,
    #                                  exp_info[2], n])

