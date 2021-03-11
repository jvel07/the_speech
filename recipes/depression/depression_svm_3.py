import os

import pandas as pd
from sklearn import preprocessing

from classifiers.svm_utils import svm_fits
from scipy import stats
import numpy as np

# load data
from common import util

task = 'depression'
exp_info = ['xvecs', '23fbanks', 'BEA16k']  # feat_type, frame-level feat, DNN class

file = '/media/jose/hk-data/PycharmProjects/the_speech/data/depression/depression/{0}-{2}-0del-512dim-{1}_VAD_aug-train.{0}'\
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
keep_feats = [42]
for n in keep_feats:
    scores = []
    for c in list_c:
        preds, trues = svm_fits.loocv_NuSVR_cpu_pearson(X=x_train, Y=y_train, c=c, kernel='linear', keep_feats=n)
        # preds, trues = svm_fits.loocv_SVR_cpu(X=x_train, Y=y_train, c=c, kernel='linear')
        corr, _ = stats.pearsonr(trues, preds)
        scores.append(corr)
        print("with {}:".format(c), corr)
    print()

    best_c = list_c[np.argmax(scores)]
    best_corr = np.max(scores)
    util.results_to_csv(file_name='exp_results/results_depression.csv',#.format(task, feat_type[0]),
                        list_columns=['Exp. Details', 'C', 'STD', 'x-vec model', 'PEARSON', 'Gender', 'Age', 'KeepFeats'],
                        list_values=[os.path.basename(file), best_c, std, exp_info[2], best_corr, concat_sex, concat_age, n])

