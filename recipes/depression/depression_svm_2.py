"""
Created by José Vicente Egas López
on 2021. 02. 02. 14 39

"""

import pandas as pd
from sklearn import preprocessing

from classifiers.svm_utils import svm_fits
from scipy import stats
import numpy as np


# load data
file = '/media/jose/hk-data/PycharmProjects/the_speech/data/depression/finaldb.txt'
df = pd.read_csv(file, delimiter='\t')
x_train = df.drop(['fileName'], axis=1)  # dropping 'fileName' column
x_train.fillna(0, inplace=True)

x_train['Sex'] = x_train['Sex'].astype('category')  # setting the 'sex' column as category
x_train_men = x_train[x_train['Sex'].str.match('F')]  # getting men's data
x_train_women = x_train[x_train['Sex'].str.match('N')]  # getting women's data

x_train_men['Sex'] = x_train_men['Sex'].cat.codes  # encoding cat to numbers
x_train_women['Sex'] = x_train_women['Sex'].cat.codes  # encoding cat to numbers

# getting male and female labels separately
y_train_men = x_train_men.BDI.values
y_train_women = x_train_women.BDI.values

# dropping class label (and do feature selection) from the train data
x_train_men = x_train_men.drop(['BDI', 'Sex', 'Age'], axis=1)
x_train_women = x_train_women.drop(['BDI', 'Sex', 'Age'], axis=1)

# getting the values from the dataframes
x_train_men = x_train_men.values
x_train_women = x_train_women.values

# std data
std_scaler = preprocessing.StandardScaler()
x_train_men = std_scaler.fit_transform(x_train_men)
x_train_women = std_scaler.fit_transform(x_train_women)

# train
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
for c in list_c:
    preds_women, trues_women = svm_fits.loocv_NuSVR_cpu(X=x_train_women, Y=y_train_women, c=c, kernel='linear')
    preds_men, trues_men = svm_fits.loocv_NuSVR_cpu(X=x_train_men, Y=y_train_men, c=c, kernel='linear')
    tot_preds = np.concatenate((preds_women, preds_men))
    tot_trues = np.concatenate((trues_women, trues_men))

    corr_women, _ = stats.pearsonr(trues_women, preds_women)
    corr_men, _ = stats.pearsonr(trues_men, preds_men)
    corr_tot, _ = stats.pearsonr(tot_trues, tot_preds)

    print("(women) with {}:".format(c), corr_women)
    print("(men) with {}:".format(c), corr_men)
    print("(total) with {}:".format(c), corr_tot)
    print()
