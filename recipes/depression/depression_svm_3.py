import pandas as pd
from sklearn import preprocessing

from classifiers.svm_utils import svm_fits
from scipy import stats
import numpy as np

# load data
net = 'sre16'
file = '/media/jose/hk-data/PycharmProjects/the_speech/data/depression/train/xvecs-23mfcc-0del-512dim-{}_vad-train.xvecs'.format(net)
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

concat_sex_age = False
if concat_sex_age:
    df['512'] = gender
    # df['513'] = age

x_train = df.values
y_train = df_lbl.label.values

#
# # std data
std = False
if std:
    std_scaler = preprocessing.StandardScaler()
    x_train = std_scaler.fit_transform(x_train)

# train SVR
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
for c in list_c:
    preds, trues = svm_fits.loocv_NuSVR_cpu(X=x_train, Y=y_train, c=c, kernel='linear')
    # preds, trues = svm_fits.loocv_SVR_cpu(X=x_train, Y=y_train, c=c, kernel='linear')
    corr, _ = stats.pearsonr(trues, preds)
    print("with {}:".format(c), corr)
