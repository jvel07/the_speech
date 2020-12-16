import pandas as pd
from sklearn import preprocessing

from classifiers.svm_utils import svm_fits
from scipy import stats


# load data
file = '/media/jose/hk-data/PycharmProjects/the_speech/data/depression/finaldb.txt'
df = pd.read_csv(file, delimiter='\t')
x_train = df.drop(['fileName', 'BDI'], axis=1)
x_train.fillna(0, inplace=True)
x_train['Sex'] = x_train['Sex'].astype('category')  # setting the 'sex' column as category
x_train['Sex'] = x_train['Sex'].cat.codes  # encodingc cat to numbers

x_train = x_train.drop(['Age', 'Sex'], axis=1)

x_train = x_train.values
y_train = df.BDI.values

# std data
std_scaler = preprocessing.RobustScaler()
x_train = std_scaler.fit_transform(x_train)

# train
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
for c in list_c:
    preds, trues = svm_fits.normalCV_svr_cpu(X=x_train, Y=y_train, test_size=0.4, c=c, kernel='linear')
    corr, _ = stats.pearsonr(trues, preds)
    print("with {}:".format(c), corr)