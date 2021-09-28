import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import make_scorer

from classifiers.svm_utils import svm_fits
from scipy import stats


# load data
file = '/home/jvel/PycharmProjects/the_speech/data/depression/xvecs-23fbanks-0del-512dim-BEA16k_VAD_aug-train.xvecs'
df = pd.read_csv(file, delimiter=' ')
# x_train = df.drop(['fileName', 'BDI'], axis=1)
# x_train.fillna(0, inplace=True)

# x_train['Sex'] = x_train['Sex'].astype('category')  # setting the 'sex' column as category
# x_train['Sex'] = x_train['Sex'].cat.codes  # encoding cat to numbers

# x_train = x_train.drop(['Age', 'Sex'], axis=1)  # dropping sex and age columns

x_train = df.values
y_train = df.BDI.values

# std data
std_scaler = preprocessing.StandardScaler()
x_train = std_scaler.fit_transform(x_train)

def pearson_scoring(y_true, y_pred, **kwargs):
    corr, _ = stats.pearsonr(y_true, y_pred)
    return corr

my_scorer = make_scorer(pearson_scoring, greater_is_better=True)

# train
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
for c in list_c:
    # preds, trues = svm_fits.loocv_NuSVR_cpu(X=x_train, Y=y_train, c=c, kernel='linear')
    mean_nested_score = svm_fits.train_nested_cv_nuSVR(X=x_train, Y=y_train, kernel='linear', metric=my_scorer)
    print("with {}:".format(c), mean_nested_score)
