# from optunity.solvers import GridSearch
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import sklearn as sk
import numpy as np
import bob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bob.learn.linear import WCCNTrainer
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import roc_auc_score

from classifiers.cross_val.cv_helper import grid_search
from common import data_proc_tools as t
from collections import Counter

work_dir = '/home/egasj/PycharmProjects/the_speech'
feat_type = '23mf'
n_filters = '512'
deltas = ''
vad = ''
num_gauss = ''

# Set data directories
file_train = work_dir + '/data/xvecs/xvecs-{}-{}-{}-{}-{}_ctrain2'.format(num_gauss, feat_type, deltas, vad, n_filters)
lbl_train = work_dir + '/data/labels/labels.num.train.txt'

file_dev = work_dir + '/data/xvecs/xvecs-{}-{}-{}-{}-{}_cdev2'.format(num_gauss, feat_type, deltas, vad, n_filters)
lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

file_test = work_dir + '/data/xvecs/xvecs-{}-{}-{}-{}-{}_ctest2'.format(num_gauss, feat_type, deltas, vad, n_filters)
lbl_test = work_dir + '/data/labels/labels.num.test.txt'

# Load data
X_train = np.loadtxt(file_train)
Y_train = np.loadtxt(lbl_train)

X_dev = np.loadtxt(file_dev)
Y_dev = np.loadtxt(lbl_dev)

X_test = np.loadtxt(file_test)
Y_test = np.loadtxt(lbl_test)

# y_train[y_train == 2] = 0
# le = preprocessing.LabelBinarizer()
# le.fit(y_train_str)
# y_train = le.transform(y_train_str)
# y_dev = le.transform(y_dev_str)
# y_test = le.transform(y_test_str)

print(sorted(Counter(Y_train).items()))
# Resampling
smote_enn = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, Y_train)
print(sorted(Counter(y_resampled).items()))

# pipeline

pipeline = Pipeline(
    [
        ('standardize', preprocessing.StandardScaler()),
        ('svm', LinearSVC(verbose=0, class_weight='balanced', max_iter=10000))
    ])

parameters = {
    'svm__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
}

gridder = GridSearchCV(pipeline, parameters, scoring='roc_auc', n_jobs=-1, cv=5)
gridder.fit(X_train, Y_train)

y_pred = gridder.predict(X_dev)
y_pr=gridder.decision_function(X_dev)

print('predict', roc_auc_score(Y_dev, y_pred))
print('decision', roc_auc_score(Y_dev, y_pr))
print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_dev, y_pred))
