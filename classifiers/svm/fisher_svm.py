from collections import Counter

from common import data_proc_tools

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'
work_dir2 = 'D:/VHD/fishers'
feat_type = '23mf'
n_filters = '512'
deltas = ''
vad = ''
num_gauss = ''

# Set data directories
#file_train = 'D:/VHD/fishers/fisher-13-2del-2-train'
file_train = work_dir2 + '/features.fv-mfcc.improved.2.train.txt'
lbl_train = work_dir + '/data/labels/labels.num.train.txt'

#file_dev = 'D:/VHD/fishers/fisher-13-2del-2-dev'
file_dev = work_dir2 + '/features.fv-mfcc.improved.2.dev.txt'
lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

#file_test = 'D:/VHD/fishers/fisher-13-2del-2-test'
file_test = work_dir2 + '/features.fv-mfcc.improved.2.test.txt'
lbl_test = work_dir + '/data/labels/labels.num.test.txt'

# Load dataste correo realizo cor
X_train = np.loadtxt(file_train, delimiter=',')
Y_train = np.loadtxt(lbl_train)

X_dev = np.loadtxt(file_dev, delimiter=',')
Y_dev = np.loadtxt(lbl_dev)

X_test = np.loadtxt(file_test, delimiter=',')
Y_test = np.loadtxt(lbl_test)

# Putting train and dev together
X_combined = np.concatenate((X_train, X_dev))
Y_combined = np.concatenate((Y_train, Y_dev))

# Binarizing labels
Y_train[Y_train == 2] = 0
Y_dev[Y_dev == 2] = 0
Y_test[Y_test == 2] = 0
Y_combined[Y_combined == 2] = 0


class LinearSVC_proba(LinearSVC):

    def __platt_func(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)
        probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        return probs


# Resampling
def resampling(X, Y, r):
   # print(sorted(Counter(Y).items()))
    smote_enn = RandomUnderSampler(random_state=r)
    X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
    #print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def powert(X, X_dev, X_test):
    power = preprocessing.PowerTransformer()
    x_pow = power.fit_transform(X)
    x_powdev = power.transform(X_dev)
    x_powtest = power.transform(X_test)
    return x_pow, x_powdev, x_powtest


def normalization(X, X_dev, X_test):
    norm = data_proc_tools.fit_normalize_data(X)
    X_train_norm = norm.transform(X)
    X_dev_norm = norm.transform(X_dev)
    X_test_norm = norm.transform(X_test)
    return X_train_norm, X_dev_norm, X_test_norm


def do_lda(X, Y, X_dev, X_test):
    lda = LinearDiscriminantAnalysis().fit(X, Y)
    X_lda = lda.transform(X)
    X_lda_dev = lda.transform(X_dev)
    X_lda_test = lda.transform(X_test)
    return X_lda, X_lda_dev, X_lda_test


com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
for c in com_values:
    posteriors = []
    for number in [37, 42, 16, 59, 77, 15, 1, 9, 25, 6]:
        X_resampled, Y_resampled = resampling(X_combined, Y_combined, r=number)
        X_pow, X_pow_dev, X_pow_test = powert(X_resampled, X_dev, X_test)
        X_norm, X_norm_dev, X_norm_test = normalization(X_pow, X_pow_dev, X_pow_test)


        clf = LinearSVC(C=c, verbose=0, max_iter=1000, class_weight='balanced').fit(X_norm, Y_resampled)
        #pipeline.set_params(svm__C=c, und__random_state=number).fit_resample(X_train, Y_train)
        # clf = CalibratedClassifierCV(base_estimator=pipeline, cv=10).fit(X,Y)
        # y_pr = pipeline.decision_function(X_dev)
        #y_pred = pipeline.predict(X_dev)
        posteriors.append(clf._predict_proba_lr(X_norm_dev))
    mean_post = np.mean(posteriors, axis=0)
    np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_test_posteriors_{}.txt".format(str(c)),
               mean_post, fmt='%.7f')
    print("With:", c)
    p0 = mean_post[:, 0:1]
    p1 = mean_post[:, 1:]
    y_p = 1*(p1 > p0)
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_dev, y_p))
    one = sk.metrics.recall_score(Y_dev, y_p, pos_label=0)
    two = sk.metrics.recall_score(Y_dev, y_p, pos_label=1)
    print("UAR:", (one + two) / 2, "\n")
