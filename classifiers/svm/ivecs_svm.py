from collections import Counter

from sklearn.decomposition import PCA

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

work_dir = '/home/egasj/PycharmProjects/the_speech'
work_dir2 = 'D:/fishers'
feat_type = '23mf'
n_filters = '512'
deltas = ''
vad = ''
num_gauss = ''

# Set data directories
file_train = work_dir + '/data/ivecs/ivecs-32-23mf---300_train'
lbl_train = work_dir + '/data/labels/labels.num.train.txt'

file_dev = work_dir + '/data/ivecs/ivecs-32-23mf---300_dev'
lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

file_test = work_dir + '/data/ivecs/ivecs-32-23mf---300_test'
lbl_test = work_dir + '/data/labels/labels.num.test.txt'

# Load dataste correo realizo cor
X_train = np.loadtxt(file_train, delimiter=' ')
Y_train = np.loadtxt(lbl_train)

X_dev = np.loadtxt(file_dev, delimiter=' ')
Y_dev = np.loadtxt(lbl_dev)

X_test = np.loadtxt(file_test, delimiter=' ')
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
    # print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


# pipeline
pipeline = Pipeline(
    [
        # ('power', preprocessing.PowerTransformer()),
        # ('standardize', preprocessing.StandardScaler()),
        ('normalizer', preprocessing.MinMaxScaler()),
        ('und', RandomUnderSampler()),
        # ('lda', LinearDiscriminantAnalysis()),
        # ('logistic', sk.linear_model.SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None))
        ('svm', SVC(kernel='linear', verbose=1, max_iter=3000, class_weight='balanced', probability=True)),
    ])


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
    norm = data_proc_tools.fit_standardize_data(X)
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


def do_pca(X, Y, X_dev, X_test):
    lda = PCA(n_components=0.95, svd_solver='full', whiten=True).fit(X, Y)
    X_lda = lda.transform(X)
    X_lda_dev = lda.transform(X_dev)
    X_lda_test = lda.transform(X_test)
    return X_lda, X_lda_dev, X_lda_test

#plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=20, cmap=plt.cm.Spectral);


com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
for c in com_values:
    posteriors = []
    for number in [137, 42, 15986, 4242, 7117, 15, 1, 923, 25, 656]:
        X_resampled, Y_resampled = resampling(X_train, Y_train, r=number)  # resampling
        # X_pow, X_pow_dev, X_pow_test = powert(X_resampled, X_dev, X_test)  # Power Norm
        X_norm, X_norm_dev, X_norm_test = normalization(X_resampled, X_dev, X_test) # (X_pow, X_pow_dev, X_pow_test)
        # X_lda, X_lda_dev, X_lda_test = do_lda(X_norm, Y_resampled, X_norm_dev, X_norm_test)  # LDA
       #  clf = svm.SVC(kernel='linear', C=c, verbose=0, max_iter=1000, probability=True, class_weight='balanced')
       #  clf = LinearSVC(C=c, verbose=0, max_iter=1000)
       #  clf.fit(X_norm, Y_resampled)
        #pipeline.set_params(svm__C=c, und__random_state=number).fit_resample(X_train, Y_train)
        clf = CalibratedClassifierCV(base_estimator=LinearSVC(C=c), cv=10).fit(X_norm, Y_resampled)
        # y_pr = pipeline.decision_function(X_dev)
        #y_pred = pipeline.predict(X_dev)
        posteriors.append(clf.predict_proba(X_norm_dev))
    mean_post = np.mean(posteriors, axis=0)
    # np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_dev2_posteriors_{}.txt".format(str(c)),
    #            mean_post, fmt='%.7f')
    print("With:", c)
    p0 = mean_post[:, 0:1]
    p1 = mean_post[:, 1:]
    y_p = 1*(p1 > p0)
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_dev, y_p))
    one = sk.metrics.recall_score(Y_dev, y_p, pos_label=0)
    two = sk.metrics.recall_score(Y_dev, y_p, pos_label=1)
    print("UAR:", (one + two) / 2, "\n")
