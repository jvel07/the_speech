import numpy as np
import pandas as pd
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm, preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer

from classifiers.cross_val import StatifiedGroupK_Fold

work_dir = 'C:/Users/Win10/PycharmProjects/the_speech' # windows machine
# work_dir = '/home/egasj/PycharmProjects/the_speech'  # ubuntu machine
work_dir2 = 'D:/VHD'


def load_data(gauss):
    # Set data directories
    # file_train = work_dir + '/data/cold/train/fisher-13mf-2del-{}g-train.fish'.format(gauss)
    file_train = work_dir2 + '/cold/matlab-src/features.fv-mfcc-jose.improved.{}.train.txt'.format(gauss)
    # file_train = work_dir2 + '/features.fv-mfcc.improved.{}.train.txt'.format(gauss)
    lbl_train = work_dir + '/data/labels/labels.num.train.txt'

    # file_dev = work_dir + '/data/cold/dev/fisher-13mf-2del-{}g-dev.fish'.format(gauss)
    file_dev = work_dir2 + '/cold/matlab-src/features.fv-mfcc-jose.improved.{}.dev.txt'.format(gauss)
    # file_dev = work_dir2 + '/features.fv-mfcc.improved.{}.dev.txt'.format(gauss)
    lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

    # file_test = work_dir + '/data/cold/test/fisher-13mf-2del-{}g-test.fish'.format(gauss)
    file_test = work_dir2 + '/cold/matlab-src/features.fv-mfcc-jose.improved.{}.test.txt'.format(gauss)
    # file_test = work_dir2 + '/features.fv-mfcc.improved.{}.test.txt'.format(gauss)
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

    return X_test, Y_test, X_combined, Y_combined


def load_compare_data():
    # Set data directories
    file_train = work_dir2 + '/cold/compare/features.train.txt'
    lbl_train = work_dir + '/data/labels/labels.num.train.txt'

    file_dev = work_dir2 + '/cold/compare/features.dev.txt'
    lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

    file_test = work_dir2 + '/cold/compare/features.test.txt'
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

    return X_test, Y_test, X_combined, Y_combined


def powert_data(X, X_dev, X_test):
    power = preprocessing.PowerTransformer()
    x_pow = power.fit_transform(X)
    x_powdev = power.transform(X_dev)
    x_powtest = power.transform(X_test)
    return x_pow, x_powdev, x_powtest


def normalize_data(X, X_dev, X_test):
    scaler = preprocessing.Normalizer().fit(X)
    X_train_norm = scaler.transform(X)
    X_dev_norm = scaler.transform(X_dev)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_dev_norm, X_test_norm


def resample_data(X, Y, r):
    smote_enn = RandomUnderSampler(random_state=r)
    X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
    indi = smote_enn.sample_indices_
    return X_resampled, y_resampled, indi


def read_utt_spk_lbl():
    df = pd.read_csv("C:/Users/Win10/PycharmProjects/the_speech/data/cold/uttNspkNlbl.txt", sep=" ", header=None)
    df.columns = ['wav', 'spk', 'label']
    utt = df.wav.values
    spk = df.spk.values
    lbl = df.label.values
    return spk


# Read utterance, speaker and labels to generating groups for CV
# e.g. devel_9596.wav vp053 0 (wav spk label)
def power_n(x, x_t):
    x_train = np.sqrt(np.abs(x)) * np.sign(x)
    x_test = np.sqrt(np.abs(x_t)) * np.sign(x_t)
    return x_train, x_test


# train SVM model with stratified cross-validation
def train_model_stk_cv(X, Y, n_splits, _c):
    # posteriors = []
    skf = StratifiedKFold(n_splits=n_splits)
    svc = svm.LinearSVC(C=_c, verbose=0, max_iter=3000)  # class_weight='balanced',
    list_uar = []
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test, y_train, y_test = \
            X[train_index], X[test_index], Y[train_index], Y[test_index]
        # clf = CalibratedClassifierCV(base_estimator=svc, cv=7).fit(x_train, y_train)
        # Training the SVM
        svc.fit(x_train, y_train)
        posteriors = svc._predict_proba_lr(x_test)
        # mean_post = np.mean(posteriors, axis=0)
        # np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_dev2_posteriors_{}.txt".format(str(c)),
        #            mean_post, fmt='%.7f')
        y_p = np.argmax(posteriors, axis=1)
        # print("Confusion matrix:\n", sk.metrics.confusion_matrix(y_test, y_p))
        one = sk.metrics.recall_score(y_test, y_p, pos_label=0)
        two = sk.metrics.recall_score(y_test, y_p, pos_label=1)
        uar = (one + two) / 2
        # print("UAR:", uar, "\n")
        list_uar.append(uar)

    uar_tot = np.mean(list_uar)
    return uar_tot, svc


# train SVM model with stratified GROUP cross-validation
def train_model_stkgroup_cv(X, Y, n_splits, _c, groups, gaussians):
    seeds = [137, 895642, 15986, 4242, 7117, 1255, 1, 923, 75, 9656]
    array_posteriors = np.zeros((len(Y), 2))

    sgkf = StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=n_splits)
    svc = svm.LinearSVC(C=_c, verbose=0, max_iter=3000)  # class_weight='balanced',
    for number in seeds:
        #print("\nSEED", number)
        for train_index, test_index in sgkf.split(X, Y, groups):
            x_train, x_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
            X_resampled, Y_resampled, indi = resample_data(x_train, y_train, r=number)  # resampling
            svc.fit(X_resampled, Y_resampled)  # Training the SVM
            posteriors = svc._predict_proba_lr(x_test)  # getting probs (posteriors)
            array_posteriors[test_index] = posteriors
            # print("index:", test_index)
            # print("post:", array_posteriors[test_index])
            # print("shape:", array_posteriors[test_index].shape)
        np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_cv_post_compare _{}_{}g.txt".format(str(_c), str(gaussians)),
            array_posteriors, fmt='%.7f')
        y_p = np.argmax(array_posteriors, axis=1)
        one = sk.metrics.recall_score(Y, y_p, pos_label=0)
        two = sk.metrics.recall_score(Y, y_p, pos_label=1)
        uar_tot = (one + two) / 2
    return uar_tot, svc


def train_model(X, Y, c):
    seeds = [137, 895642, 15986, 4242, 7117, 1255, 1564111, 923, 75, 9656]
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    for number in seeds:
        X_resampled, Y_resampled, indi = resample_data(X, Y, r=number)  # resampling
        svc.fit(X_resampled, Y_resampled)  # Training the SVM

    return svc
