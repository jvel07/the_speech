import numpy as np
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def load_data():
    work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'
    work_dir2 = 'D:/VHD/fishers'

    # Set data directories
    # file_train = work_dir + '/data/cold/train/fisher-13mf-2del-64g-train.fish'
    file_train = work_dir + '/data/cold/matlab-src/features.fv-mfcc-jose.improved.2.train.txt'
    # file_train = work_dir2 + '/features.fv-mfcc.improved.2.train.txt'
    lbl_train = work_dir + '/data/labels/labels.num.train.txt'

    # file_dev = work_dir + '/data/cold/dev/fisher-13mf-2del-64g-dev.fish'
    file_dev = work_dir + '/data/cold/matlab-src/features.fv-mfcc-jose.improved.2.dev.txt'
    # file_dev = work_dir2 + '/features.fv-mfcc.improved.2.dev.txt'
    lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

    # file_test = work_dir + '/data/cold/test/fisher-13mf-2del-64g-test.fish'
    file_test = work_dir + '/data/cold/matlab-src/features.fv-mfcc-jose.improved.2.test.txt'
    # file_test = work_dir2 + '/features.fv-mfcc.improved.2.test.txt'
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

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, X_combined, Y_combined


def resample_data(X, Y, r):
    smote_enn = RandomUnderSampler(random_state=r)
    X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
    return X_resampled, y_resampled


# train SVM model with stratified cross-validation
def train_model_stk_cv(X, Y, n_splits, _c):
    posteriors = []
    skf = StratifiedKFold(n_splits=n_splits)
    svc = svm.LinearSVC(C=_c, verbose=0, max_iter=3000)  # class_weight='balanced',
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test, y_train, y_test = \
            X[train_index], X[test_index], Y[train_index], Y[test_index]
        clf = CalibratedClassifierCV(base_estimator=svc, cv=7).fit(x_train, y_train)
        posteriors.append(clf.predict_proba(x_test))
        mean_post = np.mean(posteriors, axis=0)
        # np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_dev2_posteriors_{}.txt".format(str(c)),
        #            mean_post, fmt='%.7f')
        print("With:", _c)
        p0 = mean_post[:, 0:1]
        p1 = mean_post[:, 1:]
        y_p = 1 * (p1 > p0)
        print("Confusion matrix:\n", sk.metrics.confusion_matrix(y_test, y_p))
        one = sk.metrics.recall_score(y_test, y_p, pos_label=0)
        two = sk.metrics.recall_score(y_test, y_p, pos_label=1)
        print("UAR:", (one + two) / 2, "\n")

    return posteriors
