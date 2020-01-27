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


# Resampling
def resampling(X, Y, r):
   # print(sorted(Counter(Y).items()))
    smote_enn = RandomUnderSampler(random_state=r)
    X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
    #print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled



# pipeline
pipeline = Pipeline(
    [
        ('power', preprocessing.PowerTransformer()),
        # ('standardize', preprocessing.StandardScaler()),
        ('normalizer', preprocessing.Normalizer()),
        ('und', RandomUnderSampler()),
        #('lda', PCA(n_components=0.95)),
        #('logistic', sk.linear_model.SGDClassifier(loss="hinge", eta0=1, learning_rate="constant", penalty='l2'))
        ('svm', LinearSVC(verbose=0, max_iter=3000, class_weight='balanced')),
    ])

com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
for c in com_values:
    pipeline.set_params(svm__C=c, und__random_state=42).fit(X_train, Y_train)
    # clf = CalibratedClassifierCV(base_estimator=pipeline, cv=10).fit(X,Y)
    y_p = pipeline.decision_function(X_dev)
    y_pred = pipeline.predict(X_dev)
    print("With:", c)
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_dev, y_pred))
    one = sk.metrics.recall_score(Y_dev, y_pred, pos_label=0)
    two = sk.metrics.recall_score(Y_dev, y_pred, pos_label=1)
    print("UAR:", (one + two) / 2, "\n")
