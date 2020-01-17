from collections import Counter

import numpy as np
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

work_dir = '/home/egasj/PycharmProjects/the_speech'
feat_type = '23mf'
n_filters = '512'
deltas = ''
vad = ''
num_gauss = ''

# Set data directories
file_train = work_dir + '/data/fisher/fisher-13-2del-32-train'
#file_train = work_dir + '/data/fisher/features.fv-mfcc.improved.2.train.txt'
lbl_train = work_dir + '/data/labels/labels.num.train.txt'

file_dev = work_dir + '/data/fisher/fisher-13-2del-32-test'
#file_dev = work_dir + '/data/fisher/features.fv-mfcc.improved.2.dev.txt'
lbl_dev = work_dir + '/data/labels/labels.num.test.txt'

#file_test = work_dir + '/data/fisher/features.fv-mfcc.improved.2.test.txt'
#lbl_test = work_dir + '/data/labels/labels.num.test.txt'

# Load dataste correo realizo cor
X_train = np.loadtxt(file_train, delimiter=' ')
Y_train = np.loadtxt(lbl_train)

X_dev = np.loadtxt(file_dev, delimiter=' ')
Y_dev = np.loadtxt(lbl_dev)

#X_test = np.loadtxt(file_test, delimiter=',')
#Y_test = np.loadtxt(lbl_test)

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
        ('power', preprocessing.PowerTransformer()),
        #('standardize', preprocessing.StandardScaler()),
        ('normalizer', preprocessing.Normalizer()),
        ('svm', SVC(verbose=0, max_iter=10000))
    ])

com_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
for c in com_values:
    pipeline.set_params(svm__C=c).fit(X_resampled, y_resampled)
    y_pr = pipeline.decision_function(X_dev)
    y_pred = pipeline.predict(X_dev)
    print("With:", c)
    print('predict', roc_auc_score(Y_dev, y_pred))
    print('score', pipeline.score(X_dev, Y_dev))
    #print('decision', roc_auc_score(Y_dev, y_pr))
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_dev, y_pred))
    #one = sk.metrics.recall_score(Y_dev, y_pred, pos_label=1)
    #two = sk.metrics.recall_score(Y_dev, y_pred, pos_label=2)
