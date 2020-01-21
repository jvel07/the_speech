from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'
work_dir2 = 'D:/fishers'
feat_type = '23mf'
n_filters = '512'
deltas = ''
vad = ''
num_gauss = ''

# Set data directories
file_train = 'D:/fishers/fisher-13-2del-8-train'
#file_train = work_dir2 + '/features.fv-mfcc.improved.2.train.txt'
lbl_train = work_dir + '/data/labels/labels.num.train.txt'

file_dev = 'D:/fishers/fisher-13-2del-8-dev'
#file_dev = work_dir2 + '/features.fv-mfcc.improved.2.dev.txt'
lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

file_test = 'D:/fishers/fisher-13-2del-8-test'
#file_test = work_dir2 + '/features.fv-mfcc.improved.2.test.txt'
lbl_test = work_dir + '/data/labels/labels.num.test.txt'

# Load dataste correo realizo cor
X_train = np.loadtxt(file_train, delimiter=' ')
Y_train = np.loadtxt(lbl_train)

X_dev = np.loadtxt(file_dev, delimiter=' ')
Y_dev = np.loadtxt(lbl_dev)

X_test = np.loadtxt(file_test, delimiter=' ')
Y_test = np.loadtxt(lbl_test)

# Putting train and dev together
X = np.concatenate((X_train, X_dev))
Y = np.concatenate((Y_train, Y_dev))

# Binarizing labels
Y_train[Y_train == 2] = 0
Y_dev[Y_dev == 2] = 0
Y_test[Y_test == 2] = 0
Y[Y == 2] = 0


# le = preprocessing.LabelBinarizer()
# le.fit(y_train_str)
# y_train = le.transform(y_train_str)
# y_dev = le.transform(y_dev_str)
# y_test = le.transform(y_test_str)

# Resampling
def resampling(X, Y):
    print(sorted(Counter(Y).items()))
    smote_enn = RandomUnderSampler()
    X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


#X_resampled, y_resampled = resampling(X=X, Y=Y)


# pipeline
pipeline = Pipeline(
    [
        #('power', preprocessing.PowerTransformer()),
        #('standardize', preprocessing.StandardScaler()),
        ('normalizer', preprocessing.Normalizer()),
        ('smt', RandomUnderSampler(random_state=0, n_jobs=-1)),
        #('lda', LinearDiscriminantAnalysis()),
        #('logistic', sk.linear_model.SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None))
        ('svm', LinearSVC(verbose=0, max_iter=1000, class_weight='balanced'))
    ])


com_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
for c in com_values:
    for number in randomnumbers:
        pipeline.set_params(svm__C=c, smt__random_state=number).fit(X, Y)
        #pipeline.fit(X, Y)
        y_pr = pipeline.decision_function(X_test)
        y_pred = pipeline.predict(X_test)
        print("With:", c)
        # print('predict', roc_auc_score(Y_test, y_pred))
        # print('score', pipeline.score(X_test, Y_test))
        # print('Precision recall curve', precision_recall_curve(X_dev, Y_dev))
        # print('decision', roc_auc_score(Y_dev, y_pr))

        print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_pred))
        one = sk.metrics.recall_score(Y_test, y_pred, pos_label=0)
        two = sk.metrics.recall_score(Y_test, y_pred, pos_label=1)
        print("UAR:", (one + two) / 2, "\n")
