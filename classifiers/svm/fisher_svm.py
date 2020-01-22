from collections import Counter

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
file_train = 'D:/VHD/fishers/fisher-13-2del-8-train'
#file_train = work_dir2 + '/features.fv-mfcc.improved.2.train.txt'
lbl_train = work_dir + '/data/labels/labels.num.train.txt'

file_dev = 'D:/VHD/fishers/fisher-13-2del-8-dev'
#file_dev = work_dir2 + '/features.fv-mfcc.improved.2.dev.txt'
lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

file_test = 'D:/VHD/fishers/fisher-13-2del-8-test'
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


class LinearSVC_proba(LinearSVC):

    def __platt_func(self,x):
        return 1/(1+np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)
        probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        return probs

# Resampling
def resampling(X, Y):
    print(sorted(Counter(Y).items()))
    smote_enn = RandomUnderSampler()
    X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


# pipeline
pipeline = Pipeline(
    [
        #('power', preprocessing.PowerTransformer()),
        #('standardize', preprocessing.StandardScaler()),
        ('normalizer', preprocessing.Normalizer()),
        ('und', RandomUnderSampler()),
        #('lda', LinearDiscriminantAnalysis()),
        #('logistic', sk.linear_model.SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None))
        ('svm', LinearSVC(verbose=0, max_iter=1000, class_weight='balanced')),
    ])


#y_pred_prob = svc_clf._predict_proba_lr(X_dev_scaled)

com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
for c in com_values:
    posteriors = []
    for number in [37, 44, 16, 59, 77, 15,  1,  9,  25,  6]:
        pipeline.set_params(svm__C=c, und__random_state=number)#.fit(X,Y)
        clf = CalibratedClassifierCV(base_estimator=pipeline, cv=10).fit(X,Y)
        #y_pr = pipeline.decision_function(X_dev)
        y_pred = pipeline.predict(X_dev)
        posteriors.append(clf.predict_proba(X_dev))
    mean_post = np.mean(posteriors, axis=0)
    np.savetxt("C:/Users/Win10/PycharmProjects/the_speech/data/cold/posteriors/mean_posteriors_{}.txt".format(str(c)), mean_post, fmt='%.7f')
    print("With:", c)
    for i in mean_post:
        highest = np.amax(i)
    #print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_pred))
    #one = sk.metrics.recall_score(Y_test, y_pred, pos_label=0)
    #two = sk.metrics.recall_score(Y_test, y_pred, pos_label=1)
    #print("UAR:", (one + two) / 2, "\n")
