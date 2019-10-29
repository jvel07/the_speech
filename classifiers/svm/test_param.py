import optunity
import optunity.metrics
import sklearn.svm
import numpy as np
from sklearn import preprocessing

X_train = np.loadtxt('../data/ivecs/cold/ivecs-16g-cold-128i-train')
X_dev = np.loadtxt('../data/ivecs/cold/ivecs-16g-cold-128i-dev')
y_train = np.loadtxt("../data/ivecs/cold/labels.num.train.txt")
y_dev = np.loadtxt("../data/ivecs/cold/labels.num.dev.txt")

y_train = y_train[-7604:]

X_train_scaled = preprocessing.scale(X_train)
X_dev_scaled = preprocessing.scale(X_dev)


# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=X_train_scaled, y=y_train, num_folds=100, num_iter=7)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma, kernel='linear').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)


# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=100, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
print("learning on the training dataset")
optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(X_train_scaled, y_train)
