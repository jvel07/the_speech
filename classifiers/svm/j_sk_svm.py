from sklearn import svm
from sklearn import preprocessing
import sklearn as sk
import numpy as np
import bob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bob.learn.linear import WCCNTrainer
#from imblearn.combine import SMOTETomek, SMOTEENN
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler

# Load data
work_dir = '/home/egasj/PycharmProjects/the_speech'
feat_type = '23mf'
n_filters = '512'
deltas = ''
vad = ''
num_gauss = ''

# Set data directories
file_train = work_dir + '/data/xvecs/xvecs-{}-{}-{}-{}-{}'.format(num_gauss, feat_type, deltas, vad, n_filters)
lbl_train = work_dir + '/data/labels/labels.num.train.txt'

file_dev = work_dir + '/data/xvecs/xvecs-{}-{}-{}-{}-{}'.format(num_gauss, feat_type, deltas, vad, n_filters)
lbl_dev = work_dir + '/data/labels/labels.num.dev.txt'

file_test = work_dir + '/data/xvecs/xvecs-{}-{}-{}-{}-{}'.format(num_gauss, feat_type, deltas, vad, n_filters)
lbl_test = work_dir + '/data/labels/labels.num.test.txt'

# Load data
X_train = np.loadtxt(file_train)
Y_train = np.loadtxt(lbl_train)

X_dev = np.loadtxt(file_dev)
Y_dev = np.loadtxt(lbl_dev)

X_test = np.loadtxt(file_test)
Y_test = np.loadtxt(lbl_test)

#y_train[y_train == 2] = 0
#le = preprocessing.LabelBinarizer()
#le.fit(y_train_str)
#y_train = le.transform(y_train_str)
#y_dev = le.transform(y_dev_str)
#y_test = le.transform(y_test_str)

# Train data
svc_clf = svm.LinearSVC(C=1, class_weight='balanced', max_iter=15886, verbose=1)
svc_clf.fit(X_train_und, y_train)
print("Data trained...")

# Predicting
y_pred = svc_clf.predict(X_dev)
#y_pred_prob = svc_clf._predict_proba_lr(X_dev_scaled)
#util.save_data_to_file("sk_svm_dem_prob_results.txt", y_pred_prob, '%.4f')

# Evaluation
print("Confusion matrix:\n", sk.metrics.confusion_matrix(y_dev, y_pred))
cm = sk.metrics.confusion_matrix(y_dev, y_pred)
recall1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
recall2 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
uar = (recall1 + recall2) / 2
print(uar)

