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

num_gauss = 2
ivector_dim = 100
obs = 'full-tmtubm-scl'
a = 'bob1'

# Load data
X_train_und = np.loadtxt('../data/ivecs/cold/ivecs-{}-g-cold-{}i-train-{}'.format(num_gauss, ivector_dim, a))
y_train = np.loadtxt('../data/ivecs/cold/labels.num.train.txt')
X_dev = np.loadtxt('../data/ivecs/cold/ivecs-{}g-cold-{}i-dev-{}'.format(num_gauss, ivector_dim, obs))
y_dev = np.loadtxt('../data/ivecs/cold/labels.num.dev.txt')
#X_test = np.loadtxt('../data/ivecs/cold/ivecs-{}g-cold-{}i-test-{}'.format(num_gauss, ivector_dim, obs))
#y_test_str = np.loadtxt('../data/ivecs/cold/labels.test.txt', dtype='str')

#y_train[y_train == 2] = 0
#le = preprocessing.LabelBinarizer()
#le.fit(y_train_str)
#y_train = le.transform(y_train_str)
#y_dev = le.transform(y_dev_str)
#y_test = le.transform(y_test_str)

#smt = RandomOverSampler(random_state=0)
#X_train_r, y_train = smt.fit_resample(X_train_und, np.ravel(y_train_und))


# LDA on i-vecs
#lda = LDA(n_components=16)
#X_train_und = lda.fit_transform(X_train_und, y_train)
#X_dev = lda.transform(X_dev)
#X_test = lda.transform(X_test)

# Preprocessing data
#print(X_train[0, 0])
#scaler = preprocessing.StandardScaler().fit(X_train_und)
#X_train_scaled = scaler.transform(X_train_und)
#X_dev_scaled = scaler.transform(X_dev)
#X_test_scaled = scaler.transform(X_test)
print("Data standarized...")

a_ivectors_3d = np.expand_dims(X_train_und, axis=1)
lda_trainer = bob.learn.linear.FisherLDATrainer(use_pinv=True)
lda_machine = bob.learn.linear.Machine(100, 100)
lda_trainer.train(a_ivectors_3d, lda_machine)
X_train_und = lda_machine(X_train_und)
X_dev = lda_machine(X_dev)



# Training data
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

'''
######Parameter Tuning######
# Set the parameters by cross-validation
tuned_parameters = [
                    #{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]},
                 #   {'LinearSVC': ['LinearSVC()'], 'C': [1, 10, 100, 1000]}
                    ]
lsvc_params = [
    {'C': [0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]}
]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf_svc = GridSearchCV(LinearSVC(), lsvc_params, cv=5)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)
    clf_svc.fit(X_dev, y_dev)

    print("Best parameters set found on development set:")
    print()
    
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_dev)
    print(classification_report(y_true, y_pred))
    print
'''


