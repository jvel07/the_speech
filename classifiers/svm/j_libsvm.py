#from svmutil import *
from common import util
#from svm
from sklearn import preprocessing

######Loading training, dev and test features######
X_train = util.read_txt_file_as_list("../data/features.fv-mfcc26.4.train.txt")
X_dev = util.read_txt_file_as_list("../data/features.fv-mfcc26.4.dev.txt")
X_test = util.read_txt_file_as_list("../data/features.fv-mfcc26.4.test.txt")

######Loading training, dev and test labels######
y_train = util.read_txt_file_as_list("../data/labels.num.train.txt")
y_dev = util.read_txt_file_as_list("../data/labels.num.dev.txt")
y_test = util.read_txt_file_as_list("../data/labels.num.test.txt")

######Preprocessing data######
#print(X_train[0, 0])
X_train_new = preprocessing.scale(X_train)
X_train_scaled = X_train_new.tolist()
print("Data standarized...")

#####Building the model's parameters####
prob = svm_problem(y_train, X_train_scaled)
param = svm_parameter()
param.kernel_type = LINEAR
param.C = 0.01

###Training model###
m = svm_train(prob, param)

###Testing model###
pred_lbl, pred_acc, pred_val = svm_predict(y_dev, X_dev, m)


##########FOR GRID SEARCH (BEST PARAMS)###############
'''
results = []
for c in range(-5,10):
  for g in range(-8,4):
    param.C, param.gamma = 2**c, 2**g
    m = svm_train(problem,param)
    p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)
    results.append([param.C, param.gamma, p_acc[0]])

bestIdx = np.argmax(np.array(results)[:,2])
'''
###################