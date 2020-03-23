import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import recall_score

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

task = 'mask'
feat_type = 'fisher'

# Loading data: 'fisher' or 'ivecs'
x_train, x_dev, x_test, y_train, y_dev = rutils.load_data_full(gauss=64, task=task, feat_type=feat_type, n_feats=23,
                                                               n_deltas=1, label_1='mask', label_0='clear')
# x_train, y_train = rutils.load_data_alternate(64, 'monologue')
x_combined = np.concatenate((x_train, x_dev))
y_combined = np.concatenate((y_train, y_dev))

# Scale data
std_scaler = preprocessing.StandardScaler()
pow_scaler = preprocessing.PowerTransformer()
norm_scaler = preprocessing.Normalizer()
#x_train = preprocessing.PowerTransformer().fit_transform(x_train)
#x_train = preprocessing.Normalizer().fit_transform(x_train)
x_train = std_scaler.fit_transform(x_train)
x_dev = std_scaler.transform(x_dev)
x_combined = std_scaler.fit_transform(x_combined)
x_test = std_scaler.transform(x_test)

# Training data and evaluating (stratified k-fold CV)
for c in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]:
    # svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    # scores = cross_validate(svc, x_train, y_train, cv=5, scoring=('roc_auc', 'accuracy', 'f1'))
    # print("with c", c, "-->", np.mean(scores["test_accuracy"]), np.mean(scores["test_roc_auc"]), np.mean(scores["test_f1"]))
    list_scores = []
    #for seed in [44654]:# [1367, 684531, 8754, 3215, 54, 3551, 63839845, 11538, 148111, 4310]:
     #   scores = svm_fits.train_simple_skfcv(x_train, y_train.ravel(), n_folds=10, c=c, seed=seed)
      #  list_scores.append(scores)
       # print("with c", c, "-->", scores["accuracy"], scores["auc"], scores["f1"])


def predict(x, y, x_t, c):
    svc = svm.LinearSVC(C=c, verbose=0, max_iter=100000, class_weight='balanced', loss='hinge')
    svc.fit(x, y)
    y_pred = svc._predict_proba_lr(x_t)
    y_pred = np.argmax(y_pred, axis=1)
    #write_out(y_pred)
    return y_pred, svc


team_name = 'NaN'
submission_index = 1
label_file = '../data/mask/labels/labels.csv'
def write_out(y_pred):
    df_labels = pd.read_csv(label_file)
    pred_file_name = task + '.' + feat_type + '.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': y_pred.flatten()},
                      columns=['file_name', 'prediction'])
    df.to_csv(pred_file_name, index=False)


# Predicting and generating submission
for c in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]:
    y_pred, svc = predict(x=x_train, y=y_train.ravel(), x_t=x_dev, c=c)
    print(recall_score(y_dev, y_pred, labels=[0, 1], average='macro'))
