import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

task = 'mask'
feat_type = 'fisher'

# Loading data: 'fisher' or 'ivecs's
x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_full(gauss=128, task=task, feat_type=feat_type, n_feats=23, n_deltas=2, list_labels=['mask','clear'])
# x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_compare()

# x_train, y_train = rutils.load_data_alternate(64, 'monologue')
# x_combined = np.concatenate((x_train, x_dev))
# y_combined = np.concatenate((y_train, y_dev))


# Scale data
std_scaler = preprocessing.StandardScaler()
# pow_scaler = preprocessing.PowerTransformer()
# norm_scaler = preprocessing.PowerTransformer()

x_train = std_scaler.fit_transform(x_train)
x_dev = std_scaler.transform(x_dev)
# x_combined = std_scaler.fit_transform(x_combined)
# x_test = std_scaler.transform(x_test)

# pca = PCA(n_components=0.98)
# x_train = pca.fit_transform(x_train)
# x_dev = pca.transform(x_dev)
# x_combined = pca.fit_transform(x_combined)
# x_test = pca.transform(x_test)
#
del x_test

# Training data and evaluating (stratified k-fold CV)
for c in [1e-7]:# [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]:
    # svc = svm.LinearSVC(C=c, verbose=0, max_iter=3000)  # class_weight='balanced',
    # scores = cross_validate(svc, x_train, y_train, cv=5, scoring=('roc_auc', 'accuracy', 'f1'))
    # print("with c", c, "-->", np.mean(scores["test_accuracy"]), np.mean(scores["test_roc_auc"]), np.mean(scores["test_f1"]))
    list_scores = []
    for seed in [44654]:  # [1367, 684531, 8754, 3215, 54, 3551, 63839845, 11538, 148111, 4310]:
        # score, posteriors = svm_fits.train_thunder_svm_simple(x_train, y_train.ravel(), c=c, X_eval=x_dev, Y_eval=y_dev)
        # score, posteriors = svm_fits.train_simple_skfcv(x_combined, y_combined.ravel(), n_folds=10, c=c, seed=seed)
        score, posteriors = svm_fits.train_model_normal(x_train, y_train.ravel(), c=c, X_t=x_dev, Y_t=y_dev)
        # list_scores.append(score)
    # print("with c", c, "-->", score["uar"])
    np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_{}_fisher.txt'.format(c), posteriors)
    print("with c", c, score)


# submission
def predict(best_c):
    svc = svm.LinearSVC(C=best_c, verbose=0, max_iter=20000, class_weight='balanced')
    svc.fit(x_combined, y_combined)
    y_prob = svc._predict_proba_lr(x_test)
    y_pred = np.argmax(y_prob, axis=1)

    team_name = 'TeamFOSAI'
    submission_index = 2
    label_file = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/labels/labels.csv'
    df_labels = pd.read_csv(label_file)
    # Write out predictions to csv file (official submission format)
    pred_file_name = task + '.' + feat_type +'.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': lencoder.inverse_transform(y_pred).flatten()},
                      columns=['file_name','prediction'])
    df.to_csv(pred_file_name, index=False)

    print('Done.\n')
