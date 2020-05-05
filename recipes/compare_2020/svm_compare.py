import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import recall_score, make_scorer
from sklearn.preprocessing import MinMaxScaler

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

######## scoring #####
def uar_scoring(y_true, y_pred, **kwargs):
    uar = recall_score(y_true, y_pred, labels=[1, 0], average='macro')
    return uar

my_scorer = make_scorer(uar_scoring, greater_is_better=True)

##### scoring #####


task = 'mask'
feat_type = ['xvecs', 'plp']  # provide the types of features and frame-level features to use e.g.: 'fisher', 'mfcc', 'xvecs'
deli = 0
# Loading data: 'fisher' or 'xvecs'
# gaussians = [2, 4, 8, 16, 32, 64, 128]
gaussians = [128]
for gauss in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_full(
                                                                             gauss='512dimL6',
                                                                             # gauss='{}g'.format(gauss),
                                                                             task=task,
                                                                             feat_type=feat_type, n_feats=23,
                                                                             n_deltas=deli, list_labels=['mask','clear'])
    # x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_compare()

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    # Generate small test set
    # x_combined = x_combined[:20000, :]
    # y_combined = y_combined[:20000, :]
    # alternate_x_test = x_combined[-5342:, :]
    # alternate_y_test = y_combined[-5342:, :]

    # Scale data
    std_scaler = preprocessing.StandardScaler()
    # pow_scaler = preprocessing.PowerTransformer()
    # norm_scaler = preprocessing.PowerTransformer()

    # x_train = std_scaler.fit_transform(x_train)
    # x_dev = std_scaler.transform(x_dev)

    x_combined = std_scaler.fit_transform(x_combined)
    # x_test = std_scaler.transform(x_test)
    # alternate_x_test = std_scaler.transform(alternate_x_test)

    # pca = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    # pca = PCA(n_components=0.95)
    # x_train = pca.fit_transform(x_train, y_train.ravel())
    # x_dev = pca.transform(x_dev)
    # x_combined = pca.fit_transform(x_combined)
    # x_test = pca.transform(x_test)
    #
    del x_test
    print(x_train.shape)

    list_gamma = [1, 0.1, 1e-2, 1e-3, 1e-4]
    # list_gamma = [0.01]

    # list_c2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    list_c2 = [1e-2, 0.1, 1]
    # list_c2 = [1]

    # params for rbf (gridsearch)
    tuned_parameters = [
        # {'kernel': ['rbf'], 'gamma': [0.1, 1e-2, 1e-3, 1e-4, 1e-5], 'C': list_c},
        # {'kernel': ['linear'], 'C': list_c}
                        ]

    # Training data and evaluating (stratified k-fold CV)
    folds = 10
    kernel='linear'
    print("with --> ", feat_type, deli, gauss, "gauss", kernel)
    for c in list_c2:
        for g in list_gamma:  # [1367, 684531, 8754, 3215, 54, 3551, 63839845, 11538, 148111, 4310]:
            # svc = svm_fits.grid_skfcv_gpu(x_combined, y_combined.ravel(), params=tuned_parameters, metrics=[my_scorer])

            posteriors, clf = svm_fits.train_skfcv_SVM_gpu(x_combined, y_combined.ravel(), c=c, kernel=kernel, gamma=g, n_folds=folds)
            # posteriors, clf = svm_fits.train_skfcv_SVM_cpu(x_combined, y_combined.ravel(), c=c, n_folds=10)
            # posteriors, clf = svm_fits.train_skfcv_RBF_cpu(x_combined, y_combined.ravel(), c=c, n_folds=5, gamma=g)

            # posteriors = svm_fits.train_svm_gpu(x_combined, y_combined.ravel(), c=c, X_eval=alternate_x_test, kernel=kernel, gamma=g)
            # posteriors = svm_fits.train_linearsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev)
            # posteriors = svm_fits.train_rbfsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, gamma=g)

            # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_k{}_{}_fisher_{}.txt'.format(folds, c, kernel), posteriors)
            y_pred = np.argmax(posteriors, axis=1)
            print("with", c, "-", g, recall_score(y_combined, y_pred, labels=[1, 0], average='macro'))
            # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_{}_fisher32plp_{}.txt'.format(c, kernel), posteriors)


# submission
def predict(best_c):
    from thundersvm import SVC as thunder
    # svc = thunder(kernel='rbf', C=best_c, gamma=0.01, probability=True, class_weight='balanced', max_iter=100000, gpu_id=0)
    svc = svm.LinearSVC(C=best_c, verbose=0, max_iter=100000, class_weight='balanced')
    # svc = svm.SVC(kernel='rbf', gamma=0.01, probability=True, C=best_c, verbose=0, max_iter=20000, class_weight='balanced')
    svc.fit(x_combined, y_combined.ravel())
    y_prob = svc._predict_proba_lr(x_test)
    y_pred = np.argmax(y_prob, axis=1)

    team_name = 'TeamFOSAI'
    submission_index = 2
    label_file = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/labels/labels.csv'
    df_labels = pd.read_csv(label_file)
    # Write out predictions to csv file (official submission format)
    pred_file_name = task + '.' + 'x' +'.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': lencoder.inverse_transform(y_pred).flatten()},
                      columns=['file_name','prediction'])
    df.to_csv(pred_file_name, index=False)

    print('Done.\n')

