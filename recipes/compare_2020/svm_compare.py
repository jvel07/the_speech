import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

task = 'mask'
feat_type = ['fisher', 'mf']  # provide the types of features and frame-level features to use e.g.: 'fisher', 'mf'

# Loading data: 'fisher' or 'xvecs'
gaussians = [2, 4, 8, 16, 32, 64, 128, 256]
# gaussians = [32]
for gauss in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_full(gauss='{}g'.format(gauss),
                                                                             task=task,
                                                                             feat_type=feat_type, n_feats=40,
                                                                             n_deltas=1, list_labels=['mask','clear'])
    # x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_compare()

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

    # pca = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto') #PCA(n_components=0.95)
    # x_train = pca.fit_transform(x_train, y_train.ravel())
    # x_dev = pca.transform(x_dev)
    # x_combined = pca.fit_transform(x_combined)
    # x_test = pca.transform(x_test)
    #
    del x_test
    print(x_train.shape)

    # list_gamma = [0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    list_gamma = [0.1]

    list_c2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    list_c = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    # list_c = [1e-7]

    # params for rbf
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': list_c2},
                        # {'kernel': ['linear'], 'C': list_c2}
                        ]

    # Training data and evaluating (stratified k-fold CV)
    print("with --> ", feat_type, gauss)
    for c in list_c2:
        for g in list_gamma:  # [1367, 684531, 8754, 3215, 54, 3551, 63839845, 11538, 148111, 4310]:
            # posteriors, svc = svm_fits.grid_skfcv_gpu(x_combined, y_combined.ravel(), params=tuned_parameters, metrics=['accuracy'])
            # posteriors, clf = svm_fits.train_skfcv_SVM_gpu(x_train, y_train.ravel(), c=c, kernel='linear', gamma=0.001, n_folds=5)
            # posteriors, clf = svm_fits.train_skfcv_SVM_cpu(x_train, y_train.ravel(), c=c, n_folds=5)
            posteriors = svm_fits.train_svm_gpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, kernel='linear', gamma=g)
            # posteriors = svm_fits.train_linearsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev)
            # posteriors = svm_fits.train_rbfsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, gamma=g)
        # print("with c", c, "-->", score["uar"])
        # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_dev_{}_fisher.txt'.format(c), posteriors)
        # print("with c", c, score)
        # Metrics
            y_pred = np.argmax(posteriors, axis=1)
            print("with", c, "-", g, recall_score(y_dev, y_pred, labels=[1,0], average='macro'))


# submission
def predict(best_c):
    svc = svm.LinearSVC(C=best_c, verbose=0, max_iter=20000, class_weight='balanced')
    # svc = svm.SVC(kernel='rbf', gamma=0.01, probability=True, C=best_c, verbose=0, max_iter=20000, class_weight='balanced')
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

