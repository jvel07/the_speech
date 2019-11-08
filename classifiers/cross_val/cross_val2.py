import csv
import os
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

from common import util, data_proc_tools as tools


# Loading data (if k=0, loads from txt; loads from pickle otherwise)
def load_data(_x, _y, load_mode):
    x = None
    if load_mode == 'txt':
        x = np.loadtxt(_x)
    elif load_mode == 'pickle':
        x = util.read_pickle(_x)

    y = np.load(_y)
    return x, y


# Encoding labels to numbers
def encode_labels_alz(_y):
    le = preprocessing.LabelEncoder()
    le.fit(["k", "e", "a"])
    y = le.transform(_y)
    y = y.reshape(-1, 1)
    return y


# Group wavs every n number
# (For Alzheimer's) Each speaker has 3 samples, group every 3 samples
# Returns list of lists, 3 arrays within each list.
def group_speakers_wavs(_x, n):
    print("Speakers' wavs grouped into {} from {}".format(n, _x[0].shape))
    return util.group_wavs_speakers_12(_x)


# Concatenate speakers wavs (3) in one single array
# Returns list of concatenated arrays
def join_speakers_wavs(list_group_wavs):
    x = []
    for element in list_group_wavs:
        for a, b, c in zip_longest(*[iter(element)] * 3):  # iterate over the sublist of the list
            array = np.concatenate((a, b, c))  # concatenating arrays (every 3)
            x.append(array)
    print("Speakers' wavs concatenated!")
    return np.vstack(x)

def join_speakers_wavs_2(list_group_wavs):
    x = []
    for element in list_group_wavs:
        for x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 in zip_longest(*[iter(element)] * 12):  # iterate over the sublist of the list
            array = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12))  # concatenating arrays (every 3)
            x.append(array)
    print("Speakers' wavs concatenated!")
    return np.vstack(x)


# Enter the estimator (svm, rbf)
def grid_search(_x_train, _y_train):
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 20, 30, 100]},
                  {'kernel': ['linear'], 'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 20, 30, 100]}
                  ]

    gd_sr = GridSearchCV(estimator=SVC(),
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(_x_train, np.ravel(_y_train))
    best_c = gd_sr.best_params_
    print(gd_sr.best_params_)
    print(gd_sr.best_estimator_)
    print("Best complexity value:", best_c['C'])
    return best_c['C']


def train_model_grid_search_cv(_x_train, _y_train, n_splits):
    predicciones = []
    ground_truths = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for train, test in skf.split(_x_train, _y_train):
        best_c = grid_search(_x_train[train], np.ravel(_y_train[train]))
        svc = svm.LinearSVC(C=best_c, verbose=0, max_iter=965000)  # class_weight='balanced',
        svc.fit(_x_train[train], np.ravel(_y_train[train]))
        y_pred = svc.predict(_x_train[test])
        predicciones.append(y_pred)
        ground_truths.append(_y_train[test])

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))

    return predicciones, ground_truths


# train SVM model with stratified cross-validation
def train_model_cv(_x_train, _y_train, n_splits, _c):
    predicciones = []
    ground_truths = []

    skf = StratifiedKFold(n_splits=n_splits)
    for train, test in skf.split(_x_train, _y_train):
        svc = svm.LinearSVC(C=_c, verbose=0, max_iter=965000)  # class_weight='balanced',
        svc.fit(_x_train[train], np.ravel(_y_train[train]))
        y_pred = svc.predict(_x_train[test])
        predicciones.append(y_pred)
        ground_truths.append(_y_train[test])
        # pp['final-average'] = predicciones+ground_truths

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))

    return predicciones, ground_truths


def best_pca_components(x):
    var = tools.get_var_ratio_pca(x)
    return tools.sel_pca_comp(var, 0.95)


def train_model_cv_PCA(_x_train, _y_train, n_splits):
    predicciones = []
    ground_truths = []
    #svc = svm.LinearSVC(C=_c, verbose=1, max_iter=965000) #class_weight='balanced',
    skf = StratifiedKFold(n_splits=n_splits)
    for train, test in skf.split(_x_train, _y_train):
        comp = best_pca_components(_x_train[train])
        pca_train = PCA(n_components=comp, svd_solver='auto', whiten=False)
        pca_train.fit(_x_train[train])
        x_train_reduced = pca_train.transform(_x_train[train])
        #pca_test = PCA(n_components=comp, svd_solver='full', whiten=False)
        #pca_test.fit(_x_train[test])
        x_test_reduced = pca_train.transform(_x_train[test])

        best_c = grid_search(x_train_reduced, _y_train[train])
        svc = svm.LinearSVC(C=best_c, verbose=1, max_iter=965000)
        svc.fit(x_train_reduced, np.ravel(_y_train[train]))

        y_pred = svc.predict(x_test_reduced)
        predicciones.append(y_pred)
        ground_truths.append(_y_train[test])
        # pp['final-average'] = predicciones+ground_truths

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))

    return predicciones, ground_truths


def train_model_cv_LDA(_x_train, _y_train, n_splits):
    predicciones = []
    ground_truths = []
    #svc = svm.LinearSVC(C=_c, verbose=1, max_iter=965000) #class_weight='balanced',
    skf = StratifiedKFold(n_splits=n_splits)
    for train, test in skf.split(_x_train, _y_train):
        lda = LDA(n_components=8)
        lda2 = LDA(n_components=8)
        lda.fit(_x_train[train], _y_train[train])
        x_train_reduced = lda.transform(_x_train[train])
        lda2.fit(_x_train[test], _y_train[test])
        test_x_reduced = lda.transform(_x_train[test])
        y_encoded = encode_labels_alz(y)

        best_c = grid_search(x_train_reduced, _y_train[train])
        svc = svm.LinearSVC(C=best_c, verbose=0, max_iter=965000)
        svc.fit(x_train_reduced, np.ravel(y_encoded[train]))
        #svc.fit(_x_train[train], np.ravel(_y_train[train]))

        y_pred = svc.predict(test_x_reduced)
        predicciones.append(y_pred)
        ground_truths.append(y_encoded[test])
        # pp['final-average'] = predicciones+ground_truths

    predicciones = np.ravel(np.vstack(predicciones))
    ground_truths = np.ravel(np.vstack(ground_truths))

    return predicciones, ground_truths


def metrics(ground_truths, preds):
    accuracy = sk.metrics.accuracy_score(ground_truths, preds)
#    f1 = sk.metrics.f1_score(ground_truths, preds)
 #   precision = sk.metrics.precision_score(ground_truths, preds)
  #  recall = sk.metrics.recall_score(ground_truths, preds)
    print('acc:', accuracy)
    return accuracy


def print_conf_matrix(ground_truths, preds):
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(ground_truths, preds))


# Writing results to a csv
def results_to_csv(file_name, g, feat_type, num_filters, deltas, vad, pca, acc):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['Gaussians', 'Feature', 'N_filters', 'VAD', 'PCA', 'Accuracy'])
            file_writer.writerow([g, feat_type, num_filters, deltas, vad, pca, acc])
            print("File " + file_name + " created!")
    else:
        with open(file_name, 'a') as csv_file:
            file_writer = csv.writer(csv_file)
            file_writer.writerow([g, feat_type, num_filters, deltas, vad, pca, acc])
            print("File " + file_name + " updated!")


def plot_pca_variance():
    pca_var = PCA().fit(x_train)
    plt.figure()
    plt.plot(np.cumsum(pca_var.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Dataset Explained Variance')
    plt.show()


if __name__ == '__main__':

    pca_ = 1
    list_num_gauss = [2, 4, 8, 16, 32, 64, 128]
    #obs = 'fbanks_40'
    feat_type = 'fbanks'
    n_filters = '40_imp'
    deltas = ''
    vad = ''
    pca_comp = 20

    for num_gauss in list_num_gauss:
        file_x = 'C:/Users/Win10/PycharmProjects/the_speech/data/fisher_vecs/fisher-{}-{}-{}-{}-{}'.format(num_gauss, feat_type,
                                                                                               n_filters, deltas, vad)
        file_y = 'labels_75.npy'

        Y = np.load('labels_75.npy')
        x_train_data, y = load_data(file_x, file_y, load_mode='txt')
        y_train = encode_labels_alz(y)
        x_train_grouped = group_speakers_wavs(x_train_data, 12)
        x_train = join_speakers_wavs_2(x_train_grouped)
        if pca_ == 1:
            scl = PowerTransformer()
            scl.fit(x_train)
            x_train = scl.transform(x_train)
            x_train = tools.normalize_data(x_train)
            c = grid_search(x_train, y_train)
            pred, ground = train_model_cv(x_train, y, 5, c)
            acc = metrics(ground, pred)
           # print_conf_matrix(ground, pred)
            results_to_csv('C:/Users/Win10/PycharmProjects/the_speech/data/results_dem.csv',
                           str(num_gauss), feat_type, n_filters, deltas, str(vad), str(pca_comp), str(acc))
        else:
            #x_train = tools.standardize_data(x_train)
            c = grid_search(x_train, y_train)
            train_model_cv(x_train, y_train, 5, c)
            pred, ground = train_model_cv(x_train, y_train, 5, c)
            metrics(ground, pred)
            acc = metrics(ground, pred)
            #results_to_csv('results_dem.csv', str(num_gauss), feat_type, str(n_feats), deltas, str(vad), str(pca_comp), str(acc))
            print(file_x)





