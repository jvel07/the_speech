import pandas as pd
from mango import Tuner
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits
import numpy as np

######## scoring #####
def uar_scoring(y_true, y_pred, **kwargs):
    uar = recall_score(y_true, y_pred, labels=[1, 0], average='macro')
    return uar

my_scorer = make_scorer(uar_scoring, greater_is_better=True)


conf_Dict = dict()
conf_Dict['batch_size'] = 3
conf_Dict['num_iteration'] = 7
conf_Dict['domain_size'] = 1000


task = 'mask'
feat_info = ['xvecs', 'mf', 0, 40]  #  provide the types of features and frame-level features to use e.g.: 'fisher', 'mfcc', 'xvecs'
# Loading data: 'fisher' or 'xvecs'
# gaussians = [2, 4, 8, 16, 32, 64, 128]
gaussians = [64]
for gauss in gaussians:
    x_train, x_dev, x_test, y_train, y_dev, y_test, file_name, lencoder = rutils.load_data_full_2(
                                                                             gauss='512dim',
                                                                             # gauss='{}g'.format(gauss),
                                                                             task=task,
                                                                             feat_info=feat_info,
                                                                             list_labels=['mask','clear'])
    # x_train, x_dev, x_test, y_train, y_dev, lencoder = rutils.load_data_compare()

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev))

    # Scale data
    std_scaler = preprocessing.StandardScaler()

    # x_train = std_scaler.fit_transform(x_train)
    # x_dev = std_scaler.transform(x_dev)

    # x_combined = std_scaler.fit_transform(x_combined)
    # x_test = std_scaler.transform(x_test)

    del x_train, x_dev  # freeing up space

    # tuner = svm_fits.train_mango_skcv(X=x_combined, Y=y_combined, n_splits=10)

    param_space = {
        # 'kernel': ['rbf', 'linear'],
        # 'gamma': uniform(0.1, 4),  # 0.1 to 4.1
        'C': [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]  # loguniform(-6, 6)  # 10^-7 to 10
    }


    # @scheduler.serial
    def objectiveSVM(args_list):
        results = []
        for hyper_par in args_list:
            svc = svm.LinearSVC(**hyper_par, max_iter=100000,
                                class_weight='balanced')
            # svc = thunder(**hyper_par, max_iter=100000,
            #               class_weight='balanced')
            result = cross_val_score(svc, x_combined, y_combined, scoring=my_scorer, n_jobs=-1, cv=10).mean()
            results.append(result)
        return results

    tuner = Tuner(param_dict=param_space, objective=objectiveSVM)
    results = tuner.maximize()

    print('best hyper parameters:', results['best_params'])
    print('best objective:', results['best_objective'])
