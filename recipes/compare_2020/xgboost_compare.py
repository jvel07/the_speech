import numpy as np
import sklearn as sk
from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import xgboost


import recipes.utils_recipes.utils_recipe as rutils



def uar_scoring(y_true, y_pred, **kwargs):
    uar = sk.metrics.recall_score(y_true, y_pred, labels=[1, 0], average='macro')
    return uar

my_scorer = make_scorer(uar_scoring, greater_is_better=True)

task = 'mask'
feat_type = 'fisher'

for g in [4]: #[2, 4, 8, 16, 32, 64, 128]:
    # Loading Train, Dev, Test and labels
    x_train, x_dev, x_test, y_train, y_dev = rutils.load_data_full(gauss=16, task=task, feat_type=feat_type, n_feats=23,
                                                               n_deltas=1, label_1='mask', label_0='clear')
    # X_test, Y_test, X_combined, Y_combined = ch.load_compare_data()

    x_combined = np.concatenate((x_train, x_dev))
    y_combined = np.concatenate((y_train, y_dev)).ravel()

    del x_train, x_dev

    sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    #xgd = XGBClassifier(booster='gbtree', gamma=0.0, max_depth=3, min_child_weight=3, learning_rate=0.05, n_jobs=-1,
     #            scale_pos_weight=1, reg_alpha=1e-5, reg_lambda=0.01, colsample_bytree=0.8, subsample=0.6,
      #           n_estimators=300, objective="binary:hinge", tree_method='gpu_hist', gpu_id=0)
    #xgd.fit(x_combined, y_combined)

    #probs = xgd.predict_proba(X_test)
    #y_p = np.argmax(probs, axis=1)
    #print("With {}: \nConfusion matrix:\n".format(g), sk.metrics.confusion_matrix(Y_test, y_p))
    #uar_ = uar_scoring(y_dev, y_p)
    #print(uar_)

    #

n_jobs = 1
pre = n_jobs*2

def tun_estimators_maxdep_rate(X, Y, min_child, alpha, lamb, col, sub):
    param_test1 = {
        'max_depth': range(2, 10, 1),
        'n_estimators': [100, 200, 300, 350],
        'learning_rate': [0.001, 0.01, 0.03, 0.1, 0.2, 0.003, 0.05, 0.005]
    }
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=0.0, max_depth=3, min_child_weight=min_child,
                                learning_rate=0.03, n_jobs=4,  tree_method='gpu_hist', gpu_id=0,
                                scale_pos_weight=1, reg_alpha=alpha, reg_lambda=lamb, colsample_bytree=col,
                                subsample=sub,
                                n_estimators=300, objective="binary:hinge"),
        param_grid=param_test1, scoring=my_scorer, iid=False, cv=sgkf, n_jobs=n_jobs, pre_dispatch=pre)
    gsearch1.fit(X, Y)
    return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


def tun_1(X, Y):
    param_test1 = {
        'max_depth': range(2, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=0, max_depth=3, min_child_weight=1, learning_rate=0.03,
                                n_jobs=6,
                                scale_pos_weight=1, reg_alpha=0.01, reg_lambda=0.05, colsample_bytree=0.8,
                                subsample=0.5,  tree_method='gpu_hist', gpu_id=0,
                                n_estimators=300, objective="binary:hinge"),
        param_grid=param_test1, scoring=my_scorer, iid=False, cv=sgkf, n_jobs=n_jobs, pre_dispatch=pre)
    gsearch1.fit(X, Y)
    return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


scores, params, best_score = tun_1(x_combined, y_combined)


def tun_2(X, Y, max, min):
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch3 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=0, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                n_jobs=6,
                                scale_pos_weight=1, reg_alpha=0.01, reg_lambda=0.05, colsample_bytree=0.8,
                                subsample=0.5,  tree_method='gpu_hist', gpu_id=0,
                                n_estimators=350, objective="binary:hinge"),
        param_grid=param_test3, scoring=my_scorer, iid=False, cv=sgkf, n_jobs=n_jobs, pre_dispatch=pre)
    gsearch3.fit(X, Y)
    return gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


scores2, params2, best_score2 = tun_2(x_combined, y_combined, params['max_depth'],
                                      params['min_child_weight'])


def tun_3(X, Y, max, min, gamma):
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch4 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                n_jobs=6, objective="binary:hinge",
                                scale_pos_weight=1, reg_alpha=0.001, reg_lambda=0.05, colsample_bytree=0.8,
                                subsample=0.5,  tree_method='gpu_hist', gpu_id=0,
                                n_estimators=350),
        param_grid=param_test4, scoring=my_scorer, iid=False, cv=sgkf, n_jobs=n_jobs, pre_dispatch=pre)
    gsearch4.fit(X, Y)
    return gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_


scores3, params3, best_score3 = tun_3(x_combined, y_combined, params['max_depth'],
                                      params['min_child_weight'], params2['gamma'])



def tun_4(X, Y, max, min, gamma, subsample, colsample):
    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    }
    gsearch6 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                n_jobs=6, objective="binary:hinge",
                                scale_pos_weight=1, reg_alpha=0.001, reg_lambda=0.05, colsample_bytree=colsample,
                                subsample=subsample,  tree_method='gpu_hist', gpu_id=0,
                                n_estimators=350),
        param_grid=param_test6, scoring=my_scorer, iid=False, cv=sgkf, n_jobs=n_jobs, pre_dispatch=pre)
    gsearch6.fit(X, Y)
    return gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


scores4, params4, best_score4 = tun_4(x_combined, y_combined, params['max_depth'],
                                      params['min_child_weight'], params2['gamma'], params3['subsample'],
                                      params3['colsample_bytree'])


def tun_6(X, Y, max, min, gamma, subsample, colsample, reg_alpha):
    param_test6 = {
        'reg_lambda': [1e-5, 1e-3, 1e-2, 0.1, 0.03, 0.003, 0.0003, 1, 100],
    }
    gsearch6 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                n_jobs=6, objective="binary:hinge",
                                scale_pos_weight=1, reg_alpha=reg_alpha, colsample_bytree=colsample,
                                subsample=subsample,  tree_method='gpu_hist', gpu_id=0,
                                n_estimators=350),
        param_grid=param_test6, scoring=my_scorer, iid=False, cv=sgkf, n_jobs=n_jobs, pre_dispatch=pre)
    gsearch6.fit(X, Y)
    return gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


scores6, params6, best_score6 = tun_6(x_combined, y_combined, params['max_depth'],
                                      params['min_child_weight'], params2['gamma'], params3['subsample'],
                                      params3['colsample_bytree'], params4['reg_alpha'])

scores7, params7, best_score7 = tun_estimators_maxdep_rate(x_combined, y_combined,
                                                           min_child=params['min_child_weight'],
                                                           alpha=params4['reg_alpha'], lamb=params6['reg_lambda'],
                                                           col=params3['colsample_bytree'], sub=params3['subsample'])

