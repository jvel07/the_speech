import sklearn as sk
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from recipes.pc_gita.utils_pcgita import load_data
from common import util
from xgboost import XGBClassifier
import numpy as np

# Loading data: 'fisher' or 'ivecs'
X, Y = load_data(16, 'monologue', 'fisher')

# Training data and evaluating (stratified k-fold CV)
sgkf = StratifiedKFold(n_splits=5, shuffle=True)
array_posteriors = np.zeros((len(Y), 2))

scores = []
for train_index, test_index in sgkf.split(X, Y):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    xgd = XGBClassifier(booster='gbtree', gamma=0.0, max_depth=3, min_child_weight=1, learning_rate=0.1, n_jobs=6,
                        scale_pos_weight=1, reg_alpha=1, reg_lambda=1, colsample_bytree=0.9, subsample=0.6,
                        n_estimators=100, objective="binary:logistic")
    # {'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 100} 0.03 350
    xgd.fit(x_train, y_train)
    scores.append(xgd.score(x_test, y_test))
print(np.mean(scores))


###### hyperparameter tunning phase #########

my_scorer = 'accuracy'


def tun_estimators_maxdep_rate(X, Y):
    param_test1 = {
        'max_depth': [3, 4, 5, 6, 7],
        'n_estimators': [100, 200, 300, 350, 450, 500, 550, 650],
        'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    }
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=0.0, max_depth=5, min_child_weight=1, learning_rate=0.01, n_jobs=6,
                                scale_pos_weight=1, reg_alpha=1, reg_lambda=1, colsample_bytree=0.9, subsample=0.6,
                                n_estimators=400, objective="binary:logistic"),
        param_grid=param_test1, scoring=my_scorer, n_jobs=6, iid=False, cv=sgkf)
    gsearch1.fit(X, Y)
    return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# scores7, params7, best_score7 = tun_estimators_maxdep_rate(X_resampled, Y_resampled, groups)


def tun_1(X, Y):
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=0, max_depth=3, min_child_weight=1, learning_rate=0.03,
                                n_jobs=6,
                                scale_pos_weight=1, reg_alpha=0.01, reg_lambda=0.05, colsample_bytree=0.8,
                                subsample=0.5,
                                n_estimators=400, objective="binary:hinge"),
        param_grid=param_test1, scoring=my_scorer, n_jobs=6, iid=False, cv=sgkf)
    gsearch1.fit(X, Y)
    return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


scores, params, best_score = tun_1(X, Y)


def tun_2(X, Y, max, min):
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch3 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=0, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                n_jobs=6,
                                scale_pos_weight=1, reg_alpha=0.01, reg_lambda=0.05, colsample_bytree=0.8,
                                subsample=0.5,
                                n_estimators=400, objective="binary:hinge"),
        param_grid=param_test3, scoring=my_scorer, n_jobs=6, iid=False, cv=sgkf)
    gsearch3.fit(X, Y)
    return gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


scores2, params2, best_score2 = tun_2(X, Y, params['max_depth'],
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
                                subsample=0.5,
                                n_estimators=400),
        param_grid=param_test4, scoring=my_scorer, n_jobs=6, iid=False, cv=sgkf)
    gsearch4.fit(X, Y)
    return gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_


scores3, params3, best_score3 = tun_3(X, Y, params['max_depth'],
                                      params['min_child_weight'], params2['gamma'])


def tun_4(X, Y, max, min, gamma, subsample, colsample):
    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    }
    gsearch6 = GridSearchCV(
        estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                n_jobs=6, objective="binary:hinge",
                                scale_pos_weight=1, reg_alpha=0.001, reg_lambda=0.05, colsample_bytree=colsample,
                                subsample=subsample,
                                n_estimators=400),
        param_grid=param_test6, scoring=my_scorer, n_jobs=6, iid=False, cv=sgkf)
    gsearch6.fit(X, Y)
    return gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


scores4, params4, best_score4 = tun_4(X, Y, params['max_depth'],
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
                                subsample=subsample,
                                n_estimators=400),
        param_grid=param_test6, scoring=my_scorer, n_jobs=6, iid=False, cv=sgkf)
    gsearch6.fit(X, Y)
    return gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


scores6, params6, best_score6 = tun_6(X, Y, params['max_depth'],
                                      params['min_child_weight'], params2['gamma'], params3['subsample'],
                                      params3['colsample_bytree'], params4['reg_alpha'])
