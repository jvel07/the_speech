import numpy as np
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from classifiers.cross_val import StatifiedGroupK_Fold

from recipes.cold import cold_helper as ch


def uar_scoring(y_true, y_pred, **kwargs):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar_ = (one + two) / 2
    return uar_

my_scorer = make_scorer(uar_scoring, greater_is_better=True)

# retrieving groups for stratified group k-fold CV
groups = ch.read_utt_spk_lbl()

for g in [64]:  # [2, 4, 8, 16, 32, 64, 128]:
    # Loading Train, Dev, Test, and Combined (T+D)
    X_test, Y_test, X_combined, Y_combined = ch.load_data(g)
    # X_test, Y_test, X_combined, Y_combined = ch.load_compare_data()

    # # Normalize data
    scaler = preprocessing.PowerTransformer().fit(X_combined)
    X_train_norm = scaler.transform(X_combined)
    X_test_norm = scaler.transform(X_test)

    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, Y_resampled = undersampler.fit_resample(X_combined, Y_combined)

    # X_resampled, Y_resampled, indi = ch.resample_data(X_train_pca, Y_combined, r=1334599)  # resampling
    groups = groups[undersampler.sample_indices_]
    # gskf = list(StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=5).split(X_resampled, Y_resampled, groups))
    sgkf = StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=5)

    xgd = XGBClassifier(booster='gbtree', gamma=0.4, max_depth=3, min_child_weight=5, learning_rate=0.03, n_jobs=-1,
                     scale_pos_weight=1, reg_alpha=100, reg_lambda=0.01, colsample_bytree=0.6, subsample=0.9,
                     n_estimators=350, objective="binary:hinge")
    #{'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 100} 0.03 350
    xgd.fit(X_resampled, Y_resampled)

    probs = xgd.predict_proba(X_test)
    y_p = np.argmax(probs, axis=1)
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_p))
    one = sk.metrics.recall_score(Y_test, y_p, pos_label=0)
    two = sk.metrics.recall_score(Y_test, y_p, pos_label=1)
    uar_ = (one + two) / 2
    print(uar_)


    def tun_estimators_maxdep_rate(X, Y, groups):
        param_test1 = {
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [100, 200, 300, 350, 450, 500, 550, 650],
            'learning_rate': [0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 1]
        }
        gsearch1 = GridSearchCV(
            estimator=XGBClassifier(booster='gbtree', gamma=0, max_depth=3, min_child_weight=5, learning_rate=0.03, n_jobs=-1,
                                    scale_pos_weight=1, reg_alpha=100, reg_lambda=0.01, colsample_bytree=0.6, subsample=0.9,
                                    n_estimators=350, objective="binary:hinge"),
            param_grid=param_test1, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch1.fit(X, Y, groups=groups)
        return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

    # scores7, params7, best_score7 = tun_estimators_maxdep_rate(X_resampled, Y_resampled, groups)


    def tun_1(X, Y, groups):
        param_test1 = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)
        }
        gsearch1 = GridSearchCV(
            estimator=XGBClassifier(booster='gbtree', gamma=0, max_depth=3, min_child_weight=1, learning_rate=0.03, n_jobs=-1,
                                    scale_pos_weight=1, reg_alpha=0.01, reg_lambda=0.05, colsample_bytree=0.8, subsample=0.5,
                                    n_estimators=150, objective="binary:hinge"),
            param_grid=param_test1, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch1.fit(X, Y, groups=groups)
        return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

    # scores, params, best_score = tun_1(X_resampled, Y_resampled, groups)

    def tun_2(X, Y, groups, max, min):
        param_test3 = {
            'gamma': [i / 10.0 for i in range(0, 5)]
        }
        gsearch3 = GridSearchCV(
            estimator=XGBClassifier(booster='gbtree', gamma=0, max_depth=max, min_child_weight=min, learning_rate=0.03, n_jobs=-1,
                                    scale_pos_weight=1, reg_alpha=0.01, reg_lambda=0.05, colsample_bytree=0.8, subsample=0.5,
                                    n_estimators=150, objective="binary:hinge"),
            param_grid=param_test3, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch3.fit(X, Y, groups)
        return gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


    # scores2, params2, best_score2 = tun_2(X_resampled, Y_resampled, groups, params['max_depth'],
    #                                       params['min_child_weight'])


    def tun_3(X, Y, groups, max, min, gamma):
        param_test4 = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        }
        gsearch4 = GridSearchCV(
            estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                    n_jobs=-1, objective="binary:hinge",
                                    scale_pos_weight=1, reg_alpha=0.001, reg_lambda=0.05, colsample_bytree=0.8,
                                    subsample=0.5,
                                    n_estimators=150),
            param_grid=param_test4, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch4.fit(X, Y, groups)
        return gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_


    # scores3, params3, best_score3 = tun_3(X_resampled, Y_resampled, groups, params['max_depth'],
    #                                       params['min_child_weight'], params2['gamma'])


    def tun_4(X, Y, groups, max, min, gamma, subsample, colsample):
        param_test6 = {
            'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        }
        gsearch6 = GridSearchCV(
            estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                    n_jobs=-1, objective="binary:hinge",
                                    scale_pos_weight=1, reg_alpha=0.001, reg_lambda=0.05, colsample_bytree=colsample,
                                    subsample=subsample,
                                    n_estimators=150),
            param_grid=param_test6, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch6.fit(X, Y, groups)
        return gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


    # scores4, params4, best_score4 = tun_4(X_resampled, Y_resampled, groups, params['max_depth'],
    #                                       params['min_child_weight'], params2['gamma'], params3['subsample'], params3['colsample_bytree'])


    def tun_5(X, Y, groups, max, min, gamma, subsample, colsample, reg_alpha):
        param_test6 = {
            'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
        }
        gsearch6 = GridSearchCV(
            estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                    n_jobs=-1, objective="binary:hinge",
                                    scale_pos_weight=1, reg_alpha=reg_alpha, reg_lambda=0.05, colsample_bytree=colsample,
                                    subsample=subsample,
                                    n_estimators=150),
            param_grid=param_test6, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch6.fit(X, Y, groups)
        return gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_

    # scores5, params5, best_score5 = tun_5(X_resampled, Y_resampled, groups, params['max_depth'],
    #                                       params['min_child_weight'], params2['gamma'], params3['subsample'],
    #                                       params3['colsample_bytree'], params4['reg_alpha'])

    def tun_6(X, Y, groups, max, min, gamma, subsample, colsample, reg_alpha, reg_lambda):
        param_test6 = {
            'reg_lambda': [1e-5, 1e-3, 1e-2, 0.1, 0.03, 0.003, 0.0003],
        }
        gsearch6 = GridSearchCV(
            estimator=XGBClassifier(booster='gbtree', gamma=gamma, max_depth=max, min_child_weight=min, learning_rate=0.03,
                                    n_jobs=-1, objective="binary:hinge",
                                    scale_pos_weight=1, reg_alpha=reg_alpha, reg_lambda=0.05, colsample_bytree=colsample,
                                    subsample=subsample,
                                    n_estimators=150),
            param_grid=param_test6, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch6.fit(X, Y, groups)
        return gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


    # scores6, params6, best_score6 = tun_6(X_resampled, Y_resampled, groups, params['max_depth'],
    #                                       params['min_child_weight'], params2['gamma'], params3['subsample'],
    #                                       params3['colsample_bytree'], params4['reg_alpha'], params5['reg_lambda'])
