import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import xgboost as xgb


import recipes.utils_recipes.utils_recipe as rutils


def uar_scoring(y_true, y_pred, **kwargs):
    uar = sk.metrics.recall_score(y_true, y_pred, labels=[1, 0], average='macro')
    return uar


def eval_error(preds, dtrain):
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    labels = dtrain.get_label()
    return 'uar', recall_score(labels, preds, labels=[1, 0], average='macro')

my_scorer = make_scorer(eval_error, greater_is_better=True)


task = 'mask'
feat_type = 'fisher'

for g in [32]:#, 4, 8, 16, 32, 64, 128]:
    # Loading Train, Dev, Test and labels
    X_train, X_dev, X_test, Y_train, Y_dev, le = rutils.load_data_full(gauss=g, task=task, feat_type=feat_type, n_feats=23,
                                                               n_deltas=1, label_1='mask', label_0='clear')
    # X_test, Y_test, X_combined, Y_combined = ch.load_compare_data()

    # x_combined = np.concatenate((X_train, X_dev))
    # y_combined = np.concatenate((Y_train, Y_dev)).ravel()

    params = {'booster':'gbtree', 'gamma':0.0, 'max_depth':8, 'min_child_weight': 3, 'learning_rate': 0.05,
               'n_jobs' : 6, 'scale_pos_weight' : 1, 'reg_alpha': 0.1, 'reg_lambda': 1, 'colsample_bytree': 0.9,
                'subsample' : 0.8, 'objective':'binary:logistic', #'eval_metric': 'auc',
               'tree_method':'gpu_hist', 'gpu_id':0
              }

    sgkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # total_probs = np.zeros((len(y_combined),))
    # for number in [34647]:#, 578, 2154, 4242, 54]:
    #     # for train_index, test_index in sgkf.split(x_combined, y_combined):
    #     #
    #     #     x_train, x_test, y_train, y_test = \
    #     #         x_combined[train_index], x_combined[test_index], y_combined[train_index], y_combined[test_index]
    #
    #     dtrain = xgb.DMatrix(X_train, label=Y_train)
    #     dtest = xgb.DMatrix(X_dev, label=Y_dev)
    #
    #     model = xgb.train(params, dtrain, num_boost_round=15, evals=[(dtest, "Test")], feval=eval_error,
    #                       early_stopping_rounds=8, maximize=True)
    #
    #     probs = model.predict(dtest)
    #     # total_probs[test_index] = probs
    #     total_probs = probs
    #     total_probs[total_probs > 0.5] = 1
    #     total_probs[total_probs <= 0.5] = 0
    # print("With {}: \nConfusion matrix:\n".format(g), sk.metrics.confusion_matrix(Y_dev, total_probs))
    # uar_ = uar_scoring(Y_dev, total_probs)
    # print(uar_)


# submission
def pred():
    xgb = XGBClassifier(booster='gbtree', gamma=0.0, max_depth=8, min_child_weight=3, learning_rate=0.03, n_jobs=-1,
                        scale_pos_weight=1, reg_alpha=0.1, reg_lambda=1, colsample_bytree=0.9, subsample=0.8,
                        n_estimators=370, objective="binary:hinge", tree_method='gpu_hist', gpu_id=0,
                        random_state=5477113)
    xgb.fit(x_combined, y_combined)
    y_pred = xgb.predict(X_test)
    team_name = 'TeamFOSAI'
    submission_index = 2
    label_file = '/media/jose/hk-data/PycharmProjects/the_speech/data/mask/labels/labels.csv'
    df_labels = pd.read_csv(label_file)
    # Write out predictions to csv file (official submission format)
    pred_file_name = task + '.' + feat_type +'.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': le.inverse_transform(y_pred).flatten()},
                      columns=['file_name','prediction'])
    df.to_csv(pred_file_name, index=False)

    print('Done.\n')



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
        'min_child_weight': range(1, 6, 2),
        'gamma': [i / 10.0 for i in range(0, 5)]
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


scores, params, best_score = tun_1(X_train, Y_train)
print(best_score, params)


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


scores3, params3, best_score3 = tun_3(X_train, Y_train, params['max_depth'],
                                      params['min_child_weight'], params['gamma'])

print(best_score3, params3)


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


scores4, params4, best_score4 = tun_4(X_train, Y_train, params['max_depth'],
                                      params['min_child_weight'], params['gamma'], params3['subsample'],
                                      params3['colsample_bytree'])

print(best_score4, params4)



def tun_6(X, Y, max, min, gamma, subsample, colsample, reg_alpha):
    param_test6 = {
        'reg_lambda': [1e-4, 1e-3, 1e-2, 0.1, 0.03, 0.003, 1, 100],
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


scores6, params6, best_score6 = tun_6(X_train, Y_train, params['max_depth'],
                                      params['min_child_weight'], params['gamma'], params3['subsample'],
                                      params3['colsample_bytree'], params4['reg_alpha'])

print(best_score6, params6)

scores7, params7, best_score7 = tun_estimators_maxdep_rate(X_train, Y_train,
                                                           min_child=params['min_child_weight'],
                                                           alpha=params4['reg_alpha'], lamb=params6['reg_lambda'],
                                                           col=params3['colsample_bytree'], sub=params3['subsample'])

print(best_score7, params7)