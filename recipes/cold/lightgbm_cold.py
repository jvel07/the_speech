import numpy as np
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from classifiers.cross_val import StatifiedGroupK_Fold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from recipes.cold import cold_helper as ch
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def uar_scoring(y_true, y_pred):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar = (one + two) / 2
    return 'UAR', uar, True

def uar_scoring2(y_true, y_pred):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar = (one + two) / 2
    return uar

my_scorer = make_scorer(uar_scoring2, greater_is_better=True)


def lgb_uar(y_pred, data):
    y_true = data.get_label()
    return 'uar', roc_auc_score(y_true, y_pred), True


# retrieving groups for stratified group k-fold CV
groups_orig = ch.read_utt_spk_lbl()

for g in [64]:  # [2, 4, 8, 16, 32, 64, 128]:
    # Loading Train, Dev, Test, and Combined (T+D)
    X_test, Y_test, X_combined, Y_combined = ch.load_data(g)
    # X_test, Y_test, X_combined, Y_combined = ch.load_compare_data()

    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, Y_resampled = undersampler.fit_resample(X_combined, Y_combined)

    # X_resampled, Y_resampled, indi = ch.resample_data(X_train_pca, Y_combined, r=1334599)  # resampling
    groups = groups_orig[undersampler.sample_indices_]
    # gskf = list(StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=5).split(X_resampled, Y_resampled, groups))
    sgkf = StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=5)

    params = {'colsample_bytree': 0.952164731370897, 'min_child_samples': 111, 'min_child_weight': 0.01, 'num_leaves': 38, 'reg_alpha': 0, 'reg_lambda': 0.1, 'subsample': 0.3029313662262354}

    # GRID
    param_test = {'num_leaves': sp_randint(6, 50),
                  'min_child_samples': sp_randint(100, 500),
                  'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                  'subsample': sp_uniform(loc=0.2, scale=0.8),
                  'colsample_bytree': sp_uniform(loc=0.4, scale=0.7),
                  'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                  'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                  #'feval': 'auc',
                  }

    # This parameter defines the number of HP points to be tested
    n_HP_points_to_test = 100

    # n_estimators is set to a "large value". The actual number of trees build will depend on early
    # stopping and 5000 define only the absolute maximum
    clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metrics='none', n_jobs=-1, n_estimators=5000, class_weight='balanced')
    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test,
        n_iter=n_HP_points_to_test,
        scoring=my_scorer,
        cv=5,
        refit=True,
        random_state=314,
        verbose=True)

    gs.fit(X_combined, Y_combined, groups_orig)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

    # train_data = lgb.Dataset(X_resampled, Y_resampled)
    # validation_data = lgb.Dataset(X_resampled, reference=train_data)
    #
    # model = lgb.train(parameters, train_data, #valid_sets=[validation_data], early_stopping_rounds=5,
    #                   num_boost_round=70, feval=roc_auc_score)
    #
    def predict(mdl):
        y_pred = mdl.predict(X_test, num_iteration=mdl.best_iteration)
        print("uar:", uar_scoring(Y_test, y_pred.round()))

