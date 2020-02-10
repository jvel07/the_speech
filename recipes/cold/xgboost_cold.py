import numpy as np
import sklearn as sk
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from classifiers.cross_val import StatifiedGroupK_Fold

from classifiers.cold import cold_helper as ch

def uar_scoring(y_true, y_pred, **kwargs):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar_ = (one + two) / 2
    return uar_



my_scorer = make_scorer(uar_scoring, greater_is_better=True)


# retrieving groups for stratified group k-fold CV
groups = ch.read_utt_spk_lbl()

for g in [64]: #[2, 4, 8, 16, 32, 64, 128]:
    # Loading Train, Dev, Test, and Combined (T+D)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, X_combined, Y_combined = ch.load_data(g)

    # Normalize data
    scaler = preprocessing.Normalizer().fit(X_combined)
    X_train_norm = scaler.transform(X_combined)
    X_test_norm = scaler.transform(X_test)

    # PCA
    scaler = PCA(n_components=0.95)
    X_train_pca = scaler.fit_transform(X_train_norm)
    X_test_pca = scaler.transform(X_test_norm)

    undersampler = RandomUnderSampler(random_state=42)#sampling_strategy={0:60, 1:40})
    X_resampled, Y_resampled = undersampler.fit_resample(X_train_pca, Y_combined)

    # X_resampled, Y_resampled, indi = ch.resample_data(X_train_pca, Y_combined, r=1334599)  # resampling
    # groups = groups[indi]
    #gskf = list(StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=5).split(X_resampled, Y_resampled, groups))
    sgkf = StatifiedGroupK_Fold.StratifiedGroupKfold(n_splits=5)

    xgd = XGBClassifier(booster='gbtree', gamma=0, max_depth=3, min_child_weight=1, learning_rate=0.03, n_jobs=-1,
                        scale_pos_weight=1, reg_alpha=0.003, colsample_bytree=0.7, subsample=0.7, n_estimators=300)
    xgd.fit(X_resampled, Y_resampled)

    probs = xgd.predict_proba(X_test_pca)
    y_p = np.argmax(probs, axis=1)
    print("Confusion matrix:\n", sk.metrics.confusion_matrix(Y_test, y_p))
    one = sk.metrics.recall_score(Y_test, y_p, pos_label=0)
    two = sk.metrics.recall_score(Y_test, y_p, pos_label=1)
    uar_ = (one + two) / 2
    print(uar_)

    def tun_1(X, Y):
        param_test1 = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)
        }
        gsearch1 = GridSearchCV(estimator=XGBClassifier(booster='gblinear', learning_rate=0.1, n_estimators=200, max_depth=5,
                                                        min_child_weight=1, gamma=0, subsample=0.8,
                                                        colsample_bytree=0.8,
                                                        objective='binary:logistic', nthread=4,
                                                        seed=27),
                                param_grid=param_test1, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch1.fit(X, Y, groups=groups)
        return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

    #scores, params, best_score = tun_1(X_resampled, Y_resampled)

    def tun_2(X, Y, max_depth, min_child_weight):
        param_test3 = {
            'gamma': [i / 10.0 for i in range(0, 5)]
        }
        gsearch3 = GridSearchCV(estimator=XGBClassifier( booster='gblinear', learning_rate=0.1, n_estimators=140, max_depth=max_depth,
                                                        min_child_weight=min_child_weight, gamma=0, subsample=0.8,
                                                        colsample_bytree=0.8,
                                                        objective='binary:logistic', nthread=4, #scale_pos_weight=1,
                                                        seed=27),
                                param_grid=param_test3, scoring=my_scorer, n_jobs=-1, iid=False, cv=sgkf)
        gsearch3.fit(X, Y, groups)
        return gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


    #scores2, params2, best_score2 = tun_2(X_resampled, Y_resampled, params['max_depth'], params['min_child_weight'])