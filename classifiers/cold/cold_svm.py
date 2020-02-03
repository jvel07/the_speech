import numpy as np
import sklearn as sk

from classifiers.cold import cold_helper as ch

# Loading Train, Dev, Test and Combined (T+D)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test, X_combined, Y_combined = ch.load_data()

# Resampling, Strat k-fold Cross-val, SVM
com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
for c in com_values:
    for number in [137, 42, 15986, 4242, 7117, 15, 1, 923, 25, 9656]:
        X_resampled, Y_resampled = ch.resample_data(X_combined, Y_combined, r=number) # resampling
        ch.train_model_stk_cv(X_resampled, Y_resampled, 7, c)


