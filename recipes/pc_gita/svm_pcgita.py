from sklearn import preprocessing

from classifiers.svm_utils import svm_fits
from recipes.pc_gita.utils_pcgita import load_data

# Loading data: 'fisher' or 'ivecs'
x_train, y_train = load_data(1024, 'monologue', 'fisher')

# Scale data
# x_train = preprocessing.PowerTransformer().fit_transform(x_train)
# x_train = preprocessing.Normalizer().fit_transform(x_train)
# x_train = preprocessing.RobustScaler().fit_transform(x_train)
x_train = preprocessing.StandardScaler().fit_transform(x_train)

# Training data and evaluating (stratified k-fold CV)
for c in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]:
    score = svm_fits.train_simple_skfcv_pca(x_train, y_train, n_folds=5, c=c, metric_type='f1')
    print("with c", c, score)