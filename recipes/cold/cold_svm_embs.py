import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.metrics import make_scorer, recall_score
from sklearn.utils import shuffle

from classifiers.svm_utils.svm_fits import train_svm_gpu, train_linearsvm_cpu
from recipes.cold import cold_helper as ch


from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

work_dir = '/media/hk-data/PycharmProjects/the_speech'  # ubuntu machine
#work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'  # windows machine

com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
#com_values = [1e-5]

srand = 42

x_train, x_dev, x_test, y_train, y_dev, y_test = ch.load_embeddings()

x_combined = np.concatenate((x_train, x_dev))
y_combined = np.concatenate((y_train, y_dev))

under_sam = False
if under_sam:
    # undersampler = ClusterCentroids(random_state=0)
    undersampler = RandomUnderSampler(random_state=srand)
    x_train, y_train = undersampler.fit_resample(x_train, y_train)
    x_combined, y_combined = undersampler.fit_resample(x_combined, y_combined)

    std_flag = True
    if std_flag:
        std_scaler = preprocessing.RobustScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)

        x_combined = std_scaler.fit_transform(x_combined)
        x_test = std_scaler.transform(x_test)

x_train, y_train = shuffle(x_train, y_train, random_state=srand)
x_combined, y_combined = shuffle(x_combined, y_combined, random_state=srand)

# PCA
pca_flag = False
if pca_flag:
    pca = PCA(n_components=0.95, random_state=srand) # KernelPCA(kernel='linear', fit_inverse_transform=True, n_components=3500, eigen_solver='arpack')
    x_train = pca.fit_transform(x_train)
    x_dev = pca.transform(x_dev)
    x_test = pca.transform(x_test)
    x_combined = pca.transform(x_combined)

scores = []
for c in com_values:
    posteriors = train_linearsvm_cpu(x_train, np.ravel(y_train), x_dev, c)
    # posteriors = train_svm_gpu(x_train, np.ravel(y_train), x_dev, c)
    y_pred = np.argmax(posteriors, axis=1)
    uar = recall_score(y_dev, y_pred, average='macro')
    scores.append(uar)
    print("with", c, "-", uar)

# Train SVM model on the whole training data with optimum complexity and get predictions on test data
optimum_complexity = com_values[np.argmax(scores)]
print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))

posteriors_test = train_linearsvm_cpu(x_combined, np.ravel(y_combined), x_test, optimum_complexity)
# posteriors_test = train_svm_gpu(x_combined, np.ravel(y_combined), x_test, optimum_complexity)
test_preds = np.argmax(posteriors_test, axis=1)
uar_test = recall_score(y_test, test_preds, average='macro')
print("Test:", uar_test)


# def uar_scoring(y_true, y_pred, **kwargs):
#     one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
#     two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
#     uar_ = (one + two) / 2
#     return uar_
#
#
# my_scorer = make_scorer(uar_scoring, greater_is_better=True)

