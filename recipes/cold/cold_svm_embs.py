import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.metrics import make_scorer, recall_score

from classifiers.svm_utils.svm_fits import train_svm_gpu
from recipes.cold import cold_helper as ch

work_dir = '/media/hk-data/PycharmProjects/the_speech'  # ubuntu machine
#work_dir = 'C:/Users/Win10/PycharmProjects/the_speech'  # windows machine

com_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
#com_values = [1e-5]

# retrieving groups for stratified group k-fold CV
# groups = ch.read_utt_spk_lbl()

# iterating over the gaussians
# print("CV Process (gaussians):", g)
# Loading Test, and Combined (Train+Dev)
x_train, x_dev, x_test, y_train, y_dev, y_test = ch.load_embeddings()
#X_test, Y_test, X_combined, Y_combined = ch.load_compare_data()
# X, Y = shuffle(X_train, Y_train, random_state=0)

x_combined = np.concatenate((x_train, x_dev))
y_combined = np.concatenate((y_train, y_dev))

std_flag = False
if std_flag:
    std_scaler = preprocessing.StandardScaler()
    x_train = std_scaler.fit_transform(x_train)
    x_dev = std_scaler.transform(x_dev)

    x_combined = std_scaler.fit_transform(x_combined)
    x_test = std_scaler.transform(x_test)

# PCA
#pca = PCA(n_components=0.99) # KernelPCA(kernel='linear', fit_inverse_transform=True, n_components=3500, eigen_solver='arpack')
#X_train_norm = pca.fit_transform(X_train_norm)
#X_test_norm = pca.transform(X_test_norm)

scores = []
for c in com_values:
    # groups = pre_groups[indi]
    posteriors = train_svm_gpu(x_train, y_train, x_dev, c)
    print("With:", c, "->")

    y_pred = np.argmax(posteriors, axis=1)
    uar = recall_score(y_dev, y_pred, average='macro')
    scores.append(uar)
    print("with", c, "-", uar)

# Train SVM model on the whole training data with optimum complexity and get predictions on test data
optimum_complexity = com_values[np.argmax(scores)]
print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))

posteriors_test = train_svm_gpu(x_combined, y_combined, x_test, optimum_complexity)
test_preds = np.argmax(posteriors_test, axis=1)
uar_test = recall_score(y_test, test_preds, average='macro')
print("Test:", uar_test)


def uar_scoring(y_true, y_pred, **kwargs):
    one = sk.metrics.recall_score(y_true, y_pred, pos_label=0)
    two = sk.metrics.recall_score(y_true, y_pred, pos_label=1)
    uar_ = (one + two) / 2
    return uar_


my_scorer = make_scorer(uar_scoring, greater_is_better=True)

