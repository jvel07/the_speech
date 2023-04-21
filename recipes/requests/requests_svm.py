import os

import numpy as np
import yaml
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn.utils import shuffle

import recipes.utils_recipes.utils_recipe as rutils
from classifiers.svm_utils import svm_fits

task = 'requests'

with open("../../conf/requests_flevel.yml") as f:
    config = yaml.safe_load(f)
cepstral_type = "mfcc"  # choose between "mfcc" or "fbank"
params = config[cepstral_type]
observation = 'Deltas{}'.format(str(params['deltas']))

# Format is: "featureType_recipeName_nMFCCs_nDeltas.mfcc"
# this is: ../data/recipe_name/train/mfcc/40_mfcc_DeltasTrue
# this is: ../data/recipe_name/train/fbank/DeltasTrue
n_feats = params['n_mfcc'] if cepstral_type == 'mfcc' else params['n_mels']
feat_info = ['fisher', n_feats, cepstral_type, observation]  # needed termporarily... to be removed

# Loading data: 'fisher' or 'ivecs's, training and evaluating it
gaussians = [2, 4, 8, 16, 32, 64, 128, 256, 512]
list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
# list_c = [0.0001]
list_nu = [0.5]
# list_c = [0.001] # pretrainedXvecs

preds_dev = 0
# xvecs-23mfcc-0del-512dim-train_dev-7234786_fbanks-test.xvecs
# Test results with 0.0001 - spe: 0.37575972110242667

# srand_list = ['389743', '564896', '2656842', '2959019', '4336987', '7234786', '9612365', '423877642', '987236753',
#               '764352323']
srand_list = ['389743']

dev_preds_dic = {}
obs = 'vad'
net = 'sre16'



for ga in gaussians:
    print("With gaussians: ", ga)
    x_train, x_dev, x_test, y_train, y_dev, file_n, enc = rutils.load_features(
                                            feat_dir='/home/user/data/features/{}/'.format(task),
                                            task=task, gauss=ga, feat_info=feat_info, list_labels=['affil', 'presta'])
    # x_combined = np.concatenate((x_train, x_dev))
    # y_combined = np.concatenate((y_train, y_dev))
    # lbl = df.values[:, -1]
    # x_t = df.values[:, 1:-1]
    # x_combined, y_combined = shuffle(x_combined, y_combined)
    # x_train, y_train = shuffle(x_train, y_train)
    # x_dev, y_dev = shuffle(x_dev, y_dev)

    std_flag = True
    if std_flag:
        std_scaler = preprocessing.StandardScaler()
        x_train = std_scaler.fit_transform(x_train)
        x_dev = std_scaler.transform(x_dev)

        # x_combined = std_scaler.fit_transform(x_combined)
        # x_test = std_scaler.transform(x_test)

    scores = []
    for c in list_c:
        posteriors, clf = svm_fits.train_linearsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, class_weight='balanced')
        # posteriors = svm_fits.train_rbfsvm_cpu(x_train, y_train.ravel(), c=c, X_eval=x_dev, gamma=g)

        # np.savetxt('/media/jose/hk-data/PycharmProjects/the_speech/data/mask/probs_mask_k{}_{}_fisher_{}.txt'.format(folds, c, kernel), posteriors)
        y_pred = np.argmax(posteriors, axis=1)
        uar = recall_score(y_dev, y_pred, average='macro')
        scores.append(uar)
        print("with", c, "-", uar)

        # util.results_to_csv(file_name='exp_results/results_{}_{}_srand_spec.csv'.format(task, feat_type[0]),
        #                     list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
        #                     list_values=[os.path.basename(file_n), ga, feat_type[2], c, uar,
        #                                  std_flag, 'DEV', srand])

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = list_c[np.argmax(scores)]
    print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))

    # Saving best dev posteriors
    # dev_preds = svm_fits.train_svr_gpu(x_train, y_train.ravel(), X_eval=x_dev, c=optimum_complexity, nu=0.5)
    # np.savetxt('preds_{}/best_preds_dev_{}_srand_{}.txt'.format(feat, optimum_complexity, srand), dev_preds)
    #
    # y_pred = svm_fits.train_svr_gpu(x_combined, y_combined.ravel(), X_eval=x_test, c=optimum_complexity, nu=list_nu[0])
    # # y_pred = sh.linear_trans_preds_test(y_train=y_train, preds_dev=preds_orig, preds_test=y_pred)
    # coef_test, p_2 = stats.spearmanr(y_test, y_pred)
    # # coef_test2 = np.corrcoef(y_test, y_pred)
    #
    # print(os.path.basename(file_n), "\nTest results with", optimum_complexity, "- spe:", coef_test)
    # print(20*'-')
    # util.results_to_csv(file_name='exp_results/results_{}_{}_srand_spec.csv'.format(task, feat_type[0]),
    #                     list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
    #                     list_values=[os.path.basename(file_n), ga, feat_type[2], optimum_complexity, coef_test,
    #                                  std_flag, 'TEST', srand])
    # # np.savetxt('preds_{}/preds_test_{}_srand_{}.txt'.format(feat, optimum_complexity, srand), y_pred)
    #
    # a = confusion_matrix(y_test, np.around(y_pred), labels=np.unique(y_train))
    # plot_confusion_matrix_2(a, np.unique(y_train), 'conf.png', cmap='Oranges', title="Spearman CC .365")