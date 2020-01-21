from sklearn import preprocessing
import numpy as np
import pickle


# Opening MFCCs from file
def read_mfcc(file_name):
    with open(file_name, 'rb') as f:
        mfccs = pickle.load(f)
        print("MFCCs loaded from:", file_name)
    return mfccs


# Saving MFCCs to file
def save_mfccs(file, mfccs):
    with open(file, 'wb') as fp:
        pickle.dump(mfccs, fp)
    print("MFCCs saved to:", file)


if __name__ == '__main__':

    set_ = 'ubms'
    obs = 'fulltmt'
    num_gauss = 16
    proj = 'cold'

    # Output files of new scaled MFCCs
    mfccs_file_ivec_train_scl = '../data/hc/{}/mfccs_{}_train_20_scl'.format(proj, proj)
    mfccs_file_ivec_dev_scl = '../data/hc/{}/mfccs_{}_dev_20_scl'.format(proj, proj)
    mfccs_file_ivec_test_scl = '../data/hc/{}/mfccs_{}_test_20_scl'.format(proj, proj)
    mfccs_file_ubm_scl = '../data/hc/{}/mfccs_{}_ubm_20_{}_scl'.format(proj, proj, obs)

    # Standarize-scale the MFCCs and pickle-dump them
    if set_ == 'ubm':
        # Load MFCCs to be used for training the UBM
        mfccs_file_ubm = '../data/hc/{}/mfccs_cold_ubm_20_{}'.format(proj, obs)
        mfccs_ubm = np.vstack(read_mfcc(mfccs_file_ubm))

        # Scale UBM MFCCs
        scaler = preprocessing.StandardScaler().fit(mfccs_ubm)
        mfccs_ubm = scaler.transform(mfccs_ubm)

        save_mfccs(mfccs_file_ubm_scl, mfccs_ubm)  # Save MFCCs as array
        print("UBM MFCCs scaled and saved successfully to:", mfccs_file_ubm_scl)
    else:
        # MFCCs files to be used in i-vecs extraction
        mfccs_file_ivec_train = '../data/hc/{}/mfccs_{}_train_20_'.format(proj, proj)
        mfccs_file_ivec_dev = '../data/hc/{}/mfccs_{}_dev_20_'.format(proj, proj)
        mfccs_file_ivec_test = '../data/hc/{}/mfccs_{}_test_20_'.format(proj, proj)

        # Scale i-vecs MFCCs and save them
        mfccs_ivecs_train = np.vstack(read_mfcc(mfccs_file_ivec_train))  # Load MFCCs as array for fitting the scaler
        scaler = preprocessing.StandardScaler().fit(mfccs_ivecs_train)  # Fit the scaler on the training MFCCs array
        del mfccs_ivecs_train  # Free mem
        list_mfccs_ivecs_train = read_mfcc(mfccs_file_ivec_train)  # Load MFCCs as list (format needed for feeding the i-vec extractor)
        list_mfccs_ivecs_train_scl = []
        for vector in list_mfccs_ivecs_train:
            scaled_vector = scaler.transform(vector)  # Scale train
            list_mfccs_ivecs_train_scl.append(scaled_vector)
        save_mfccs(mfccs_file_ivec_train_scl, list_mfccs_ivecs_train_scl)  # Save MFCCs as list
        del list_mfccs_ivecs_train_scl, list_mfccs_ivecs_train  # Free mem

        list_mfccs_ivecs_dev = read_mfcc(mfccs_file_ivec_dev)  # Load MFCCs as list
        list_mfccs_ivecs_dev_scl = []
        for vector in list_mfccs_ivecs_dev:
           scaled_vector = scaler.transform(vector)  # Scale dev with the already fitted scaler
           list_mfccs_ivecs_dev_scl.append(scaled_vector)
        save_mfccs(mfccs_file_ivec_dev_scl, list_mfccs_ivecs_dev_scl)  # Save
        del list_mfccs_ivecs_dev, list_mfccs_ivecs_dev_scl  # Free mem

        list_mfccs_ivecs_test = read_mfcc(mfccs_file_ivec_test)  # Load MFCCs
        list_mfccs_ivecs_test_scl = []
        for vector in list_mfccs_ivecs_test:
            scaled_vector = scaler.transform(vector)  # Scale dev with the already fitted scaler
            list_mfccs_ivecs_test_scl.append(scaled_vector)
        save_mfccs(mfccs_file_ivec_test_scl, list_mfccs_ivecs_test_scl)  # Save
        del list_mfccs_ivecs_test, list_mfccs_ivecs_test_scl  # Free mem

        print("i-vectors MFCCs scaled and saved successfully to above mentioned directories!")
