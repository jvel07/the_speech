from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


def pca_trainer_transformer(data_to_fit, data_to_transform, n_components):
    pca = PCA(n_components=n_components)
    pca.fit_transform(data_to_fit)
    list_transformed_data=[]
    for i in data_to_transform:
        list_transformed_data.append(pca.transform(i))
    return list_transformed_data


def pca_trainer(file, n_components):
    data = np.load(file, allow_pickle=True)
    print(np.vstack(data).shape)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    print("PCA trained using:", file, "\nWith variance/components:", n_components)
    return pca


def pca_transformer(obj_pca_trainer, file):
    data = np.load(file, allow_pickle=True)
    data_transformed = obj_pca_trainer.transform(data)
    print("Data transformed successfuly from file:", file , "\n Final shape:", data_transformed.shape)
    return data_transformed


def lda_trainer(file, labels):
    data = np.load(file, allow_pickle=True)
    lda = LinearDiscriminantAnalysis()
    lda.fit(data, labels)
    return lda


def lda_transformer(obj_lda_trainer, data):
    data_transformed = obj_lda_trainer.transform(data)
    return data_transformed

