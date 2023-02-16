import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pkg_resources import resource_filename
from sklearn import cluster
import fatez.test as test
import fatez.model as model


# Ignoring warnings because of using LazyLinear
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # Create the cache directory if not present
    if not os.path.exists('../data/ignore'): os.makedirs('../data/ignore')

    # Init testing model first
    faker = test.Faker()
    testM = faker.test_full_model()
    # model.Save(faker.test_gat(), '../data/ignore/gat.model')
    # model.Save(testM, '../data/ignore/trainer.model')

    # Prepare flatten data for clustering
    dataset = faker.make_data_loader().dataset
    for x, labels in DataLoader(dataset, batch_size = len(dataset)):
        all_fea_mat = x[0]
        all_adj_mat = x[1]

    origin = [torch.reshape(ele, (-1,)).tolist() for ele in all_fea_mat]
    # The encoded representaions made by GAT -> BERT encoder
    encode = [
        torch.reshape(ele, (-1,)).tolist() for ele in testM.get_encoder_output(
            all_fea_mat, all_adj_mat
        )
    ]


    # Set clustering models
    eps = 0.5
    n_clusters = len(np.unique(labels))
    dbscan = cluster.DBSCAN(eps = eps)
    kmeans = cluster.KMeans(n_clusters = n_clusters)



    # Fit models with original data
    dbscan.fit(origin)
    kmeans.fit(origin)
    # Get labels
    print(dbscan.labels_.astype(int))
    print(kmeans.labels_.astype(int))

    # Re-init models and fit with encoded representaions
    dbscan = cluster.DBSCAN(eps = eps)
    kmeans = cluster.KMeans(n_clusters = n_clusters)
    dbscan.fit(encode)
    kmeans.fit(encode)
    # Get labels
    print(dbscan.labels_.astype(int))
    print(kmeans.labels_.astype(int))
