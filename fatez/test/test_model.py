import os
import torch
from torch.utils.data import DataLoader
import fatez.test as test
import fatez.lib as lib
import fatez.model as model
from pkg_resources import resource_filename

# Ignoring warnings because of using LazyLinear
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # Create the cache directory if not present
    if not os.path.exists('../data/ignore'):
        os.makedirs('../data/ignore')
    faker = test.Faker()
    model.Save(faker.test_gat(), '../data/ignore/gat.model')
    model.Save(faker.test_full_model(), '../data/ignore/bert_encoder.model')


    from sklearn import cluster
    eps = 0.5
    # Set model
    dbscan = cluster.DBSCAN(eps = eps)
    kmeans = cluster.KMeans(n_clusters = 2)
    # Flatten feature matrices for clustering
    data = [
        torch.reshape(x[0][0], (-1,)).tolist() for x, y in DataLoader(
            faker.make_data_loader().dataset, batch_size = 1
        )
    ]

    dbscan.fit(data)
    kmeans.fit(data)
    print(dbscan.labels_.astype(int))
    print(kmeans.labels_.astype(int))
