#!/usr/bin/env python3
"""
This file contains basic objects for storing data.

author: jy
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data



class FateZ_Dataset(Dataset):
    """
	Basic Dataset object for DataLoader
	"""
    def __init__(self, samples, labels):
        assert len(samples) == len(labels)
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]



class Adj_Mat(object):
    """
    docstring for Adjacent Matix.
    """
    def __init__(self,
        dense_mat:torch.Tensor = None,
        indices:torch.Tensor = None,
        values:torch.Tensor = None,
        size = None,
        sparse_dim:int = 2,
        sparse_mat:torch.Tensor = None,
        **kwargs
        ):
        super(Adj_Mat, self).__init__()
        if sparse_mat is None and dense_mat is None:
            self.sparse = torch.sparse_coo_tensor(indices, values, size)
        elif sparse_mat is None and dense_mat is not None:
            self.sparse = dense_mat.to_sparse(sparse_dim = sparse_dim)
        elif sparse_mat is not None:
            self.sparse = sparse_mat
        self.sparse = self.sparse.coalesce()

    def get_index_value(self,):
        indices = self.sparse.indices()
        values = self.sparse.values()
        # Reshape value matrix if edges only have single features
        if len(values.shape) == 1:
            values = torch.reshape(values, (len(values), 1))
        return indices, values

    def to_dense(self):
        # If each edge only has one feature
        if len(self.sparse.shape) > 2 and self.sparse.shape[-1] == 1:
            return self.sparse.to_dense().reshape(self.sparse.shape[:-1])
        else:
            return self.sparse.to_dense()

# if __name__ == '__main__':
#     t = torch.tensor([[0., -1, 0], [2., 0., 1]])
#     adj_mat = Adj_Mat(t)
#
#     a = torch.Tensor([ [[0,0], [0,0]], [[1,2], [0,0]] ])
#     adj_mat = Adj_Mat(a)
#
#     edge_index = torch.tensor([[0, 1, 1,], [1, 0, 2,]], dtype = torch.long)
#     edge_weight = torch.tensor([[-1], [2], [1]], dtype=torch.float)
#     adj_mat = Adj_Mat(indices=edge_index, values=edge_weight, size=(2,3,1))
