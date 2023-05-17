#!/usr/bin/env python3
"""
This file contains basic objects for storing data.

author: jy
"""
import re
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data



class FateZ_Dataset(Dataset):
    """
	Basic Dataset object for DataLoader
	"""
    def __init__(self, samples, **kwargs):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = [sample.x, sample.edge_index, sample.edge_attr]
        return data, sample.y



class Adj_Mat(object):
    """
    docstring for Adjacent Matix.
    """
    def __init__(self,
        full_mat:torch.Tensor = None,
        ind:torch.Tensor = None,
        val:torch.Tensor = None,
        size = None,
        sparse_dim:int = 2,
        **kwargs
        ):
        super(Adj_Mat, self).__init__()
        if full_mat is None and len(ind.shape) == 3:
            t=[torch.sparse_coo_tensor(ind[i],v,size) for i,v in enumerate(val)]
            self.sparse = torch.stack(t, 0).coalesce()
        elif full_mat is None and len(ind.shape) == 2:
            self.sparse =torch.sparse_coo_tensor(ind, val, size).coalesce()
        elif str(full_mat.layout) == 'torch.strided':
            self.sparse = full_mat.to_sparse(sparse_dim = sparse_dim).coalesce()
        elif str(full_mat.layout) == 'torch.sparse_coo':
            self.sparse = full_mat.coalesce()

    def unpack(self,):
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
#     adj_mat = Adj_Mat(indices=edge_index, values=edge_weight, shape=(2,3,1))
