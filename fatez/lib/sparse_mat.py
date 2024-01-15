#!/usr/bin/env python3
"""
This file contains basic objects for transforming matrices between dense format
and sparse coo format.

author: jy
"""
import torch

def get_sparse_coo(mat:torch.Tensor, sparse_dim:int = 2,):
    """
    Transform a dense matrix into sparse COO format.
    Return indices matrices and values matrices.
    """
    coo = mat.cpu().to_sparse(sparse_dim = sparse_dim).coalesce()
    # Reshape value matrix if edges only have single features
    values = coo.values()
    if len(values.shape) == 1:
        values = torch.reshape(values, (len(values), 1))
    return coo.indices().to(mat.device), values.to(mat.device)

def get_dense(inds:torch.Tensor, vals:torch.Tensor, size,):
    """
    Transform set of indices matrices and values matrices into dense format.
    """
    if len(inds.shape) == 3:
        t = [torch.sparse_coo_tensor(inds[i].cpu(),v,size) for i,v in enumerate(
            vals.cpu())]
        coo = torch.stack(t, 0).coalesce()
    elif len(inds.shape) == 2:
        coo = torch.sparse_coo_tensor(inds.cpu(), vals.cpu(), size).coalesce()

    # If each edge only has one feature, reduce the dimensionality
    if len(coo.shape) > 2 and coo.shape[-1] == 1:
        return coo.to_dense().reshape(coo.shape[:-1]).to(inds.device)
    else:
        return coo.to_dense().to(inds.device)

def get_dense_adjs(input, size):
    """
    Make densed tensors based on Adj matrices in PyG data objects.
    """
    answer = list()
    for ele in input:
        answer.append(get_dense(ele.edge_index, ele.edge_attr, size))
    return torch.stack(answer, 0)
