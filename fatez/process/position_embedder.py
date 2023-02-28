#!/usr/bin/env python3
"""
Modules for positional embedding.
Both trainable and untrainable methods are here.

author: jy

ToDo:
Randowm Walking PE not done yet.
Pos Embed before GNN or after GNN not decided yet:
    Current version embed after GNN since GNN should be pos sensitive
"""
import torch
import torch.nn as nn
# from torch_scatter import scatter_add
# import torch_geometric.utils as utils


def Set(config:dict = None, factory_kwargs:dict = None):
    """
    Set up positional embedder based on given config.
    """
    if config['pos_embedder']['type'] == 'Skip':
        return Skip()
    elif config['pos_embedder']['type'] == 'ABS':
        return Absolute_Encode(
            **config['pos_embedder']['params'], **factory_kwargs
        )
    elif config['pos_embedder']['type'] == 'ABS':
        return
    else:
        raise model.Error(f'Unknown pos_embedder type')



class Skip(nn.Module):
    """
    Skip positional encoding.
    """
    def __init__(self, **kwargs):
        super(Skip, self).__init__()

    def forward(self, input):
        return input



class Absolute_Encode(nn.Module):
    """
    Absolute positional encoding.
    """
    def __init__(self,
        n_embed:int = None,
        n_dim:int = None,
        device:str = 'cpu',
        dtype:type = torch.float32,
        **kwargs
        ):
        super(Absolute_Encode, self).__init__()
        self.encoder = nn.Embedding(
            num_embeddings = n_embed,
            embedding_dim = n_dim,
            dtype = dtype
        )
        self.factory_kwargs = {'device': device, 'dtype': dtype,}

    def forward(self, x, fea_ind:int = 1):
        return x + self.encoder(torch.arange(x.shape[fea_ind], device=x.device))



class RW_Encode(object):
    def __init__(self, dim, use_edge_attr=False, normalization=None, **kwargs):
        """
        Random walk PE from SAT. Not adjusted yet!
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
        W = W0
        vector = torch.zeros((graph.num_nodes, self.pos_enc_dim))
        vector[:, 0] = torch.from_numpy(W0.diagonal())
        for i in range(self.pos_enc_dim - 1):
            W = W.dot(W0)
            vector[:, i + 1] = torch.from_numpy(W.diagonal())
        return vector.float()


    def normalize_adj(self, edge_index, edge_weight = None, num_nodes = None):
        edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                device = edge_index.device
            )
        num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim = 0, dim_size = num_nodes)
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
        return utils.to_scipy_sparse_matrix(
            edge_index,
            edge_weight,
            num_nodes = num_nodes
        )

    def apply_to(self, dataset):
        dataset.abs_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.abs_pe_list.append(pe)
        return dataset
