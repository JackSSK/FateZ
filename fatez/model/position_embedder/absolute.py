#!/usr/bin/env python3
"""
Absolute Positional Embedding

author: jy
"""
import torch
import torch.nn as nn
# from torch_scatter import scatter_add
# import torch_geometric.utils as utils



class Embedder(nn.Module):
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
        super(Embedder, self).__init__()
        self.encoder = nn.Embedding(
            num_embeddings = n_embed,
            embedding_dim = n_dim,
            dtype = dtype
        )
        self.factory_kwargs = {'device': device, 'dtype': dtype,}

    def forward(self, x, fea_ind:int = 1, **kwargs):
        return x + self.encoder(torch.arange(x.shape[fea_ind], device=x.device))
