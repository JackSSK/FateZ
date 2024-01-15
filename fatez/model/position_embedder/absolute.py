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
        mode:str = 'TENSOR',
        dtype:type = torch.float32,
        **kwargs
        ):
        super(Embedder, self).__init__()
        self.mode = mode.upper()
        self.encoder = nn.Embedding(
            num_embeddings = n_embed,
            embedding_dim = n_dim,
            dtype = dtype
        )

    def forward(self, x, fea_ind:int = 1, **kwargs):
        if self.mode == 'TENSOR':
            ans = x + self.encoder(
                torch.arange(
                    x.shape[fea_ind],
                    device=x.device
                )
            )
        elif self.mode == 'PYG':
            raise Error('PyG mode under construction!')
        return ans
