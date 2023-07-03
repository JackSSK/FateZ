#!/usr/bin/env python3
"""
Apply LogSoftmax as info embedding.

author: jy
"""
import torch
import torch.nn as nn



class Embedder(nn.Module):
    """
    Absolute positional encoding.
    """
    def __init__(self,
        dim:int = -2,
        mode:str = 'PYG',
        dtype:type = torch.float32,
        **kwargs
        ):
        super(Embedder, self).__init__()
        self.mode = mode.upper()
        self.encoder = nn.LogSoftmax(dim = dim)

    def forward(self, input, **kwargs):
        if self.mode == 'TENSOR':
            return self.encoder(input)
        elif self.mode == 'PYG':
            for ele in input:
                ele.x = self.encoder(ele.x)
            return input
