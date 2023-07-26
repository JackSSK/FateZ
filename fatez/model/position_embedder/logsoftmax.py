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
        reverse:bool = False,
        dtype:type = torch.float32,
        **kwargs
        ):
        super(Embedder, self).__init__()
        self.mode = mode.upper()
        self.reverse = reverse
        self.encoder = nn.LogSoftmax(dim = dim)

    def forward(self, input, **kwargs):
        if self.mode == 'TENSOR':
            if not self.reverse: return self.encoder(input)
            else: return self.encoder(input) * -1
        elif self.mode == 'PYG':
            for ele in input:
                if not self.reverse: ele.x = self.encoder(ele.x)
                else: ele.x = self.encoder(ele.x) * -1
            return input
