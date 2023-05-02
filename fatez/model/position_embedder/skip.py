#!/usr/bin/env python3
"""
Just Skip positional embedding.

author: jy
"""
import torch
import torch.nn as nn



class Embedder(nn.Module):
    """
    Skip positional encoding.
    """
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()

    def forward(self, input, **kwargs):
        return input
