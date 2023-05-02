#!/usr/bin/env python3
"""
Transformer Decoder

author: jy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer



class Decoder(nn.Module):
    """
    docstring for Decoder.
    """
    def __init__(self, **kwarg):
        super(Decoder, self).__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
        )
        self.decoder = TransformerDecoder(decoder_layer, n_layer)
        self.decision = nn.Linear(d_model, n_class)
        # init_weights
        init_range = 0.1
        nn.init.zeros_(self.decision.bias)
        nn.init.uniform_(self.decision.weight, -init_range, init_range)

    def forward(self, input, encoded):
        output = self.decoder(input, encoded)
        output = torch.flatten(output, start_dim = 1)
        output = F.softmax(self.decision(output), dim = -1)
        return output
