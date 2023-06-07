#!/usr/bin/env python3
"""
Transformer Encoder

author: jy
"""
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer



class Encoder(nn.Module):
    """
    The Encoder for BERT model.
    """
    def __init__(self,
        d_model:int = 512,
        n_layer:int = 6,
        nhead:int = 8,
        dim_feedforward:int = 2048,
        dropout:float = 0.05,
        activation:str = 'gelu',
        layer_norm_eps:float = 1e-05,
        batch_first:bool = True,
        **kwargs
        ):
        """
        :param d_model <int = 512>
            Number of expected features in the inputs.

        :param n_layer <int = 6>
            Number of encoder layers.

        :param nhead <int = 8>
            Number of heads in multi-head attention.

        :param dim_feedforward <int = 2048>
            Dimension of the feedforward network model.

        :param dropout <float = 0.05>
            The dropout ratio.

        :param activation <str = 'gelu'>
            The activation method.
            Note: Original BERT used gelu instead of relu

        :param layer_norm_eps <float = 1e-05>
            The eps value in layer normalization component.

        :param batch_first <bool = True>
            Whether batch size expected as first ele in dim or not.
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        layer = TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
        )
        encoder_norm = LayerNorm(d_model, eps = layer_norm_eps,)
        self.encoder = TransformerEncoder(layer, n_layer, encoder_norm)

    def forward(self, input, mask = None):
        output = self.encoder(input, mask)
        return output
