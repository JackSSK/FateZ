#!/usr/bin/env python3
"""
Transformer objects

author: jy
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer



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
        device:str = 'cpu',
        dtype:str = None,
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

        :param device <str = 'cpu'>
            The device to load model.

        :param dtype <str = None>
            Data type of input tensor.
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.factory_kwargs = {'device':device, 'dtype':dtype}
        layer = TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
            **self.factory_kwargs
        )
        encoder_norm = LayerNorm(
            d_model,
            eps = layer_norm_eps,
            **self.factory_kwargs
        )
        self.encoder = TransformerEncoder(layer, n_layer, encoder_norm)

    def forward(self, input, mask = None):
        output = self.encoder(input, mask)
        return output



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
            **self.factory_kwargs
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
        output = func.softmax(self.decision(output), dim = -1)
        return output
