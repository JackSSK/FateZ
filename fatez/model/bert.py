#!/usr/bin/env python3
"""
BERT modeling

ToDo:
    1. Pre_Train & Fine_Tune Process
        Note: Revise the iteration part
    2. Revise Data_Reconstructor and Classifier if necessary

author: jy, nkmtmsys
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

import fatez.model as model
import fatez.model.mlp as mlp


class Encoder(nn.Module):
    """
    The Encoder for BERT model.
    """
    def __init__(self,
        id:str = 'encoder',
        encoder:TransformerEncoder = None,
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
        :param encoder <torch.nn.TransformerEncoder = None>
            The encoder to load.

    	:param encoder <torch.nn.TransformerEncoder = None>
            The encoder to load.

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
        self.id = id
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.factory_kwargs = {'device':device, 'dtype':dtype}
        if encoder is not None:
            self.encoder = encoder
        else:
            layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
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



class Pre_Train_Model(nn.Module):
    """
    The model for pre-training process.
    """
    def __init__(self,
        encoder:Encoder = None,
        n_bin:int = 100,
        n_dim:int = 2,
        n_dim_adj:int = None,
        ):
        """
        :param encoder:Encoder = None
            The Encoder to build pre-train model with.

        :param n_bin:int = None
            Depreciated now.

        :param n_dim:int = 2
            The output dimension for reconstructing node feature mat.

        :param n_dim_adj:int = None
            The output dimension for reconstructing adj mat.
        """
        super(Pre_Train_Model, self).__init__()
        # Need to delete the rename line below
        n_dim_node = n_dim

        self.encoder = encoder
        self.factory_kwargs = {
            'device': self.encoder.factory_kwargs['device'],
            'dtype': self.encoder.factory_kwargs['dtype']
        }
        self.encoder.to(self.factory_kwargs['device'])
        self.reconstructor_node = mlp.Data_Reconstructor(
            d_model = self.encoder.dim_feedforward,
            out_dim = n_dim_node,
            **self.factory_kwargs
        )
        self.reconstructor_node.to(self.factory_kwargs['device'])
        self.reconstructor_adj = None

        if n_dim_adj is not None:
            self.reconstructor_adj = mlp.Data_Reconstructor(
                d_model = self.encoder.dim_feedforward,
                out_dim = n_dim_adj,
                **self.factory_kwargs
            )
            self.reconstructor_adj.to(self.factory_kwargs['device'])


    def forward(self, input, mask = None):
        embed_rep = self.encoder(input, mask)
        node_recon = self.reconstructor_node(embed_rep)
        adj_recon = None
        if self.reconstructor_adj is not None:
            adj_recon = self.reconstructor_adj(embed_rep)
        # Need to revise output here
        # return node_recon, adj_recon
        return node_recon



class Fine_Tune_Model(nn.Module):
    """
    The model for fine-tuning process.
    """
    def __init__(self,
        encoder:Encoder = None,
        n_hidden:int = 2,
        n_class:int = 100,
        ):
        """
        :param encoder:Encoder = None
            The Encoder to build fine-tune model with.

        :param n_hidden:int = 2
            The hidden units for classifier.

        :param n_class:int = 100
            Number of classes to classify,
        """
        super(Fine_Tune_Model, self).__init__()
        self.encoder = encoder
        self.factory_kwargs = {
            'device': self.encoder.factory_kwargs['device'],
            'dtype': self.encoder.factory_kwargs['dtype']
        }
        self.encoder.to(self.encoder.factory_kwargs['device'])
        self.classifier = mlp.Classifier(
            d_model = self.encoder.d_model,
            n_hidden = n_hidden,
            n_class = n_class,
            **self.factory_kwargs
        )
        self.classifier.to(self.factory_kwargs['device'])

    def forward(self, input, mask = None):
        output = self.encoder(input, mask)
        output = self.classifier(output)
        return output
