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

        :param device <str = 'cpu'>
            The device to load model.

        :param dtype <str = None>
            Data type of input tensor.
        """
        super(Encoder, self).__init__()
        self.id = id
        self.d_model = d_model
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
                layer_norm_eps,
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



class Data_Reconstructor(nn.Module):
    """
    Data_Reconstructor can be revised later
    """
    def __init__(self,
        d_model:int = 512,
        n_bin:int = 100,
        device:str = 'cpu',
        dtype:str = None,):
        super(Data_Reconstructor, self).__init__()
        self.linear = nn.Linear(d_model, n_bin, dtype = dtype)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input):
        return self.softmax(self.linear(input))



class Classifier(nn.Module):
    """
    Easy classifier. Can be revised later.
    scBERT use 1D-Conv here
    """
    def __init__(self,
        d_model:int = 512,
        n_hidden:int = 2,
        n_class:int = 100,
        device:str = 'cpu',
        dtype:str = None,
        ):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(d_model, n_hidden, dtype = dtype)
        self.softmax = nn.LogSoftmax(dim = -1)
        self.decision = nn.LazyLinear(n_class, dtype = dtype)

    def forward(self, input):
        output = self.softmax(self.linear(input))
        output = torch.flatten(output, start_dim = 1)
        return F.softmax(self.decision(output), dim = -1)



class Pre_Train_Model(nn.Module):
    """
    The model for pre-training process.
    """
    def __init__(self,
        encoder:Encoder = None,
        n_bin:int = 100,
        ):
        super(Pre_Train_Model, self).__init__()
        self.encoder = encoder
        self.factory_kwargs = {
            'device': self.encoder.factory_kwargs['device'],
            'dtype': self.encoder.factory_kwargs['dtype']
        }
        self.encoder.to(self.factory_kwargs['device'])
        self.reconstructor = Data_Reconstructor(
            d_model = self.encoder.d_model,
            n_bin = n_bin,
            **self.factory_kwargs
        )
        self.reconstructor.to(self.factory_kwargs['device'])

    def forward(self, input, mask = None):
        return self.reconstructor(self.encoder(input, mask))



class Fine_Tune_Model(nn.Module):
    """
    The model for fine-tuning process.
    """
    def __init__(self,
        encoder:Encoder = None,
        n_hidden:int = 2,
        n_class:int = 100,
        ):
        super(Fine_Tune_Model, self).__init__()
        self.encoder = encoder
        self.factory_kwargs = {
            'device': self.encoder.factory_kwargs['device'],
            'dtype': self.encoder.factory_kwargs['dtype']
        }
        self.encoder.to(self.encoder.factory_kwargs['device'])
        self.classifier = Classifier(
            d_model = self.encoder.d_model,
            n_hidden = n_hidden,
            n_class = n_class,
            **self.factory_kwargs
        )
        self.classifier.to(self.factory_kwargs['device'])

    def forward(self, input,):
        output = self.encoder(input)
        output = self.classifier(output)
        return output
