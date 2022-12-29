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
    def __init__(self,
        id:str = 'encoder',                 # model id
        encoder:TransformerEncoder = None,  # Load encoder
        d_model:int = 512,                  # number of expected features in the inputs
        n_layer:int = 6,                    # number of encoder layers
        nhead:int = 8,                      # number of heads in multi-head attention
        dim_feedforward:int = 2048,         # dimension of the feedforward network model
        dropout:float = 0.05,               # dropout ratio
        activation:str = 'gelu',            # original bert used gelu instead of relu
        layer_norm_eps:float = 1e-05,       # the eps value in layer normalization component
        device:str = None,
        dtype:str = None,
        ):
        super(Encoder, self).__init__()
        self.id = id
        self.d_model = d_model
        self.factory_kwargs={'device':device, 'dtype':dtype}
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

    def save(self, epoch, file_path:str = "output/encoder_trained.model"):
        """
        Saving the current BERT encoder

        :param epoch: current epoch number
        :param file_path: model output path
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.encoder.cpu(), output_path)
        self.encoder.to(self.factory_kwargs['device'])
        return output_path



class Data_Reconstructor(nn.Module):
    """
    Data_Reconstructor can be revised later
    """
    def __init__(self, d_model:int = 512, n_bin:int = 100,):
        super(Data_Reconstructor, self).__init__()
        self.linear = nn.Linear(d_model, n_bin)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input):
        return self.softmax(self.linear(input))



class Classifier(nn.Module):
    """
    Easy classifier. Can be revised later.
    scBERT use 1D-Conv here
    """
    def __init__(self, d_model:int = 512, n_hidden:int = 2, n_class:int = 100,):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(d_model, n_hidden)
        self.softmax = nn.LogSoftmax(dim = -1)
        self.decision = nn.LazyLinear(n_class)

    def forward(self, input):
        output = self.softmax(self.linear(input))
        output = torch.flatten(output, start_dim = 1)
        return F.softmax(self.decision(output), dim = -1)



class Pre_Train_Model(nn.Module):
    def __init__(self,
        encoder:Encoder = None,
        n_bin:int = 100,
        device:str = None,
        dtype:str = None,
        ):
        super(Pre_Train_Model, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.encoder = encoder
        self.encoder.to(self.factory_kwargs['device'])
        self.reconstructor = Data_Reconstructor(
            d_model = self.encoder.d_model, n_bin = n_bin
        )
        self.reconstructor.to(self.factory_kwargs['device'])

    def forward(self, input, mask):
        return self.reconstructor(self.encoder(input, mask))



class Fine_Tune_Model(nn.Module):
    def __init__(self,
        encoder:Encoder = None,
        n_hidden:int = 2,
        n_class:int = 100,
        device:str = None,
        dtype:str = None,
        ):
        super(Fine_Tune_Model, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.encoder = encoder
        self.encoder.to(self.factory_kwargs['device'])
        self.classifier = Classifier(
            d_model = self.encoder.d_model, n_hidden=n_hidden, n_class=n_class,
        )
        self.classifier.to(self.factory_kwargs['device'])

    def forward(self, input,):
        output = self.encoder(input)
        output = self.classifier(output)
        return output
