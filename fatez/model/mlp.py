#!/usr/bin/env python3
"""
Some Multilayer perceptrons

author: jy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class Data_Reconstructor(nn.Module):
    """
    Data_Reconstructor can be revised later
    """
    def __init__(self,
        d_model:int = None,
        out_dim:int = None,
        device:str = 'cpu',
        dtype:str = None,
        ):
        """
        :param d_model:int = None
            Number of each gene's input features.

        :param out_dim:int = None
            Output dimension.

        :param device:str = 'cpu'
            The device to load model.

        :param dtype:str = None
            Data type of input values.
            Note: torch default using float32, numpy default using float64
        """
        super(Data_Reconstructor, self).__init__()
        self.linear = nn.Linear(d_model, out_dim, dtype = dtype)

    def forward(self, input):
        return self.linear(input)



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
        """
        :param d_model:int = None
            Number of each gene's input features.

        :param n_hidden:int = None
            Number of hidden units.

        :param n_class:int = None
            Number of classes.

        :param device:str = 'cpu'
            The device to load model.

        :param dtype:str = None
            Data type of input values.
            Note: torch default using float32, numpy default using float64
        """
        super(Classifier, self).__init__()
        self.linear = nn.Linear(d_model, n_hidden, dtype = dtype)
        self.softmax = nn.LogSoftmax(dim = -1)
        self.decision = nn.LazyLinear(n_class, dtype = dtype)

    def forward(self, input):
        output = self.softmax(self.linear(input))
        output = torch.flatten(output, start_dim = 1)
        return F.softmax(self.decision(output), dim = -1)
