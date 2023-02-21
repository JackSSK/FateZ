#!/usr/bin/env python3
"""
Some Multilayer perceptrons

author: jy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Model(nn.Module):
    """
    Easy classifier. Can be revised later.
    scBERT use 1D-Conv here
    """
    def __init__(self,
        type:str = 'CLF',
        d_model:int = 512,
        n_layer_set:int = 2,
        n_hidden:int = 2,
        n_class:int = 100,
        device:str = 'cpu',
        dtype:str = None,
        ):
        """
        :param d_model:int = None
            Number of each gene's input features.

        :param n_layer_set:int = None
            Number of layer set.

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
        super(Model, self).__init__()
        if n_layer_set == 1:
            self.model = OrderedDict([
                (f'decide', nn.Linear(d_model, n_class, dtype = dtype)),
            ])
        elif n_layer_set >= 2:
            self.model = OrderedDict([
                (f'layer0', nn.Linear(d_model, n_hidden, dtype = dtype)),
                (f'act0', nn.LogSoftmax(dim = -1)),
            ])
            for i in range(n_layer_set - 1):
                self.model.update(
                    {f'layer{i+1}': nn.Linear(n_hidden, n_hidden, dtype=dtype)}
                )
                self.model.update({f'act{i+1}': nn.LogSoftmax(dim = -1)})
            self.model.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
            self.model.update({f'decide': nn.LazyLinear(n_class, dtype=dtype)})
        else:
            raise Exception(f'Invalid n_layer_set:{n_layer_set}')

        # Add softmax activation if acting as classifier
        if type.upper() == 'CLF':
            self.model.update({f'act_last': nn.Softmax(dim = -1)})
        # Only one layer is acceptable for data reconstructor
        elif type.upper() == 'RECON':
            assert n_layer_set == 1

        self.model = nn.Sequential(self.model)

    def forward(self, input):
        return self.model(input)
