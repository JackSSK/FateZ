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
        n_features:int = None,
        d_model:int = None,
        n_layer_set:int = 2,
        n_hidden:int = 2,
        n_class:int = 100,
        dtype:str = None,
        **kwargs
        ):
        """
        :param n_features:int = None
            Number of input genes/regulons.

        :param d_model:int = None
            Number of each gene's features.

        :param n_layer_set:int = None
            Number of layer set.

        :param n_hidden:int = None
            Number of hidden units.

        :param n_class:int = None
            Number of classes.
        """
        super(Model, self).__init__()
        if n_layer_set == 1:
            model = OrderedDict([
                (f'decide', nn.Linear(d_model, n_class, dtype = dtype)),
            ])
        elif n_layer_set >= 2:
            model = OrderedDict([
                (f'layer0', nn.Linear(d_model, n_hidden, dtype = dtype)),
                (f'act0', nn.LogSoftmax(dim = -1)),
            ])
            for i in range(n_layer_set - 1):
                model.update(
                    {f'layer{i+1}': nn.Linear(n_hidden, n_hidden, dtype=dtype)}
                )
                model.update({f'act{i+1}': nn.LogSoftmax(dim = -1)})
            model.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
            if n_features is None:
                decision = nn.LazyLinear(n_class, dtype=dtype)
            else:
                decision = nn.Linear(n_features*n_hidden, n_class, dtype=dtype)
            model.update({f'decide': decision})

        else:
            raise Exception(f'Invalid n_layer_set:{n_layer_set}')

        # Add softmax activation if acting as classifier
        if type.upper() == 'CLF':
            model.update({f'act_last': nn.Softmax(dim = -1)})
        # Only one layer is acceptable for data reconstructor
        elif type.upper() == 'RECON':
            assert n_layer_set == 1

        self.model = nn.Sequential(model)

    def forward(self, input):
        return self.model(input)
