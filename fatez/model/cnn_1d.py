#!/usr/bin/env python3
"""
This file contains classes to build CNN based classifiers

Note 1
Just in case if LazyLinear having problem
You may want to try torchlayers
# import torchlayers as tl

Trying to avoid Lazy module since it's under development
But so far it's working just fine, so still using lazy module

flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
self.dense = nn.Linear(flattenLength, densed_size)

author: jy, nkmtmsys
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pkg_resources import resource_filename
from sklearn import cluster


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from collections import OrderedDict

# Ignoring warnings because of using LazyLinear
# import warnings
# warnings.filterwarnings('ignore')


class Model_1D(nn.Module):
    """
    Defining a CNN model treating input as 1D data
    with given hyperparameters
    Layer set number limited to max == 3
    """
    def __init__(self,
        in_channels:int = 1,
        n_class:int = 2,
        learning_rate:float = 0.01,
        num_layers:int = 2,
        conv_kernel_num:int = 32,
        conv_kernel_size:int = 4,
        maxpool_kernel_size:int = 2,
        densed_size:int = 32,
        device:str = 'cpu',
        dtype:str = None,
        ):
        # Initialization
        super().__init__()
        self.num_layers = num_layers
        assert self.num_layers > 0

        model_dict = OrderedDict([
            ('conv0', nn.Conv1d(
                in_channels = in_channels,
                out_channels = conv_kernel_num,
                kernel_size = conv_kernel_size
            )),
            ('relu0', nn.ReLU(inplace = False)),
            ('pool0', nn.MaxPool1d(kernel_size = maxpool_kernel_size))
        ])

        # Adding Conv blocks
        for i in range(self.num_layers - 1):
            model_dict.update(
                {
                    f'conv{i+1}': nn.Conv1d(
                        in_channels = conv_kernel_num,
                        out_channels = conv_kernel_num,
                        kernel_size = conv_kernel_size
                    )
                }
            )
            model_dict.update({f'relu{i+1}': nn.ReLU(inplace = False)})
            model_dict.update(
                {
                    f'pool{i+1}': nn.MaxPool1d(
                        kernel_size = maxpool_kernel_size
                    )
                }
            )

        # Adding FC, dense, and decision layers
        model_dict.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
        model_dict.update({f'dense': nn.LazyLinear(densed_size)})
        model_dict.update({f'relu_last': nn.ReLU(inplace = False)})
        model_dict.update({f'decide': nn.Linear(densed_size, n_class)})

        self.model = nn.Sequential(model_dict)
        self.optimizer = optim.SGD(self.model.parameters(), learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

    # Overwrite the forward function in nn.Module
    def forward(self, input, debug = False):
        if debug:
            print(input.shape)
            for layer in self.model:
                print(layer)
                input = layer(input)
                print(input.shape)
            out = input
        else:
            out = self.model(input)
        return func.softmax(out, dim = -1)


if __name__ == '__main__':
    n_fea = 100
    en_dim = 4
    param = {
        'learning_rate': 0.01,
        'num_layers': 2,
        'conv_kernel_num': 32,
        'conv_kernel_size': 4,
        'maxpool_kernel_size':2,
        'densed_size': 32
    }

    a = Model_1D(in_channels = en_dim, **param)
    print(a.model)
    # We would have (batch, nfea, en_dim) instead.
    # Remember to reshape!
    data = torch.randn(2, n_fea, en_dim).permute(0, 2, 1)
    print(a(data,))
