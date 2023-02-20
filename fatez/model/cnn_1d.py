#!/usr/bin/env python3
"""
This file contains classes to build CNN based classifiers

Note 1
Just in case if LazyLinear having problem
You may want to try torchlayers
# import torchlayers as tl

Trying to avoid Lazy module since it's under development
But so far it's working just fine, so still using lazy module

flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layer_set))
self.dense = nn.Linear(flattenLength, densed_size)

author: jy, nkmtmsys
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict

# Ignoring warnings because of using LazyLinear
# import warnings
# warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    a 1D CNN model
    """
    def __init__(self,
        in_channels:int = 1,
        n_class:int = 2,
        num_layer_set:int = 1,
        conv_kernel_num:int = 32,
        conv_kernel_size:int = 4,
        maxpool_kernel_size:int = 2,
        densed_size:int = 32,
        app_check:int = None,
        device:str = 'cpu',
        dtype:str = None,
        ):
        # Initialization
        super().__init__()
        self.num_layer_set = num_layer_set
        self.conv_kernel_num = conv_kernel_num
        self.conv_kernel_size = conv_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.factory_kwargs = {'device': device, 'dtype': dtype,}
        self.app_check = self._check_feas(app_check)

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
        for i in range(self.num_layer_set - 1):
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

    def _check_feas(self, app_check):
        """
        Check model's applicability on the given data n_featuers.
        """
        if app_check is None: return None
        answer = self.num_layer_set > 0
        for i in range(self.num_layer_set):
            app_check = app_check - self.conv_kernel_size + 1
            app_check = int(app_check / self.maxpool_kernel_size)
            answer = app_check > 0
        return answer

    @staticmethod
    def reshape(input, order = [0, 2, 1]):
        """
        Make channel the second dim
        """
        return input.permute(*order)


# if __name__ == '__main__':
#     n_fea = 100
#     en_dim = 4
#     param = {
#         'num_layer_set': 3,
#         'conv_kernel_num': 32,
#         'conv_kernel_size': 8,
#         'maxpool_kernel_size':2,
#         'densed_size': 32
#     }
#     #
#     a = Model(in_channels = en_dim, **param, app_check = n_fea)
#     print(a.app_check)
#     data = torch.randn(2, n_fea, en_dim)
#     print(a(a.reshape(data),))
