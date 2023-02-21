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

import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict


class Model_1D(nn.Module):
    """
    A 1D CNN model
    """
    def __init__(self,
        in_channels:int = 1,
        n_class:int = 2,
        num_layer_set:int = 1,
        conv_kernel_num:int = 32,
        conv_kernel_size:int = 4,
        maxpool_kernel_size:int = 2,
        densed_size:int = 32,
        data_shape = None,
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
        self.applicable = self._check_applicability(data_shape)

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
            model_dict.update({
                f'conv{i+1}': nn.Conv1d(
                    in_channels = conv_kernel_num,
                    out_channels = conv_kernel_num,
                    kernel_size = conv_kernel_size
                )
            })
            model_dict.update({f'relu{i+1}': nn.ReLU(inplace = False)})
            model_dict.update({
                f'pool{i+1}': nn.MaxPool1d(kernel_size = maxpool_kernel_size)
            })

        # Adding FC, dense, and decision layers
        model_dict.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
        model_dict.update({f'dense': nn.LazyLinear(densed_size)})
        model_dict.update({f'relu_last': nn.ReLU(inplace = False)})
        model_dict.update({f'decide': nn.Linear(densed_size, n_class)})

        self.model = nn.Sequential(model_dict)

    def forward(self, input, debug:bool = False):
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

    def _check_applicability(self, data_shape):
        """
        Check model's applicability on the given data shape.
        """
        if data_shape is None: return None
        answer = self.num_layer_set > 0
        n_features = data_shape[-2]
        for i in range(self.num_layer_set):
            n_features = n_features - self.conv_kernel_size + 1
            n_features = int(n_features / self.maxpool_kernel_size)
            answer = n_features > 0
        return answer

    @staticmethod
    def reshape(input, order = [0, 2, 1]):
        """
        Make channel the second dim
        """
        return input.permute(*order)



class Model_2D(nn.Module):
    """
    A standard 2D CNN model.
    """
    def __init__(self,
        in_channels:int = 1,
        n_class:int = 2,
        num_layer_set:int = 1,
        conv_kernel_num:int = 32,
        conv_kernel_size:set = (4, 2),
        maxpool_kernel_size:set = (2, 2),
        densed_size:int = 32,
        data_shape = None,
        device:str = 'cpu',
        dtype:str = None,
        ):
        super().__init__()
        # Initialization
        self.num_layer_set = num_layer_set
        self.conv_kernel_num = conv_kernel_num
        self.conv_kernel_size = conv_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.factory_kwargs = {'device': device, 'dtype': dtype,}
        self.applicable = self._check_applicability(data_shape)

        model_dict = OrderedDict([
            ('conv0', nn.Conv2d(
                in_channels = in_channels,
                out_channels = conv_kernel_num,
                kernel_size = conv_kernel_size,
            )),
            ('relu0', nn.ReLU(inplace = False)),
            ('pool0', nn.MaxPool2d(kernel_size = maxpool_kernel_size))
        ])

        # Adding Conv blocks
        for i in range(self.num_layer_set - 1):
            model_dict.update({
                f'conv{i+1}': nn.Conv2d(
                    in_channels = conv_kernel_num,
                    out_channels = conv_kernel_num,
                    kernel_size = conv_kernel_size
                )
            })
            model_dict.update({f'relu{i+1}': nn.ReLU(inplace = False)})
            model_dict.update({
                f'pool{i+1}': nn.MaxPool2d(kernel_size = maxpool_kernel_size)
            })

        # Adding FC, dense, and decision layers
        model_dict.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
        model_dict.update({f'dense': nn.LazyLinear(densed_size)})
        model_dict.update({f'relu_last': nn.ReLU(inplace = False)})
        model_dict.update({f'decide': nn.Linear(densed_size, n_class)})

        self.model = nn.Sequential(model_dict)

    def forward(self, input, debug:bool = False):
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

    def _check_applicability(self, data_shape):
        """
        Check model's applicability on the given data shape.
        """
        if data_shape is None: return None
        answer = self.num_layer_set > 0
        n_dim = data_shape[-1]
        n_features = data_shape[-2]
        for i in range(self.num_layer_set):
            n_features = n_features - self.conv_kernel_size[0] + 1
            n_features = int(n_features / self.maxpool_kernel_size[0])
            n_dim = n_dim - self.conv_kernel_size[1] + 1
            n_dim = int(n_dim / self.maxpool_kernel_size[1])
            answer = n_features > 0 and n_dim > 0
        return answer

    @staticmethod
    def reshape(input, shape = None):
        """
        Reshape the input mat
        """
        if shape is not None:
            return torch.reshape(input, shape)
        else:
            return torch.reshape(
                input,
                (input.shape[0], 1, input.shape[1], input.shape[2])
            )



class Model_Hybrid(nn.Module):
    """
    Hybrid CNN using vertical and horizontial 1D kernels.
    """
    def __init__(self,
        in_channels:int = 1,
        n_class:int = 2,
        num_layer_set:int = 1,
        conv_kernel_num:int = 32,
        horiz_kernel_size:int = 4,
        verti_kernel_size:int = 4,
        maxpool_kernel_size:int = 2,
        densed_size:int = 32,
        data_shape = None,
        device:str = 'cpu',
        dtype:str = None,
        ):
        super().__init__()
        # Initialization
        self.num_layer_set = num_layer_set
        self.conv_kernel_num = conv_kernel_num
        self.horiz_kernel_size = horiz_kernel_size
        self.verti_kernel_size = verti_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.factory_kwargs = {'device': device, 'dtype': dtype,}
        self.applicable = self._check_applicability(data_shape)

        model_horiz = OrderedDict([
            ('conv0', nn.Conv2d(
                in_channels,
                conv_kernel_num,
                (1, horiz_kernel_size)
            )),
            ('relu0', nn.ReLU(inplace = False)),
            ('pool0', nn.MaxPool2d((1, maxpool_kernel_size)))
        ])

        model_verti = OrderedDict([
            ('conv0', nn.Conv2d(
                in_channels,
                conv_kernel_num,
                (verti_kernel_size, 1)
            )),
            ('relu0', nn.ReLU(inplace = False)),
            ('pool0', nn.MaxPool2d((maxpool_kernel_size, 1)))
        ])

        for i in range(num_layer_set - 1):
            # Adding layer set to vertical model
            model_horiz.update({
                f'conv{i+1}': nn.Conv2d(
                    conv_kernel_num,
                    conv_kernel_num,
                    (1, horiz_kernel_size)
                    # Shrinking Kerenel size
                    # (1, int(horiz_kernel_size / pow(maxpool_kernel_size, i+1)))
                )
            })
            model_horiz.update({f'relu{i+1}': nn.ReLU(inplace = False)})
            model_horiz.update({
                f'pool{i+1}': nn.MaxPool2d((1, maxpool_kernel_size))
            })

            # Adding layer set to horizontial model
            model_verti.update({
                f'conv{i+1}': nn.Conv2d(
                    conv_kernel_num,
                    conv_kernel_num,
                    (verti_kernel_size, 1)
                    # Shrinking Kerenel size
                    # (int(verti_kernel_size / pow(maxpool_kernel_size, i+1)), 1)
                )
            })
            model_verti.update({f'relu{i+1}': nn.ReLU(inplace = False)})
            model_verti.update({
                f'pool{i+1}': nn.MaxPool2d((maxpool_kernel_size, 1))
            })

        # Add Fully-Connect layers
        model_horiz.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
        model_verti.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})

        self.model_horiz = nn.Sequential(model_horiz)
        self.model_verti = nn.Sequential(model_verti)
        self.decision = nn.Sequential(OrderedDict([
            ('dense', nn.LazyLinear(densed_size)),
            ('relu_last', nn.ReLU(inplace = False)),
            ('decide', nn.Linear(densed_size, n_class))
        ]))

    def forward(self, input, debug:bool = False):
        if debug:
            print('Under construction...')
            return
        else:
            verti_out = self.model_verti(input)
            horiz_out = self.model_horiz(input)
            out = self.decision(torch.cat((horiz_out, verti_out), dim = 1))
            return func.softmax(out, dim = -1)

    def _check_applicability(self, data_shape):
        """
        Check model's applicability on the given data shape.
        """
        if data_shape is None: return None
        answer = self.num_layer_set > 0
        n_dim = data_shape[-1]
        n_features = data_shape[-2]
        for i in range(self.num_layer_set):
            n_features = n_features - self.verti_kernel_size + 1
            n_features = int(n_features / self.maxpool_kernel_size)
            n_dim = n_dim - self.horiz_kernel_size + 1
            n_dim = int(n_dim / self.maxpool_kernel_size)
            answer = n_features > 0 and n_dim > 0
        return answer

    @staticmethod
    def reshape(input, shape = None):
        """
        Reshape the input mat
        """
        if shape is not None:
            return torch.reshape(input, shape)
        else:
            return torch.reshape(
                input,
                (input.shape[0], 1, input.shape[1], input.shape[2])
            )

if __name__ == '__main__':
    # n_fea = 100
    # en_dim = 4
    # data = torch.randn(2, n_fea, en_dim)
    # param_1d = {
    #     'num_layer_set': 3,
    #     'conv_kernel_num': 32,
    #     'conv_kernel_size': 8,
    #     'maxpool_kernel_size':2,
    #     'densed_size': 32
    # }
    # param_2d = {
    #     'num_layer_set': 1,
    #     'conv_kernel_num': 32,
    #     'conv_kernel_size': (8, 3),
    #     'maxpool_kernel_size': (2, 2),
    #     'densed_size': 32
    # }
    # param_hyb = {
    #     'num_layer_set': 1,
    #     'conv_kernel_num': 32,
    #     'verti_kernel_size': 8,
    #     'horiz_kernel_size': 3,
    #     'maxpool_kernel_size': 2,
    #     'densed_size': 32
    # }
    #
    # a = Model_1D(in_channels = en_dim, **param_1d, data_shape = data.shape)
    # print(a.applicable)
    # print(a(a.reshape(data),))
    #
    # a = Model_2D(in_channels = 1, **param_2d, data_shape = data.shape)
    # print(a.applicable)
    # print(a(a.reshape(data),))
    #
    # a = Model_Hybrid(in_channels = 1, **param_hyb, data_shape = data.shape)
    # print(a.applicable)
    # print(a(a.reshape(data),))
