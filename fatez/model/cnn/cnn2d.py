#!/usr/bin/env python3
"""
CNN with 2D kernels implemented with PyTorch.

author: jy, nkmtmsys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Model(nn.Module):
    """
    A standard 2D CNN model.
    """
    def __init__(self,
        in_channels:int = 1,
        n_class:int = 2,
        n_layer_set:int = 1,
        conv_kernel_num:int = 32,
        conv_kernel_size:set = (4, 2),
        pool_kernel_size:set = (2, 2),
        densed_size:int = 32,
        data_shape = None,
        dtype:str = None,
        **kwargs
        ):
        """
        :param in_channels:int = 1
            Feature numbers of input matrix.
            (Should be fixed to 1 since taking representations as 2D matrices.)

        :param n_class:int = 2
            Number of classes for predictions.

        :param n_layer_set:int = 1
            Number of layer sets.
            One layer set is consisting of 1 conv layer, 1 activation layer, and
            1 pooling layer.

        :param conv_kernel_num:int = 32
            Number of convolution kernels.

        :param conv_kernel_size:set = (4, 2)
            Size of convolution kernels.

        :param pool_kernel_size:set = (2, 2)
            Size of pooling kernels.

        :param densed_size:int = 32
            Number of hidden units in densed layer before decision layer.

        :param data_shape: = None
            Shape of expected input data.

        :param dtype:str = None
            The dtype for parameters in layers.
        """
        super().__init__()
        self.n_layer_set = n_layer_set
        self.conv_kernel_num = conv_kernel_num
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.applicable = self._check_applicability(data_shape)

        model_dict = OrderedDict([
            ('conv0', nn.Conv2d(
                in_channels = in_channels,
                out_channels = conv_kernel_num,
                kernel_size = conv_kernel_size,
                dtype = dtype,
            )),
            ('relu0', nn.ReLU(inplace = True)),
            ('pool0', nn.MaxPool2d(kernel_size = pool_kernel_size))
        ])

        # Adding Conv blocks
        for i in range(self.n_layer_set - 1):
            model_dict.update({
                f'conv{i+1}': nn.Conv2d(
                    in_channels = conv_kernel_num,
                    out_channels = conv_kernel_num,
                    kernel_size = conv_kernel_size,
                    dtype = dtype,
                )
            })
            model_dict.update({f'relu{i+1}': nn.ReLU(inplace = True)})
            model_dict.update({
                f'pool{i+1}': nn.MaxPool2d(kernel_size = pool_kernel_size)
            })

        # Adding FC, dense, and decision layers
        model_dict.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
        model_dict.update({f'dense': nn.LazyLinear(densed_size, dtype = dtype)})
        model_dict.update({f'relu_last': nn.ReLU(inplace = True)})
        model_dict.update(
            {f'decide': nn.Linear(densed_size, n_class, dtype = dtype)}
        )

        self.model = nn.Sequential(model_dict)

    def forward(self, input, debug:bool = False):
        reshaped = self.reshape(input)
        if debug:
            print(reshaped.shape)
            for layer in self.model:
                print(layer)
                reshaped = layer(reshaped)
                print(reshaped.shape)
            out = reshaped
        else:
            out = self.model(reshaped)
        return F.softmax(out, dim = -1)

    def _check_applicability(self, data_shape):
        """
        Check model's applicability on the given data shape.
        """
        if data_shape is None: return None
        answer = self.n_layer_set > 0
        n_dim = data_shape[-1]
        n_features = data_shape[-2]
        for i in range(self.n_layer_set):
            n_features = n_features - self.conv_kernel_size[0] + 1
            n_features = int(n_features / self.pool_kernel_size[0])
            n_dim = n_dim - self.conv_kernel_size[1] + 1
            n_dim = int(n_dim / self.pool_kernel_size[1])
            answer = n_features > 0 and n_dim > 0
        return answer

    @staticmethod
    def reshape(input, shape = None):
        """
        Reshape the input mat
        """
        if shape != None:
            return torch.reshape(input, shape)
        else:
            return torch.reshape(
                input,
                (input.shape[0], 1, input.shape[1], input.shape[2])
            )
