#!/usr/bin/env python3
"""
CNN with Hybrid 1D kernels (Horizontial set & Vertical set)
implemented with PyTorch.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Model(nn.Module):
    """
    Hybrid CNN using vertical and horizontial 1D kernels.
    """
    def __init__(self,
        n_features:int = 4,
        n_dim:int = 4,
        in_channels:int = 1,
        n_class:int = 2,
        n_layer_set:int = 1,
        conv_kernel_num:int = 32,
        horiz_kernel_size:int = 3,
        verti_kernel_size:int = 3,
        pool_kernel_size:int = 2,
        densed_size:int = 32,
        data_shape = None,
        dtype:str = None,
        **kwargs
        ):
        """
        :param n_features:int = None
            Number of input genes/regulons.

        :param n_dim:int = None
            Exp

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
            (horizontial kernels and vertical kernels counted seperately.)

        :param horiz_kernel_size:int = 4
            Size of horizontial convolution kernels.

        :param verti_kernel_size:int = 4
            Size of vertical convolution kernels.

        :param pool_kernel_size:int = 2
            Size of pooling kernels.

        :param densed_size:int = 32
            Number of hidden units in densed layer before decision layer.

        :param data_shape: = None
            Shape of expected input data.

        :param dtype:str = None
            The dtype for parameters in layers.
        """
        super().__init__()
        self.n_features = n_features
        self.n_dim = n_dim
        self.n_layer_set = n_layer_set
        self.conv_kernel_num = conv_kernel_num
        self.horiz_kernel_size = horiz_kernel_size
        self.verti_kernel_size = verti_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.applicable = self._check_applicability(data_shape)

        model_horiz = OrderedDict([
            ('conv0', nn.Conv2d(
                in_channels = in_channels,
                out_channels = conv_kernel_num,
                kernel_size = (1, horiz_kernel_size),
                dtype = dtype,
            )),
            ('relu0', nn.ReLU(inplace = True)),
            ('pool0', nn.MaxPool2d((1, pool_kernel_size)))
        ])
        horiz_size = self._cal_fc_size(
            n_dim,
            horiz_kernel_size,
            pool_kernel_size
            )*n_features

        model_verti = OrderedDict([
            ('conv0', nn.Conv2d(
                in_channels = in_channels,
                out_channels = conv_kernel_num,
                kernel_size = (verti_kernel_size, 1),
                dtype = dtype,
            )),
            ('relu0', nn.ReLU(inplace = True)),
            ('pool0', nn.MaxPool2d((pool_kernel_size, 1)))
        ])
        verti_size = self._cal_fc_size(
            n_features,
            horiz_kernel_size,
            pool_kernel_size
            )*n_dim

        for i in range(n_layer_set - 1):
            # Adding layer set to vertical model
            model_horiz.update({
                f'conv{i+1}': nn.Conv2d(
                    in_channels = conv_kernel_num,
                    out_channels = conv_kernel_num,
                    kernel_size = (1, horiz_kernel_size),
                    # Shrinking Kerenel size
                    # (1, int(horiz_kernel_size / pow(pool_kernel_size, i+1))),
                    dtype = dtype,
                )
            })
            model_horiz.update({f'relu{i+1}': nn.ReLU(inplace = True)})
            model_horiz.update({
                f'pool{i+1}': nn.MaxPool2d((1, pool_kernel_size))
            })
            horiz_size = self._cal_fc_size(
                horiz_size,
                horiz_kernel_size,
                pool_kernel_size
                )*n_features

            # Adding layer set to horizontial model
            model_verti.update({
                f'conv{i+1}': nn.Conv2d(
                    in_channels = conv_kernel_num,
                    out_channels = verti_size,
                    kernel_size = (verti_kernel_size, 1),
                    # Shrinking Kerenel size
                    # (int(verti_kernel_size / pow(pool_kernel_size, i+1)), 1),
                    dtype = dtype,
                )
            })
            model_verti.update({f'relu{i+1}': nn.ReLU(inplace = True)})
            model_verti.update({
                f'pool{i+1}': nn.MaxPool2d((pool_kernel_size, 1))
            })
            verti_size = self._cal_fc_size(
                verti_size,
                horiz_kernel_size,
                pool_kernel_size
                )*n_dim

        # Add Fully-Connect layers
        model_horiz.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
        model_verti.update({f'fc': nn.Flatten(start_dim = 1, end_dim = -1)})
        fc_size = (horiz_size *conv_kernel_num) + (verti_size *conv_kernel_num)

        self.model_horiz = nn.Sequential(model_horiz)
        self.model_verti = nn.Sequential(model_verti)
        self.decision = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(fc_size, densed_size, dtype = dtype)),
            ('relu_last', nn.ReLU(inplace = True)),
            ('decide', nn.Linear(densed_size, n_class, dtype = dtype))
        ]))

    def forward(self, input, debug:bool = False):
        if debug:
            print('Under construction...')
            return
        else:
            reshaped = self.reshape(input)
            verti_out = self.model_verti(reshaped)
            horiz_out = self.model_horiz(reshaped)
            out = self.decision(torch.cat((horiz_out, verti_out), dim = 1))
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
            n_features = n_features - self.verti_kernel_size + 1
            n_features = int(n_features / self.pool_kernel_size)
            n_dim = n_dim - self.horiz_kernel_size + 1
            n_dim = int(n_dim / self.pool_kernel_size)
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

    def _cal_fc_size(self, n_fea, kernel_size, n_pool, padding=0, stride=1,):
        """
        Calculate the output size of a layer set.
        """
        conv = ((n_fea - kernel_size + (2 * padding)) / stride) + 1
        pool = ((conv - n_pool + (2 * padding)) / n_pool) + 1
        return int(pool)
