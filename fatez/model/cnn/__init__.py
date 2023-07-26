#!/usr/bin/env python3
"""
This file contains classes to build CNN based classifiers

Note 1
Just in case if LazyLinear having problem
You may want to try torchlayers
# import torchlayers as tl

author: jy, nkmtmsys
"""
from fatez.model.cnn.cnn1d import Model as Model_1D
from fatez.model.cnn.cnn2d import Model as Model_2D
from fatez.model.cnn.cnnhyb import Model as Model_Hybrid

__all__ = [
    'Model_1D',
    'Model_2D',
    'Model_Hybrid',
]



# if __name__ == '__main__':
#     import torch
#     n_fea = 1100
#     en_dim = 4
#     data = torch.randn(2, n_fea, en_dim)
#     param_1d = {
#         'n_layer_set': 3,
#         'conv_kernel_num': 32,
#         'conv_kernel_size': 8,
#         'pool_kernel_size':2,
#         'densed_size': 32
#     }
#     param_2d = {
#         'n_layer_set': 1,
#         'conv_kernel_num': 32,
#         'conv_kernel_size': (8, 3),
#         'pool_kernel_size': (2, 2),
#         'densed_size': 32
#     }
#     param_hyb = {
#         'n_layer_set': 1,
#         'conv_kernel_num': 8,
#         'verti_kernel_size': 8,
#         'horiz_kernel_size': 3,
#         'pool_kernel_size': 2,
#         'densed_size': 32
#     }
#
#     a = Model_1D(in_channels = en_dim, **param_1d, data_shape = data.shape)
#     print(a.applicable)
#     print(a(data))
#
#     a = Model_2D(in_channels = 1, **param_2d, data_shape = data.shape)
#     print(a.applicable)
#     print(a(data))
#
#     a = Model_Hybrid(in_channels = 1, **param_hyb, data_shape = data.shape)
#     print(a.applicable)
#     print(a(data))
