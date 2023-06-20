#!/usr/bin/env python3
"""
GRU model implemented with PyTorch.

author: jy, nkmtmsys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Model(nn.Module):
    """
    Gated Recurrent Unit
    """
    def __init__(self,
        n_features:int = None,
        n_dim:int = None,
        hidden_size:int = None,
        densed_size:int = None,
        num_layers:int = 1,
        bias:bool = True,
        batch_first:bool = True,
        dropout:float = 0.0,
        bidirectional:bool = True,
        n_class:int = 2,
        dtype:str = None,
        **kwargs
        ):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        # self.dropout = nn.Dropout(p = dropout, inplace = True)
        self.gru = nn.GRU(
            input_size = n_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias,
            batch_first = batch_first,
            # dropout = dropout,
            bidirectional = bidirectional
        )
        densed_size = hidden_size
        fc_size = self._cal_fc_size(n_features)
        self.decision = nn.Sequential(OrderedDict([
            ('fc', nn.Flatten(start_dim = 1, end_dim = -1)),
            ('dense', nn.Linear(fc_size, densed_size, dtype=dtype)),
            ('relu', nn.ReLU(inplace = True)),
            ('decide', nn.Linear(densed_size, n_class, dtype=dtype))
        ]))

    def forward(self, input):
        # input needs to be: (batch_size, seq, input_size)
        # input = self.dropout(input, training=self.training)
        # Set initial hidden states
        if not self.bidirectional:
            h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        else:
            h0 = torch.zeros(2*self.num_layers, input.size(0), self.hidden_size)
        h0 = h0.to(input.device)
        out, states = self.gru(input, h0)
        out = self.decision(out)
        return F.softmax(out, dim = 1)

    def _cal_fc_size(self, n_fea, ):
        out = n_fea * self.hidden_size
        if self.bidirectional: out *= 2
        return out
