#!/usr/bin/env python3
"""
LSTM model implemented with PyTorch.

author: jy, nkmtmsys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    """
    Long-Short-Term Memory
    """
    def __init__(self,
        n_features:int = None,
        n_dim:int = None,
        hidden_size:int = None,
        num_layers:int = 1,
        bias:bool = True,
        batch_first:bool = True,
        dropout:float = 0.0,
        bidirectional:bool = True,
        proj_size:int = 0,
        n_class:int = 2,
        **kwargs
        ):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p = dropout, inplace = True)
        self.lstm = nn.LSTM(
            input_size = n_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias,
            batch_first = batch_first,
            # dropout = dropout,
            bidirectional = bidirectional,
            proj_size = proj_size,
        )
        self.fc = nn.Flatten(start_dim = 1, end_dim = -1)
        densed_size = hidden_size
        self.dense = nn.Linear(n_features * hidden_size, densed_size)
        self.relu = nn.ReLU(inplace = True)
        self.decision = nn.Linear(densed_size, n_class)

    def forward(self, input):
        # input needs to be: (batch_size, seq, input_size)
        input = self.dropout(input, training=self.training)
        # Set initial hidden and cell states
        if not self.bidirectional:
            h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        else:
            h0 = torch.zeros(2*self.num_layers, input.size(0), self.hidden_size)
            c0 = torch.zeros(2*self.num_layers, input.size(0), self.hidden_size)
        h0 = h0.to(input.device)
        c0 = c0.to(input.device)
        out, states = self.lstm(input, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out)
        out = self.dense(out)
        out = self.relu(out)
        out = self.decision(out)
        # Decode the hidden state of the last time step
        return F.softmax(out, dim = 1)
