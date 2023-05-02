#!/usr/bin/env python3
"""
Standard RNN model implemented with PyTorch.

author: jy, nkmtmsys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    """
    Standard RNN
    """
    def __init__(self,
        input_size:int = None,
        hidden_size:int = None,
        num_layers:int = 1,
        nonlinearity:str = 'tanh',
        bias:bool = True,
        batch_first:bool = True,
        dropout:float = 0.0,
        bidirectional:bool = False,
        n_class:int = 2,
        **kwargs
        ):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p = dropout, inplace = True)
        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            nonlinearity = nonlinearity,
            bias = bias,
            batch_first = batch_first,
            # dropout = dropout,
            bidirectional = bidirectional
        )
        self.fc = nn.Linear(self.hidden_size, n_class)

    def forward(self, input):
        # input needs to be: (batch_size, seq, input_size)
        input = self.dropout(input, training=self.training)
        # Set initial hidden states
        if not self.bidirectional:
            h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        else:
            h0 = torch.zeros(2*self.num_layers, input.size(0), self.hidden_size)
        h0 = h0.to(input.device)
        out, _ = self.rnn(input, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = F.softmax(self.fc(out[:, -1, :]), dim = 1)
        return out
