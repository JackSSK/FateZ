#!/usr/bin/env python3
"""
This file contains classes to build RNN based classifiers

author: jy, nkmtmsys
"""

import torch
import torch.nn as nn
import torch.nn.functional as func



class RNN(nn.Module):
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
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
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
        input = self.dropout(input)
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.rnn(input, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = func.softmax(self.fc(out[:, -1, :]), dim = 1)
        return out



class GRU(nn.Module):
    """
    Gated Recurrent Unit
    """
    def __init__(self,
        input_size:int = None,
        hidden_size:int = None,
        num_layers:int = 1,
        bias:bool = True,
        batch_first:bool = True,
        dropout:float = 0.0,
        bidirectional:bool = False,
        n_class:int = 2,
        **kwargs
        ):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias,
            batch_first = batch_first,
            # dropout = dropout,
            bidirectional = bidirectional
        )
        self.fc = nn.Linear(self.hidden_size, n_class)

    def forward(self, input):
        # input needs to be: (batch_size, seq, input_size)
        input = self.dropout(input)
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.gru(input, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = func.softmax(self.fc(out[:, -1, :]), dim = 1)
        return out



class LSTM(nn.Module):
    """
    Long-Short-Term Memory
    """
    def __init__(self,
        input_size:int = None,
        hidden_size:int = None,
        num_layers:int = 1,
        bias:bool = True,
        batch_first:bool = True,
        dropout:float = 0.0,
        bidirectional:bool = False,
        proj_size:int = 0,
        n_class:int = 2,
        **kwargs
        ):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias,
            batch_first = batch_first,
            # dropout = dropout,
            bidirectional = bidirectional,
            proj_size = proj_size,
        )
        self.fc = nn.Linear(self.hidden_size, n_class)

    def forward(self, input):
        # input needs to be: (batch_size, seq, input_size)
        input = self.dropout(input)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.lstm(input, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = func.softmax(self.fc(out[:, -1, :]), dim = 1)
        return out

# if __name__ == '__main__':
    # n_fea = 100
    # en_dim = 3
    # data = torch.randn(2, n_fea, en_dim)
    #
    # param_rnn = {
    #     'input_size': en_dim,
    #     'hidden_size': 32,
    #     'num_layers': 1,
    #     'nonlinearity': 'tanh',
    #     'bias': True,
    #     'batch_first': True,
    #     'dropout': 0.0,
    #     'bidirectional': False,
    # }
    # param_gru = {
    #     'input_size': en_dim,
    #     'hidden_size': 32,
    #     'num_layers': 1,
    #     'bias': True,
    #     'batch_first': True,
    #     'dropout': 0.0,
    #     'bidirectional': False,
    # }
    # param_lstm = {
    #     'input_size': en_dim,
    #     'hidden_size': 32,
    #     'num_layers': 1,
    #     'bias': True,
    #     'batch_first': True,
    #     'dropout': 0.0,
    #     'bidirectional': False,
    #     'proj_size': 0,
    # }
    # a = RNN(**param_rnn)
    # out = a(data)
    # print(out)
    # a = GRU(**param_gru)
    # out = a(data)
    # print(out)
    # a = LSTM(**param_lstm)
    # out = a(data)
    # print(out)
