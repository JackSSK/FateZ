#!/usr/bin/env python3
"""
This file contains classes to build RNN based classifiers

ToDo:
Implement _earlystopping

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
        input_size,
        num_layers,
        hidden_size,
        dropout,
        n_class = 2,
        nonlinearity = 'tanh',
        bias = 'True',
        bidirectional = 'False',
        ):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
        self.rnn = nn.RNN(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first = True,
            nonlinearity = nonlinearity,
            bias = bias,
            bidirectional = bidirectional
        )
        self.fc = nn.Linear(self.hidden_size, n_class)

    def forward(self, input):
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
        input_size,
        num_layers,
        hidden_size,
        dropout,
        n_class = 2,
        bias = 'True',
        bidirectional = 'False',
        ):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
        self.gru = nn.GRU(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first = True,
            bias = bias,
            bidirectional = bidirectional
        )
        self.fc = nn.Linear(self.hidden_size, n_class)

    def forward(self, input):
        input = self.dropout(input)
        # Set initial hidden states
        # input needs to be: (batch_size, seq, input_size)
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
        input_size,
        num_layers,
        hidden_size,
        dropout,
        n_class = 2,
        bias = 'True',
        bidirectional = 'False',
        proj_size = 0,
        ):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first = True,
            bias = bias,
            bidirectional = bidirectional,
            proj_size = proj_size
        )
        self.fc = nn.Linear(self.hidden_size, n_class)

    def forward(self, input):
        input = self.dropout(input)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        # Forward propagate LSTM
        out, _ = self.lstm(input, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = func.softmax(self.fc(out[:, -1, :]), dim = 1)
        return out
