#!/usr/bin/env python3
"""
This file contains classes to build RNN based classifiers

author: jy, nkmtmsys
"""
from fatez.model.rnn.rnn import Model as RNN
from fatez.model.rnn.gru import Model as GRU
from fatez.model.rnn.lstm import Model as LSTM

__all__ = [
    'RNN',
    'GRU',
    'LSTM'
]


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
