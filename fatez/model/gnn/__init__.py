#!/usr/bin/env python3
"""
Graph Neural Network related objects and functions.

author: jy
"""
from fatez.model.gnn.gcn import Model as GCN
from fatez.model.gnn.gat import Model as GAT
from fatez.model.gnn.gatv2 import Model as GATv2
from fatez.model.gnn.gatvd import Model as GATvD

__all__ = [
    'GCN',
    'GAT',
    'GATv2',
    'GATvD',
]

class Error(Exception):
    pass



def Set(config:dict=None, input_sizes:dict=None, dtype:str = None, **kwargs):
    """
    Set up GNN model based on given config.
    """
    # Init models accordingly
    if config['type'].upper() == 'GAT':
        return GAT(input_sizes, **config['params'],)
    elif config['type'].upper() == 'GATV2':
        return GATv2(input_sizes, **config['params'],)
    elif config['type'].upper() == 'GATVD':
        return GATvD(input_sizes, **config['params'],)
    else:
        raise Error('Unknown GNN type')
