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



def Set(config:dict=None, input_sizes:list=None, factory_kwargs:dict=None):
    """
    Set up GNN model based on given config.
    """
    # Get edge dim
    if len(input_sizes[1]) == 3:
        edge_dim = 1
    elif len(input_sizes[1]) == 4:
        edge_dim = input_sizes[1][-1]
    else:
        raise Error('Why are we still here? Just to suffer.')
    if 'edge_dim' in config['params']:
        assert config['params']['edge_dim'] == edge_dim
    else:
        config['params']['edge_dim'] = edge_dim

    # Get d_model
    # if 'd_model' in config['params']:
    #     assert config['params']['d_model'] == input_sizes[0][-1]
    # else:
    #     config['params']['d_model'] = input_sizes[0][-1]

    # Init models accordingly
    if config['type'].upper() == 'GAT':
        return GAT(**config['params'], **factory_kwargs)
    elif config['type'].upper() == 'GATV2':
        return GATv2(**config['params'], **factory_kwargs)
    elif config['type'].upper() == 'GATVD':
        return GATvD(**config['params'], **factory_kwargs)
    else:
        raise Error('Unknown GNN type')
