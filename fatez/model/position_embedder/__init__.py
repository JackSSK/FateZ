#!/usr/bin/env python3
"""
Modules for positional embedding.
Both trainable and untrainable methods are here.

author: jy

ToDo:
Randowm Walking PE not done yet.
Pos Embed before GNN or after GNN not decided yet:
    Current version embed after GNN since GNN should be pos sensitive
"""
from fatez.model.position_embedder.skip import Embedder as Skip
from fatez.model.position_embedder.logsoftmax import Embedder as LSME
from fatez.model.position_embedder.absolute import Embedder as Absolute_Embed

__all__ = [
    'Skip',
    'Absolute_Embed',
    'LogSoftmax'
]

def Set(config:dict=None, input_sizes:list=None, dtype:str = None, **kwargs):
    """
    Set up positional embedder based on given config.
    """
    if config['type'].upper() == 'SKIP':
        return Skip()
    elif config['type'].upper() == 'ABS':
        return Absolute_Embed(**config['params'], dtype = dtype)
    elif config['type'].upper() == 'LSME':
        return LSME(**config['params'], dtype = dtype)
    elif config['type'].upper() == 'RW':
        return
    else:
        raise model.Error(f'Unknown rep_embedder type')
