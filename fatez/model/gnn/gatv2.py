#!/usr/bin/env python3
"""
GAT version 2 implemented with PyG.

author: jy
"""
import torch.nn as nn
import torch_geometric.nn as gnn
from fatez.model.gnn.gat import Model as Template



class Model(Template):
    """
    GAT version 2 implemented with PyG.
    """
    def __init__(self,
        input_sizes:dict = None,
        n_hidden:int = 3,
        en_dim:int = 2,
        nhead:int = 1,
        concat:bool = False,
        dropout:float = 0.0,
        n_layer_set:int = 1,
        **kwargs
        ):
        super().__init__(input_sizes)
        self.input_sizes = input_sizes
        self.en_dim = en_dim
        self.n_layer_set = n_layer_set

        model = list()
        # May take dropout layer out later
        model.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))

        if self.n_layer_set == 1:
            layer = gnn.GATv2Conv(
                in_channels = self.input_sizes['node_attr'],
                out_channels = en_dim,
                heads = nhead,
                dropout = dropout,
                edge_dim = self.input_sizes['edge_attr'],
                concat = concat,
                **kwargs
            )
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        elif self.n_layer_set > 1:
            layer = gnn.GATv2Conv(
                in_channels = self.input_sizes['node_attr'],
                out_channels = n_hidden,
                heads = nhead,
                dropout = dropout,
                edge_dim = self.input_sizes['edge_attr'],
                concat = concat,
                **kwargs
            )
            model.append((layer, 'x, edge_index, edge_attr -> x'))
            model.append(nn.ReLU(inplace = True))

            # Adding layer set
            for i in range(self.n_layer_set - 2):
                layer = gnn.GATv2Conv(
                    in_channels = n_hidden,
                    out_channels = n_hidden,
                    heads = nhead,
                    dropout = dropout,
                    edge_dim = self.input_sizes['edge_attr'],
                    concat = concat,
                    **kwargs
                )
                model.append((layer, 'x, edge_index, edge_attr -> x'))
                model.append(nn.ReLU(inplace = True))

            # Adding last layer
            layer = gnn.GATv2Conv(
                in_channels = n_hidden,
                out_channels = en_dim,
                heads = nhead,
                dropout = dropout,
                edge_dim = self.input_sizes['edge_attr'],
                concat = concat,
                **kwargs
            )
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        else:
            raise Exception('Why are we still here? Just to suffer.')

        self.model = gnn.Sequential('x, edge_index, edge_attr', model)