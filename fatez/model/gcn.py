#!/usr/bin/env python3
"""
This file contains Graph Convolutional Networks (GCN) related objects.

author: jy
"""
import re
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from torch_geometric.data import Data
import fatez.lib as lib
import fatez.model.gat as gat



class Model(gat.Module):
    """
    A simple GCN using torch_geometric operator.
    """
    def __init__(self,
        d_model:int = 1,
        n_hidden:int = 3,
        en_dim:int = 2,
        dropout:float = 0.0,
        n_layer_set:int = 1,
        device:str = 'cpu',
        dtype:str = None,
        **kwargs
        ):
        # Initialization
        super().__init__()
        self.d_model = d_model
        self.en_dim = en_dim
        self.n_layer_set = n_layer_set
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        model = list()
        # May take dropout layer out later
        model.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))

        if self.n_layer_set == 1:
            layer = gnn.GCNConv(in_channels=d_model, out_channels=en_dim)
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        elif self.n_layer_set >= 1:
            layer = gnn.GCNConv(d_model, n_hidden)
            model.append((layer, 'x, edge_index, edge_attr -> x'))
            model.append(nn.ReLU(inplace = True))

            # Adding Conv blocks
            for i in range(self.n_layer_set - 2):
                layer = gnn.GCNConv(n_hidden,n_hidden)
                model.append((layer, 'x, edge_index, edge_attr -> x'))
                model.append(nn.ReLU(inplace = True))

            # Adding last layer
            layer = gnn.GCNConv(n_hidden, en_dim)
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        else:
            raise Exception('Why are we still here? Just to suffer.')

        self.model = gnn.Sequential('x, edge_index, edge_attr', model)
        self.model = self.model.to(self.factory_kwargs['device'])

    def explain(self, fea_mat, adj_mat,):
        return





if __name__ == '__main__':
    # from torch_geometric.datasets import TUDataset
    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    # data = dataset[0]
    # train_dataset = dataset[:5]

    # import fatez as fz
    # faker = fz.test.Faker(device = 'cuda').make_data_loader()
    # gcn = Model(d_model = 2, n_layer_set = 1, en_dim = 3, device = 'cuda')
    # for x, y in faker:
    #     result = gcn(x[0].to('cuda'), x[1].to('cuda'))
    #     break
    # print(result)
