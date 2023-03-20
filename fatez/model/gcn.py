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
import torch.nn.functional as func
from collections import OrderedDict
from torch_geometric.data import Data
import fatez.lib as lib


class GCN(nn.Module):
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
        self.n_layer_set = n_layer_set
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        model_dict = OrderedDict([('dp0', nn.Dropout(p=dropout, inplace=True))])

        if self.n_layer_set == 1:
            model_dict.update({
                f'conv-1':gnn.GCNConv(in_channels=d_model, out_channels=en_dim)
            })

        elif self.n_layer_set >= 1:
            model_dict.update({f'conv0':gnn.GCNConv(d_model, n_hidden)})
            model_dict.update({f'relu0': nn.ReLU(inplace = True)})

            # Adding Conv blocks
            for i in range(self.n_layer_set - 2):
                model_dict.update({f'conv{i+1}':gnn.GCNConv(n_hidden,n_hidden)})
                model_dict.update({f'relu{i+1}': nn.ReLU(inplace = True)})

            # Adding last layer
            model_dict.update({f'conv-1': gnn.GCNConv(n_hidden, en_dim)})

        else:
            raise Exception('Why are we still here? Just to suffer.')

        self.model = nn.Sequential(model_dict).to(self.factory_kwargs['device'])


    def forward(self, fea_mats, adj_mats):
        answer = list()
        assert len(fea_mats) == len(adj_mats)
        for i in range(len(fea_mats)):
            # Process batch data
            x = fea_mats[i].to(self.factory_kwargs['device'])
            adj_mat = lib.Adj_Mat(
                sparse_mat = adj_mats[i].to(self.factory_kwargs['device'])
            )
            edge_index, edge_weight = adj_mat.get_index_value()
            # Feed into model
            for layer in self.model:
                if re.search(r'torch_geometric.nn.', str(type(layer))):
                    x = layer(x, edge_index, edge_weight)
                else:
                    x = layer(x)
            answer.append(x)
        answer = torch.stack(answer, 0)
        return answer



if __name__ == '__main__':
    # from torch_geometric.datasets import TUDataset
    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    # data = dataset[0]
    # train_dataset = dataset[:5]

    # import fatez as fz
    # faker = fz.test.Faker(device = 'cuda').make_data_loader()
    # gcn = GCN(d_model = 2, n_layer_set = 2, en_dim = 3, device = 'cuda')
    # for x, y in faker:
    #     result = gcn(x[0].to('cuda'), x[1].to('cuda'))
    #     break
    # print(result)
