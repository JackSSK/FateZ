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

        self.model = nn.Sequential(model_dict)


    def forward(self, fea_mats, adj_mats):
        x = x.to(self.factory_kwargs['device'])
        edge_index = edge_index.to(self.factory_kwargs['device'])
        edge_weight = edge_weight.to(self.factory_kwargs['device'])
        answer = list()
        assert len(ind_mats) == len(val_mats) and len(fea_mats) == len(val_mats)
        for i in range(len(fea_mats)):
            x = fea_mats[i].to(self.factory_kwargs['device'])
            edge_index = ind_mats[i].to(self.factory_kwargs['device'])
            edge_weight = val_mats[i].to(self.factory_kwargs['device'])
        print(x.shape, edge_index.shape)
        for layer in self.model:
            if re.search(r'torch_geometric.nn.', str(type(layer))):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x
        # return func.softmax(x, dim = -1)



if __name__ == '__main__':
    # from torch_geometric.datasets import TUDataset

    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    # data = dataset[0]
    # train_dataset = dataset[:5]



    t = torch.tensor([[0., -1, 0], [2., 0., 1]])
    adj_mat = Adj_Mat(t)

    a = torch.Tensor([ [[0,0], [0,0]], [[1,2], [0,0]] ])
    adj_mat = Adj_Mat(a)

    edge_index = torch.tensor([[0, 1, 1,], [1, 0, 2,]], dtype = torch.long)
    edge_weight = torch.tensor([[-1], [2], [1]], dtype=torch.float)
    adj_mat = Adj_Mat(indices=edge_index, values=edge_weight, size=(2,3,1))

    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    # data = Data(x=x, edge_index=adj_mat.indices, edge_weight=adj_mat.values, shape = adj_mat.size)
    # print(data.shape)
    # data.validate(raise_on_error=True)

    a = GCN(n_layer_set = 1, device = 'cuda').to('cuda')
    print(a(data.x, adj_mat.sparse))
