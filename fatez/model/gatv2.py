#!/usr/bin/env python3
"""
This file contains Graph Attention Network (GAT) version 2 related objects.

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
from torch_geometric.explain import Explainer # ,AttentionExplainer
import fatez.lib as lib
import fatez.process.explainer as exp


class GAT(nn.Module):
    """
    A simple GAT using torch_geometric operator.
    """
    def __init__(self,
        d_model:int = 1,
        n_hidden:int = 3,
        en_dim:int = 2,
        nhead:int = 1,
        concat:bool = True,
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

        model_dict = OrderedDict([])
        # May take dropout layer out later
        model_dict.update({f'dp0': nn.Dropout(p=dropout, inplace=True)})

        if self.n_layer_set == 1:
            model_dict.update({f'conv0':gnn.GATConv(
                in_channels = d_model,
                out_channels = en_dim,
                heads = nhead,
                concat = concat,
                dropout = dropout,
            )})

        elif self.n_layer_set >= 1:
            model_dict.update({f'conv0':gnn.GATConv(
                in_channels = d_model,
                out_channels = n_hidden,
                heads = nhead,
                concat = concat,
                dropout = dropout,
            )})
            model_dict.update({f'relu0': nn.ReLU(inplace = True)})

            # Adding layer set
            for i in range(self.n_layer_set - 2):
                model_dict.update({f'conv{i+1}':gnn.GATConv(
                    in_channels = n_hidden,
                    out_channels = n_hidden,
                    heads = nhead,
                    concat = concat,
                    dropout = dropout,
                )})
                model_dict.update({f'relu{i+1}': nn.ReLU(inplace = True)})

            # Adding last layer
            model_dict.update({f'conv-1': gnn.GATConv(
                in_channels = n_hidden,
                out_channels = en_dim,
                heads = nhead,
                concat = concat,
                dropout = dropout,
            )})

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
            for i, layer in enumerate(self.model):
                if re.search(r'torch_geometric.nn.', str(type(layer))):
                    x = layer(x, edge_index, edge_weight)
                else:
                    x = layer(x)
            # Only take encoded presentations of TFs
            x = x[:adj_mat.sparse.shape[0],:]
            answer.append(x)
        answer = torch.stack(answer, 0)
        return answer


    def explain(self, fea_mat, adj_mat):
        """
        This part will very likely be revised due to developmental stage of
        torch_geometric.
        """
        x = fea_mat.to(self.factory_kwargs['device'])
        adjm = lib.Adj_Mat(sparse_mat=adj_mat.to(self.factory_kwargs['device']))
        edge_index, edge_weight = adjm.get_index_value()
        explainer = exp.AttentionExplainer(reduce = 'max')
        # explainer = Explainer(
        #     model = self.model,
        #     algorithm =  exp.AttentionExplainer(reduce = 'max'),
        #     # AttentionExplainer not using target anyway
        #     explainer_config = 'model',
        #     node_mask_type='object',
        #     model_config = dict(
        #         mode = 'multiclass_classification',
        #         task_level = 'graph',
        #         return_type = 'probs',  # Model returns probabilities.
        #     ),
        # )

        result = explainer.test(
            model = self.model[0],
            x = x,
            edge_index = edge_index,
            edge_attr = edge_weight,
            target = None,
        )
        print(result)
        return


if __name__ == '__main__':
    # from torch_geometric.datasets import TUDataset
    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    # data = dataset[0]
    # train_dataset = dataset[:5]

    import fatez as fz
    device = 'cuda'
    faker = fz.test.Faker(device = 'cuda').make_data_loader()
    model = GAT(d_model = 2, n_layer_set = 1, en_dim = 3, device = 'cuda')
    for x, y in faker:
        fea = x[0].to(device)
        adj = x[1].to(device)
        result = model(fea, adj)
        exp = model.explain(fea[0], adj[0])
        break
    print(result)
