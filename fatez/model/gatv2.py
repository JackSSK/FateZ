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
# from torch_geometric.explain import Explainer, AttentionExplainer
from torch_geometric.nn.conv.message_passing import MessagePassing
import fatez.lib as lib



class Model(nn.Module):
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
            edge_index, edge_weight = self._get_index_weight(adj_mats[i])
            rep = self._feed_model(fea_mats[i], edge_index, edge_weight)
            # Only take encoded presentations of TFs
            answer.append(rep[:adj_mats[i].shape[0],:])
        answer = torch.stack(answer, 0)
        return answer

    def explain(self, fea_mat, adj_mat, reduce = 'sum'):
        """
        This function will very likely be revised due to developmental stage of
        torch_geometric.
        """
        # Assert we only get Tensors for alpha matrices
        alphas: List[torch.Tensor] = list()
        hook_handles = list()

        def hook(module, msg_kwargs, out):
            """
            Set up hook for extracting alpha values from layers
            """
            if 'alpha' in msg_kwargs[0]:
                alphas.append(msg_kwargs[0]['alpha'].detach())
            elif getattr(module, '_alpha', None) is not None:
                alphas.append(module._alpha.detach())

        # Register message forward hooks
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                hook_handles.append(module.register_message_forward_hook(hook))
        # Feed data in to the model.
        edge_index, edge_weight = self._get_index_weight(adj_mat)
        rep = self._feed_model(fea_mat, edge_index, edge_weight)
        # Remove all the hooks
        del hook_handles

        for i, alpha in enumerate(alphas):
             # Respect potential self-loops.
            alpha = alpha[:edge_index.size(1)]
            # Reducing dimension
            if alpha.dim() == 2:
                alpha = getattr(torch, reduce)(alpha, dim=-1)
                if isinstance(alpha, tuple):  # Respect `torch.max`:
                    alpha = alpha[0]
            elif alpha.dim() > 2:
                raise ValueError(
                    f"Can not reduce attention coefficients of "
                    f"shape {list(alpha.size())}"
                )
            alphas[i] = alpha

        # Reducing dimension
        if len(alphas) > 1:
            alpha = torch.stack(alphas, dim=-1)
            alpha = getattr(torch, reduce)(alpha, dim=-1)
            if isinstance(alpha, tuple):  # Respect `torch.max`:
                alpha = alpha[0]
        else:
            alpha = alphas[0]

        answer = func.softmax(alpha.detach().squeeze(-1))
        return answer.reshape(adj_mat.shape)

    def _get_index_weight(self, adj_mat):
        """
        Make edge index and edge weight matrices based on given adjacent matrix.
        """
        x = lib.Adj_Mat(sparse_mat = adj_mat.to(self.factory_kwargs['device']))
        return x.get_index_value()

    def _feed_model(self, fea_mat, edge_index, edge_weight):
        """
        Feed in data to the model.
        """
        x = fea_mat.to(self.factory_kwargs['device'])
        # Feed into model
        for i, layer in enumerate(self.model):
            if re.search(r'torch_geometric.nn.', str(type(layer))):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x



if __name__ == '__main__':
    # from torch_geometric.datasets import TUDataset
    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    # data = dataset[0]
    # train_dataset = dataset[:5]

    import fatez as fz
    device = 'cuda'
    faker = fz.test.Faker(device = 'cuda').make_data_loader()
    model = Model(d_model = 2, n_layer_set = 1, en_dim = 3, device = 'cuda')
    for x, y in faker:
        fea = x[0].to(device)
        adj = x[1].to(device)
        result = model(fea, adj)

        exp = model.explain(fea[0], adj[0])
        break
    print(result)
    print(exp)
