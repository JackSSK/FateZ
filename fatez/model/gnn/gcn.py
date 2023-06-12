#!/usr/bin/env python3
"""
A simple GCN using torch_geometric operator.

author: jy
"""
import torch
import torch.nn as nn
import torch_geometric.nn as pyg
from fatez.model.gnn.gat import Model as Template



class Model(Template):
    """
    A simple GCN using torch_geometric operator.
    """
    def __init__(self,
        input_sizes:dict = None,
        n_hidden:int = 3,
        en_dim:int = 2,
        dropout:float = 0.0,
        n_layer_set:int = 1,
        device:str = 'cpu',
        dtype:str = None,
        **kwargs
        ):
        # Initialization
        super().__init__(input_sizes)
        self.input_sizes = input_sizes
        self.en_dim = en_dim
        self.n_layer_set = n_layer_set

        model = list()
        # May take dropout layer out later
        model.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))

        if self.n_layer_set == 1:
            layer = pyg.GCNConv(
                in_channels = self.input_sizes['node_attr'],
                out_channels = en_dim
            )
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        elif self.n_layer_set >= 1:
            layer = pyg.GCNConv(self.input_sizes['node_attr'], n_hidden)
            model.append((layer, 'x, edge_index, edge_attr -> x'))
            model.append(nn.ReLU(inplace = True))

            # Adding Conv blocks
            for i in range(self.n_layer_set - 2):
                layer = pyg.GCNConv(n_hidden, n_hidden)
                model.append((layer, 'x, edge_index, edge_attr -> x'))
                model.append(nn.ReLU(inplace = True))

            # Adding last layer
            layer = pyg.GCNConv(n_hidden, en_dim)
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        else:
            raise Exception('Why are we still here? Just to suffer.')

        self.model = pyg.Sequential('x, edge_index, edge_attr', model)

    def explain(self, fea_mat, adj_mat,):
        return



# if __name__ == '__main__':
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
