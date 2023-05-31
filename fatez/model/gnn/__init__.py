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



def Set(config:dict=None, input_sizes:dict=None, factory_kwargs:dict=None):
    """
    Set up GNN model based on given config.
    """
    # Init models accordingly
    if config['type'].upper() == 'GAT':
        return GAT(input_sizes, **config['params'], **factory_kwargs)
    elif config['type'].upper() == 'GATV2':
        return GATv2(input_sizes, **config['params'], **factory_kwargs)
    elif config['type'].upper() == 'GATVD':
        return GATvD(**config['params'], **factory_kwargs)
    else:
        raise Error('Unknown GNN type')


# if __name__ == '__main__':
#     import fatez as fz
#     from torch_geometric.datasets import TUDataset
#     from torch_geometric.loader import DataLoader
#     from torch_geometric.data import Data

    # dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    #
    # test_dataset = dataset[10:20]
    # train_loader = DataLoader(dataset[:10], batch_size = 2, shuffle = True)
    # sample = dataset[0]
    #
    # gatmodel = Model(
    #     d_model = 7, n_layer_set = 1, en_dim = 3, edge_dim = 4, device = 'cpu'
    # )
    #
    # for step, data, in enumerate(train_loader):
    #     result = gatmodel.model(data.x, data.edge_index, data.edge_attr)
    #     break

    # device = 'cpu'
    # faker = fz.test.Faker(device = device).make_data_loader()
    # gatmodel = Modelv2(
    #     d_model = 2, n_layer_set = 1, en_dim = 3, edge_dim = 1, device = device
    # )
    # # Process data in GPU
    # for x, y in faker:
    #     result = gatmodel(x[0].to(device), x[1].to(device))
    #     exp = gatmodel.explain(x[0][0].to(device), x[1][0].to(device))
    #     break
    # print(result.shape)
    # print(exp)
