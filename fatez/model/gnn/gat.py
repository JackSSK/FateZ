#!/usr/bin/env python3
"""
A simple GAT using torch_geometric operator.

author: jy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
# from torch_geometric.explain import Explainer, AttentionExplainer
from torch_geometric.nn.conv.message_passing import MessagePassing
import fatez.lib as lib



class Model(nn.Module):
    """
    A simple GAT using torch_geometric operator.
    """
    def __init__(self,
        input_sizes:dict = None,
        n_hidden:int = 3,
        en_dim:int = 2,
        nhead:int = 1,
        concat:bool = False,
        dropout:float = 0.0,
        n_layer_set:int = 1,
        device:str = 'cpu',
        dtype:str = None,
        **kwargs
        ):
        """
        :param input_sizes:dict = None
            Key dimensions indicating shapes of input matrices.

        :param n_hidden:int = None
            Number of hidden units.

        :param en_dim:int = 2
            Number of each gene's encoded features.

        :param nhead:int = None
            Number of attention heads.

        :param lr:float = 0.005
            Learning rate.

        :param weight_decay:float = 5e-4
            Weight decay (L2 loss on parameters).

        :param dropout:float = 0.2
            Dropout rate.

        :param alpha:float = 0.2
            Alpha value for the LeakyRelu layer.

        :param dtype:str = None
            The dtype of values in matrices.
            Note: torch default using float32, numpy default using float64
        """
        super().__init__()
        self.input_sizes = input_sizes
        self.en_dim = en_dim
        self.n_layer_set = n_layer_set
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        model = list()
        # May take dropout layer out later
        model.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))

        if self.n_layer_set == 1:
            layer = gnn.GATConv(
                in_channels = self.input_sizes['node_attr'],
                out_channels = en_dim,
                heads = nhead,
                dropout = dropout,
                concat = concat,
                **kwargs
            )
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        elif self.n_layer_set > 1:
            layer = gnn.GATConv(
                in_channels = self.input_sizes['node_attr'],
                out_channels = n_hidden,
                heads = nhead,
                dropout = dropout,
                concat = concat,
                **kwargs
            )
            model.append((layer, 'x, edge_index, edge_attr -> x'))
            model.append(nn.ReLU(inplace = True))

            # Adding layer set
            for i in range(self.n_layer_set - 2):
                layer = gnn.GATConv(
                    in_channels = n_hidden,
                    out_channels = n_hidden,
                    heads = nhead,
                    dropout = dropout,
                    concat = concat,
                    **kwargs
                )
                model.append((layer, 'x, edge_index, edge_attr -> x'))
                model.append(nn.ReLU(inplace = True))

            # Adding last layer
            layer = gnn.GATConv(
                in_channels = n_hidden,
                out_channels = en_dim,
                heads = nhead,
                dropout = dropout,
                concat = concat,
                **kwargs
            )
            model.append((layer, 'x, edge_index, edge_attr -> x'))

        else:
            raise Exception('Why are we still here? Just to suffer.')

        self.model = gnn.Sequential('x, edge_index, edge_attr', model)
        self.model = self.model.to(self.factory_kwargs['device'])

    def forward(self, fea_mats, edge_indices, edge_attrs):
        answer = list()
        # Process batch data
        for i in range(len(fea_mats)):
            rep = self.model(
                fea_mats[i].to(self.factory_kwargs['device']),
                edge_indices[i].to(self.factory_kwargs['device']),
                edge_attrs[i].to(self.factory_kwargs['device']),
            )
            # Only taking regulon representations
            answer.append(self._get_regulon_exp(rep, edge_indices[i]))
        return torch.stack(answer, 0)

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
            elif getattr(module, '_alpha', None) != None:
                alphas.append(module._alpha.detach())

        # Register message forward hooks
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                hook_handles.append(module.register_message_forward_hook(hook))
        # Feed data in to the model.
        edge_index, edge_attr = self._get_index_weight(adj_mat)
        rep = self.model(fea_mat, edge_index, edge_attr)
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

        return lib.Adj_Mat(
            indices = edge_index,
            values = F.softmax(alpha.detach().squeeze(-1), dim=-1),
            shape = adj_mat.shape[:2]
        ).to_dense()

    def switch_device(self, device = 'cpu'):
        self.factory_kwargs['device'] = device
        self.model = self.model.to(device)
        return

    def _get_index_weight(self, adj_mat):
        """
        Make edge index and edge weight matrices based on given adjacent matrix.
        """
        return lib.Adj_Mat(adj_mat.to(self.factory_kwargs['device'])).unpack()

    def _get_regulon_exp(self, rep, edge_index):
        """
        Make regulon representations according to node reps and adjacent matrix.
        """
        device = self.factory_kwargs['device']
        # Get pooling data based on adj mat
        iter = 0
        pooling_data = list()
        for i in range(self.input_sizes['n_reg']):
            batch = torch.zeros(len(rep), dtype=torch.int64).to(device)
            while iter < len(edge_index[0]) and edge_index[0][iter] <= i:
                if edge_index[0][iter] == i:
                    batch[edge_index[1][iter]] += 1
                iter += 1
            pooling_data.append(gnn.pool.global_add_pool(rep, batch = batch)[1])
        # Product pooling data to according node(TF) representations
        answer = rep[:self.input_sizes['n_reg'],:]
        assert len(answer) == len(pooling_data)
        for i,data in enumerate(pooling_data):
            answer[i] *= data
        return answer
