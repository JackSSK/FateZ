#!/usr/bin/env python3
"""
This file contains Graph Attention Network (GAT) related objects.

ToDo:
Would JAX help on speeding up model training?
Added back the GAT implementation with dense matrices, maybe a reimplementation
in JAX could help, but probably we shall wait till GRN preprocess part fixed and
tested.

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
# from torch_geometric.explain import Explainer, AttentionExplainer
from torch_geometric.nn.conv.message_passing import MessagePassing
import fatez.lib as lib



def Set(config:dict=None, input_sizes:list=None, factory_kwargs:dict=None):
    """
    Set up GAT model based on given config.
    """
    # Get edge dim
    if len(input_sizes[1]) == 3:
        edge_dim = 1
    elif len(input_sizes[1]) == 4:
        edge_dim = input_sizes[1][-1]
    else:
        raise Exception('Why are we still here? Just to suffer.')
    if 'edge_dim' in config['params']:
        assert config['params']['edge_dim'] == edge_dim
    else:
        config['params']['edge_dim'] = edge_dim

    # Get d_model
    if 'd_model' in config['params']:
        assert config['params']['d_model'] == input_sizes[0][-1]
    else:
        config['params']['d_model'] = input_sizes[0][-1]

    # Init models accordingly
    if config['type'].upper() == 'GAT':
        return Model(**config['params'], **factory_kwargs)
    elif config['type'].upper() == 'GATV2':
        return Modelv2(**config['params'], **factory_kwargs)
    elif config['type'].upper() == 'GATVD':
        return ModelvD(**config['params'], **factory_kwargs)
    else:
        raise model.Error('Unknown GAT type')



class Model(nn.Module):
    """
    A simple GAT using torch_geometric operator.
    """
    def __init__(self,
        d_model:int = 1,
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
        # Initialization
        super().__init__()
        self.d_model = d_model
        self.en_dim = en_dim
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
                dropout = dropout,
                concat = concat,
                **kwargs
            )})

        elif self.n_layer_set >= 1:
            model_dict.update({f'conv0':gnn.GATConv(
                in_channels = d_model,
                out_channels = n_hidden,
                heads = nhead,
                dropout = dropout,
                concat = concat,
                **kwargs
            )})
            model_dict.update({f'relu0': nn.ReLU(inplace = True)})

            # Adding layer set
            for i in range(self.n_layer_set - 2):
                model_dict.update({f'conv{i+1}':gnn.GATConv(
                    in_channels = n_hidden,
                    out_channels = n_hidden,
                    heads = nhead,
                    dropout = dropout,
                    concat = concat,
                    **kwargs
                )})
                model_dict.update({f'relu{i+1}': nn.ReLU(inplace = True)})

            # Adding last layer
            model_dict.update({f'conv-1': gnn.GATConv(
                in_channels = n_hidden,
                out_channels = en_dim,
                heads = nhead,
                dropout = dropout,
                concat = concat,
                **kwargs
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
            elif getattr(module, '_alpha', None) != None:
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

        x = F.softmax(alpha.detach().squeeze(-1), dim=-1).reshape(adj_mat.shape)
        return x

    def _get_index_weight(self, adj_mat):
        """
        Make edge index and edge weight matrices based on given adjacent matrix.
        """
        x = lib.Adj_Mat(adj_mat.to(self.factory_kwargs['device']))
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



class Modelv2(Model):
    """
    A simple GAT using torch_geometric operator.
    """
    def __init__(self,
        d_model:int = 1,
        n_hidden:int = 3,
        en_dim:int = 2,
        nhead:int = 1,
        concat:bool = False,
        dropout:float = 0.0,
        edge_dim:int = 1,
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

        model_dict = OrderedDict([])
        # May take dropout layer out later
        model_dict.update({f'dp0': nn.Dropout(p=dropout, inplace=True)})

        if self.n_layer_set == 1:
            model_dict.update({f'conv0':gnn.GATv2Conv(
                in_channels = d_model,
                out_channels = en_dim,
                heads = nhead,
                dropout = dropout,
                edge_dim = edge_dim,
                concat = concat,
                **kwargs
            )})

        elif self.n_layer_set >= 1:
            model_dict.update({f'conv0':gnn.GATv2Conv(
                in_channels = d_model,
                out_channels = n_hidden,
                heads = nhead,
                dropout = dropout,
                edge_dim = edge_dim,
                concat = concat,
                **kwargs
            )})
            model_dict.update({f'relu0': nn.ReLU(inplace = True)})

            # Adding layer set
            for i in range(self.n_layer_set - 2):
                model_dict.update({f'conv{i+1}':gnn.GATv2Conv(
                    in_channels = n_hidden,
                    out_channels = n_hidden,
                    heads = nhead,
                    dropout = dropout,
                    edge_dim = edge_dim,
                    concat = concat,
                    **kwargs
                )})
                model_dict.update({f'relu{i+1}': nn.ReLU(inplace = True)})

            # Adding last layer
            model_dict.update({f'conv-1': gnn.GATv2Conv(
                in_channels = n_hidden,
                out_channels = en_dim,
                heads = nhead,
                dropout = dropout,
                edge_dim = edge_dim,
                concat = concat,
                **kwargs
            )})

        else:
            raise Exception('Why are we still here? Just to suffer.')

        self.model = nn.Sequential(model_dict).to(self.factory_kwargs['device'])



def _prepare_attentions(e_values, adj_mat):
    """
    Calculate attention values for every GRP.

    :param e_values:torch.Tensor = None
        Weighted matrix.

    :param adj_mat:torch.Tensor = None
        Adjacent matrix. (Based on GRPs)
    """
    # Basically, this is a matrix of negative infinite with e_values.shape
    # neg_inf = torch.zeros_like(e_values)
    # neg_inf = neg_inf.masked_fill(neg_inf == 0, float('-inf'))
    # Left confirmed GRPs only.
    attention = torch.where(adj_mat != 0, e_values, torch.zeros_like(e_values))
    # Multiply GRP coefficient to the attention values
    attention = np.multiply(
        attention.detach().cpu().numpy(),
        adj_mat.detach().cpu()
    )
    # Change 0s to neg inf now
    attention = attention.masked_fill(attention == 0, float(-9e15))
    attention = F.softmax(attention, dim = 1)
    return attention

class Graph_Attention_Layer(nn.Module):
    """
    Simple graph attention layer for GAT.
    """
    def __init__(self,
        d_model:int = None,
        out_dim:int = None,
        lr:float = 0.005,
        weight_decay:float = 5e-4,
        gain:float = 1.414,
        dropout:float = 0.2,
        alpha:float = 0.2,
        concat:bool = True,
        device:str = 'cpu',
        dtype:str = None,
        ):
        """
        :param d_model:int = None
            Input feature dimension.

        :param out_dim:int = None
            Output feature dimension.

        :param lr:float = 0.005
            Learning rate.

        :param weight_decay:float = 5e-4
            Weight decay (L2 loss on parameters).

        :param dropout:float = 0.2
            Dropout rate.

        :param alpha:float = 0.2
            Alpha value for the LeakyRelu layer.

        :param concat:bool = True
            Whether concatenating with other layers.
            Note: False for last layer.

        :param dtype:str = None
            Data type of values in matrices.
            Note: torch default using float32, numpy default using float64
        """
        super(Graph_Attention_Layer, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat
        # Set up parameters
        self.weights = nn.Parameter(
            torch.empty(size = (d_model, out_dim), device = device, dtype=dtype)
        )
        self.a_values = nn.Parameter(
            torch.empty(size = (2 * out_dim, 1), device = device, dtype = dtype)
        )
        nn.init.xavier_uniform_(self.weights.data, gain = gain)
        nn.init.xavier_uniform_(self.a_values.data, gain = gain)

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.optimizer = optim.Adam(
            self.parameters(),
            lr = lr,
            weight_decay = weight_decay
        )
        self.device = device

    def _prepare_attentional_mechanism_input(self, w_h, n_regulons):
        """
        Calculate e values according to input matrix and weights.

        :param w_h:torch.Tensor = None
            Weighted matrix.
        """
        # w_h1.shape (n_regulons, 1) and w_h2.shape (N, 1)
        # e.shape (n_regulons, N)

        w_h1 = torch.matmul(w_h[:n_regulons,:], self.a_values[:self.out_dim, :])
        w_h2 = torch.matmul(w_h, self.a_values[self.out_dim:, :])
        # broadcast add and return
        return self.leaky_relu(w_h1 + w_h2.T)

    def forward(self, input, adj_mat):
        """
        :param input:torch.Tensor = None
            Input matrix. (Genes)

        :param adj_mat:torch.Tensor = None
            Adjacent matrix. (Based on GRPs)
        """
        # Multiply hs to ensure output dimension == out_dim
        w_h = torch.mm(input, self.weights)

        # w_h.shape == (N, out_feature)
        # Now we calculate weights for GRPs.
        e_values = self._prepare_attentional_mechanism_input(
            w_h = w_h, n_regulons = adj_mat.size()[0]
        )
        attention = _prepare_attentions(e_values, adj_mat)
        attention = F.dropout(attention, self.dropout, training = self.training)
        result = torch.matmul(attention.to(self.device), w_h)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(result)
        else:
            # if this layer is last layer,
            return result

class ModelvD(nn.Module):
    """
    A typical GAT but using dense matrix instead of sparse.
    This shall be used if the PyG sparse version is just too slow.
    """
    def __init__(self,
        d_model:int = None,
        n_hidden:int = 1,
        en_dim:int = 2,
        nhead:int = None,
        lr:float = 0.005,
        weight_decay:float = 5e-4,
        dropout:float = 0.2,
        alpha:float = 0.2,
        device:str = 'cpu',
        dtype:str = None,
        **kwargs
        ):
        """
        :param d_model:int = None
            Number of each gene's input features.

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
            Data type of values in matrices.
            Note: torch default using float32, numpy default using float64
        """
        super(ModelvD, self).__init__()
        self.d_model = d_model
        self.n_hidden = n_hidden
        self.en_dim = en_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.attentions = None
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # Add attention heads
        if nhead != None and nhead > 0:
            self.attentions = [
                Graph_Attention_Layer(
                    d_model = d_model,
                    out_dim = n_hidden,
                    lr = lr,
                    weight_decay = weight_decay,
                    dropout = dropout,
                    alpha = alpha,
                    concat = True,
                    **self.factory_kwargs,
                ) for _ in range(nhead)
            ]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            # Change input dimension for last GAT layer
            d_model = n_hidden * nhead

        # Also, we can add other layers here.

        # Last output GAT layer
        self.last = Graph_Attention_Layer(
            d_model = d_model,
            out_dim = en_dim,
            lr = lr,
            weight_decay = weight_decay,
            dropout = dropout,
            alpha = alpha,
            concat = False,
            **self.factory_kwargs,
        )

    def forward(self, fea_mats, adj_mats,):
        """
        :param fea_mats: torch.Tensor
            Feature matrices. (Genes)
        :param adj_mats: torch.Tensor
            Adjacent matrices. (Based on GRPs)
        """
        answer = list()
        assert len(fea_mats) == len(adj_mats)
        for i in range(len(fea_mats)):
            x = fea_mats[i].to(self.factory_kwargs['device']).to_dense()
            adj_mat = adj_mats[i].to(self.factory_kwargs['device']).to_dense()
            x = F.dropout(x, self.dropout, training = self.training)
            # Multi-head attention mechanism
            if self.attentions != None:
                x = torch.cat([a(x, adj_mat) for a in self.attentions], dim = 1)
                x = F.dropout(x, self.dropout, training = self.training)
                # Resize the adj_mat to top_k rows
                x = F.elu(self.last(x, adj_mat.narrow(1, 0, adj_mat.size()[0])))
            else:
                x = F.elu(self.last(x, adj_mat))
            answer.append(x)
        answer = torch.stack(answer, 0)
        return answer

    def explain(self, fea_mat, adj_mat):
        """
        Input real data, then this func is explaining in real case.
        For general explanation, just make a new mat with all values == 1.
        fake_fea_mat = torch.ones_like(fea_mat)
        fake_adj_mat = torch.ones_like(adj_mat)
        """
        fea_mat = fea_mat.to(self.factory_kwargs['device']).to_dense()
        adj_mat = adj_mat.to(self.factory_kwargs['device']).to_dense()

        att_explain = None
        last_explain = None

        def Get_Attention(fea_mat, adj_mat, weights, a_values, out_dim):
            w_h = torch.mm(fea_mat, weights)
            w_h1 = torch.matmul(w_h[:adj_mat.size()[0],:], a_values[:out_dim,:])
            w_h2 = torch.matmul(w_h, a_values[out_dim:, :])
            e_values = F.leaky_relu(w_h1 + w_h2.T)
            return _prepare_attentions(e_values, adj_mat)

        if self.attentions != None:
            weights = None
            a_values = None
            for att in self.attentions:
                if a_values is None:
                    a_values = att.a_values.detach()
                else:
                    a_values += att.a_values.detach()

                if weights is None:
                    weights = att.weights.detach()
                else:
                    weights += att.weights.detach()

            att_explain = Get_Attention(
                fea_mat,
                adj_mat,
                weights,
                a_values,
                out_dim = self.n_hidden,
            )

            last_explain = Get_Attention(
                torch.cat(
                    [a.eval()(fea_mat, adj_mat) for a in self.attentions],
                    dim = 1
                ),
                adj_mat.narrow(1, 0, adj_mat.size()[0]),
                self.last.weights,
                self.last.a_values,
                out_dim = self.en_dim,
            )
        else:
            last_explain = Get_Attention(
                fea_mat,
                adj_mat,
                self.last.weights,
                self.last.a_values,
                out_dim = self.en_dim,
            )

        # Not using multihead attention
        if att_explain is None:
            return last_explain
        else:
            return torch.matmul(last_explain, att_explain)



# if __name__ == '__main__':
    # from torch_geometric.datasets import TUDataset
    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    # data = dataset[0]
    # train_dataset = dataset[:5]

    # import fatez as fz
    # device = 'cuda'
    # faker = fz.test.Faker(device = 'cuda').make_data_loader()
    # model = ModelvD(
    #     d_model = 2, n_layer_set = 1, en_dim = 3, edge_dim = 1, device = 'cuda'
    # )
    # for x, y in faker:
    #     fea = x[0].to(device)
    #     adj = x[1].to(device)
    #     result = model(fea, adj)
    #     exp = model.explain(fea[0], adj[0])
    #     break
    # print(result.shape)
    # print(exp.shape)
