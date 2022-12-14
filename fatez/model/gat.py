#!/usr/bin/env python3
"""
This folder contains Graph Attention Network (GAT) related objects.

author: jy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def _prepare_attentions(e_values, adj_mat):
    """
    Calculate attention values for every GRP.

    :param e_values:torch.Tensor = None
        Weighted matrix.

    :param adj_mat:torch.Tensor = None
        Adjacent matrix. (Based on GRPs)
    """
    # Basically, this is a matrix of negative infinite with e_values.shape
    neg_inf = -9e15 * torch.ones_like(e_values)
    # Left confirmed GRPs only.
    attention = torch.where(adj_mat != 0, e_values, neg_inf)
    # Replace 0s in adjacent matrix to 1s
    new_adj = torch.where(adj_mat != 0, adj_mat, torch.ones_like(adj_mat))
    # Multiply GRP coefficient to the attention values
    attention = np.multiply(
        attention.detach().cpu().numpy(),
        new_adj.detach().cpu()
    )
    attention = F.softmax(attention, dim = 1)
    return attention

def Get_Attention(fea_mat, adj_mat, weights, a_values, out_dim):
    w_h = torch.mm(fea_mat, weights)
    w_h1 = torch.matmul(w_h[:adj_mat.size()[0],:], a_values[:out_dim,:])
    w_h2 = torch.matmul(w_h, a_values[out_dim:, :])
    e_values = F.leaky_relu(w_h1 + w_h2.T)
    return _prepare_attentions(e_values, adj_mat)



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



class GAT(nn.Module):
    """
    A typical GAT.
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
        super(GAT, self).__init__()
        self.n_hidden = n_hidden
        self.en_dim = en_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.attentions = None
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        # Add attention heads
        if nhead is not None and nhead > 0:
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

    def forward(self, fea_mats, adj_mats):
        """
        :param fea_mats: torch.Tensor
            Feature matrices. (Genes)
        :param adj_mats: torch.Tensor
            Adjacent matrices. (Based on GRPs)
        """
        answer = list()
        assert len(fea_mats) == len(adj_mats)
        for i in range(len(fea_mats)):
            x = fea_mats[i]
            adj_mat = adj_mats[i]
            x = F.dropout(x, self.dropout, training = self.training)
            x = x.to(self.factory_kwargs['device'])
            adj_mat = adj_mat.to(self.factory_kwargs['device'])
            # Multi-head attention mechanism
            if self.attentions is not None:
                x = torch.cat([a(x, adj_mat) for a in self.attentions], dim = 1)
                x = F.dropout(x, self.dropout, training = self.training)
                # Resize the adj_mat to top_k rows
                x = F.elu(self.last(x, adj_mat.narrow(1, 0, adj_mat.size()[0])))
            else:
                x = F.elu(self.last(x, adj_mat))
            answer.append(x)
        return torch.stack(answer, 0)

    def explain(self, fea_mat, adj_mat):
        """
        Input real data, then this func is explaining in real case.
        For general explanation, just make a new mat with all values == 1.
        fake_fea_mat = torch.ones_like(fea_mat)
        fake_adj_mat = torch.ones_like(adj_mat)
        """

        att_explain = None
        last_explain = None

        if self.attentions is not None:
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
        return att_explain, last_explain
