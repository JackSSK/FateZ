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



class Graph_Attention_Layer(nn.Module):
    """
    Simple graph attention layer for GAT.
    """
    def __init__(self,
        in_dim:int = None,
        out_dim:int = None,
        lr:float = 0.005,
        weight_decay:float = 5e-4,
        gain:float = 1.414,
        dropout:float = 0.2,
        alpha:float = 0.2,
        concat:bool = True,
        ):
        """
        :param in_dim:int = None
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
        """
        super(Graph_Attention_Layer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat
        # Set up parameters
        self.weights = nn.Parameter(torch.empty(size = (in_dim, out_dim)))
        self.a_values = nn.Parameter(torch.empty(size = (2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.weights.data, gain = gain)
        nn.init.xavier_uniform_(self.a_values.data, gain = gain)

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.optimizer = optim.Adam(
            self.parameters(),
            lr = lr,
            weight_decay = weight_decay
        )

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

    def _prepare_attentions(self, e_values, adj_mat):
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
        attention = np.multiply(attention.detach().numpy(), new_adj)
        attention = F.softmax(attention, dim = 1)
        return attention

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
        e_values = self._prepare_attentional_mechanism_input(w_h, len(adj_mat))
        attention = self._prepare_attentions(e_values, adj_mat)
        attention = F.dropout(attention, self.dropout, training = self.training)
        result = torch.matmul(attention, w_h)

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
        in_dim:int = None,
        en_dim:int = 2,
        n_class:int = 2,
        n_hidden:int = 1,
        n_head:int = None,
        lr:float = 0.005,
        weight_decay:float = 5e-4,
        dropout:float = 0.2,
        alpha:float = 0.2,
        ):
        """
        :param in_dim:int = None
            Number of each gene's input features.

        :param en_dim:int = 2
            Number of each gene's encoded features.

        :param n_class:int = None
            Number of classes.

        :param n_hidden:int = None
            Number of hidden units.

        :param n_head:int = None
            Number of attention heads.

        :param lr:float = 0.005
            Learning rate.

        :param weight_decay:float = 5e-4
            Weight decay (L2 loss on parameters).

        :param dropout:float = 0.2
            Dropout rate.

        :param alpha:float = 0.2
            Alpha value for the LeakyRelu layer.
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.attentions = None

        # Add attention heads
        if n_head is not None and n_head > 0:
            self.attentions = [
                Graph_Attention_Layer(
                    in_dim = in_dim,
                    out_dim = n_hidden,
                    lr = lr,
                    weight_decay = weight_decay,
                    dropout = dropout,
                    alpha = alpha,
                    concat = True
                ) for _ in range(n_head)
            ]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            # Change input dimension for last GAT layer
            in_dim = n_hidden * n_head

        # Also, we can add other layers here.

        # Last output GAT layer
        self.last = Graph_Attention_Layer(
            in_dim = in_dim,
            out_dim = en_dim,
            lr = lr,
            weight_decay = weight_decay,
            dropout = dropout,
            alpha = alpha,
            concat = False,
        )
        self.decision_layer = nn.Linear(en_dim, n_class)

    def forward(self, samples):
        """
        :param samples:list = None
            List of GRN representing matrix sets:
                Input matrix. (Genes)
                Adjacent matrix. (Based on GRPs)
        """
        answer = list()
        for sample in samples:
            x = sample[0]
            adj_mat = sample[1]
            x = F.dropout(x, self.dropout, training = self.training)
            # Multi-head attention mechanism
            if self.attentions is not None:
                x = torch.cat([a(x, adj_mat) for a in self.attentions], dim = 1)
                x = F.dropout(x, self.dropout, training = self.training)
            x = F.elu(self.last(x, adj_mat))
            answer.append(x)
        answer = torch.stack(answer, 0)
        return answer

    def activation(self, input):
        """
        Use activation layer
        """
        return F.log_softmax(input, dim = -1)

    def decision(self, input):
        """
        Use decsion layer
        """
        return F.softmax(self.decision_layer(input), dim = -1)
