#!/usr/bin/env python3
"""
1-layer GAT implemented only using densed matrices.

author: jy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def _prepare_attentions(e_values, adj_mat, device = 'cpu'):
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
    ).to(device)
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
            torch.empty(size = (d_model, out_dim), device=device, dtype=dtype)
        )
        self.a_values = nn.Parameter(
            torch.empty(size = (2 * out_dim, 1), device=device, dtype=dtype)
        )
        nn.init.xavier_uniform_(self.weights.data, gain = gain)
        nn.init.xavier_uniform_(self.a_values.data, gain = gain)

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.factory_kwargs = {'device': device, 'dtype': dtype}

    def forward(self, input, adj_mat):
        """
        :param input:torch.Tensor = None
            Input matrix. (Genes)

        :param adj_mat:torch.Tensor = None
            Adjacent matrix. (Based on GRPs)
        """
        device = self.factory_kwargs['device']
        # Multiply hs to ensure output dimension == out_dim
        # The Weighted matrix: w_h.shape == (N, out_feature)
        w_h = torch.mm(input, self.weights)

        # Now we calculate weights for GRPs according to input matrix & weights.
        n_regulons = adj_mat.size()[0]
        # w_h1.shape (n_regulons, 1) and w_h2.shape (N, 1)
        w_h1 = torch.matmul(w_h[:n_regulons,:], self.a_values[:self.out_dim, :])
        w_h2 = torch.matmul(w_h, self.a_values[self.out_dim:, :])
        # broadcast add: e_values.shape (n_regulons, N)
        e_values = self.leaky_relu(w_h1 + w_h2.T)

        # Then we apply e_values to calculate attention
        attention = _prepare_attentions(e_values, adj_mat, device = device)
        attention = F.dropout(attention, self.dropout, training = self.training)
        result = torch.matmul(attention, w_h)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(result)
        else:
            # if this layer is last layer,
            return result

    def switch_device(self, device:str = 'cpu'):
        self.factory_kwargs['device'] = device



class Model(nn.Module):
    """
    A typical GAT but using dense matrix instead of sparse.
    This shall be used if the PyG sparse version is just too slow.
    Note: This implementation only has 1 layer.
    """
    def __init__(self,
        input_sizes:dict = None,
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
        super(Model, self).__init__()
        self.input_sizes = input_sizes
        self.n_hidden = n_hidden
        self.en_dim = en_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.attentions = None
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        d_model = self.input_sizes['node_attr']
        # Add attention heads
        if nhead != None and nhead > 1:
            self.attentions = [
                Graph_Attention_Layer(
                    d_model = d_model,
                    out_dim = n_hidden,
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
        device = self.factory_kwargs['device']
        answer = list()
        assert len(fea_mats) == len(adj_mats)
        for i in range(len(fea_mats)):
            x, adj_mat = self._load_mats(fea_mats[i], adj_mats[i], device)
            # Dropout layer first
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
        device = self.factory_kwargs['device']
        fea_mat, adj_mat = self._load_mats(fea_mat, adj_mat, device)

        att_explain = None
        last_explain = None

        def Get_Attention(fea_mat, adj_mat, weights, a_values, out_dim):
            w_h = torch.mm(fea_mat, weights)
            w_h1 = torch.matmul(w_h[:adj_mat.size()[0],:], a_values[:out_dim,:])
            w_h2 = torch.matmul(w_h, a_values[out_dim:, :])
            e_values = F.leaky_relu(w_h1 + w_h2.T)
            return _prepare_attentions(e_values, adj_mat, device)

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

    def switch_device(self, device:str = 'cpu'):
        self.factory_kwargs['device'] = device
        if self.attentions is not None:
            for att in self.attentions:
                att.switch_device(device)
                att = att.to(device)
        self.last.switch_device(device)
        self.last = self.last.to(device)
        return

    def _load_mats(self, fea_mat, adj_mat, device):
        fea_mat = fea_mat.to(device).to_dense()
        shape = adj_mat.shape
        if len(shape) > 2: shape = shape[:2]
        adj_mat = adj_mat.to(device).to_dense().reshape(shape)
        return fea_mat, adj_mat
