#!/usr/bin/env python3
"""
This file contains Graph Attention Network (GAT) related objects.

author: jy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import fatez.model as model



def Set(config:dict=None, input_sizes:list=None, factory_kwargs:dict=None):
    """
    Set up GAT model based on given config.
    """
    if config['type'].upper() == 'GAT':
        return Model(**config['params'], **factory_kwargs)
    elif config['type'].upper() == 'SGAT':
        return Sparse_Model(**config['params'], **factory_kwargs)
    else:
        raise model.Error('Unknown GAT type')



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



def Get_Attention(fea_mat, adj_mat, weights, a_values, out_dim):
    """
    """
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



class Sparse_MatMul_Function(torch.autograd.Function):
    """
    Matrix maultiplication for sparse region backpropataion layer.
    """
    @staticmethod
    def forward(ctx, indices, values, shape, mat_2,):
        assert indices.requires_grad == False
        mat_1 = torch.sparse_coo_tensor(
            indices, values, shape, dtype = mat_2.dtype
        )
        ctx.save_for_backward(mat_1, mat_2)
        ctx.N = shape[0]
        return torch.matmul(mat_1, mat_2)

    @staticmethod
    def backward(ctx, grad_output):
        mat_1, mat_2 = ctx.saved_tensors
        grad_mat_1 = grad_mat_2 = None
        if ctx.needs_input_grad[1]:
            edge_idx = mat_1._indices()[0, :] * ctx.N + mat_1._indices()[1, :]
            grad_mat_1 = grad_output.matmul(mat_2.t()).view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_mat_2 = mat_1.t().matmul(grad_output)
        return None, grad_mat_1, None, grad_mat_2



class Sparse_MatMul(nn.Module):
    """
    Sparse matrix maultiplication object.
    """
    def forward(self, indices, values, shape, mat_2):
        """
        :param indices:
            Matrix of [2, XXX] indicating location of GRPs.

        :param values:

        :param shape:
            Should be # genes * # genes in this case.

        :param mat_2:
            Should be the [#genes, #faetures] matrix.
        """
        return Sparse_MatMul_Function.apply(indices, values, shape, mat_2)



class Sparse_Graph_Attention_Layer(nn.Module):
    """
    Sparse version GAT layer
    """
    def __init__(self,
        d_model:int = None,
        out_dim:int = None,
        lr:float = 0.005,
        weight_decay:float = 5e-4,
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
        super(Sparse_Graph_Attention_Layer, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat
        # Set up parameters
        self.weights = nn.Parameter(
            torch.zeros(size = (d_model, out_dim), device = device, dtype=dtype)
        )
        self.a_values = nn.Parameter(
            torch.zeros(size = (1, 2 * out_dim), device = device, dtype = dtype)
        )
        nn.init.xavier_normal_(self.weights.data, gain = 1.414)
        nn.init.xavier_normal_(self.a_values.data, gain = 1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.optimizer = optim.Adam(
            self.parameters(),
            lr = lr,
            weight_decay = weight_decay
        )
        self.sparse_matmul = Sparse_MatMul()

    def forward(self, input, adj_mat):
        """
        :param input:torch.Tensor = None
            Input matrix. (Genes)

        :param adj_mat:torch.Tensor = None
            Adjacent matrix. (Based on GRPs)
        """
        row_num = input.size()[0]
        n_regulons = adj_mat.size()[0]
        edge_indices = adj_mat.nonzero().t()
        # print('GRP indices:', edge_indices)

        w_h = torch.mm(input, self.weights)
        # h: row_num x out
        assert not torch.isnan(w_h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat(
            (w_h[edge_indices[0, :], :], w_h[edge_indices[1, :], :]),
            dim = 1
        ).t()

        edge_e_values = torch.exp(
            -self.leakyrelu(self.a_values.mm(edge_h).squeeze())
        )
        assert not torch.isnan(edge_e_values).any()
        # edge_e_values: E

        device = 'cuda' if input.is_cuda else 'cpu'
        row_e_values_sum = self.sparse_matmul(
            indices = edge_indices,
            values = edge_e_values,
            shape = torch.Size([n_regulons, row_num]),
            mat_2 = torch.ones(size = (row_num, 1), device = device)
        )
        edge_e_values = F.dropout(
            edge_e_values,
            self.dropout,
            training = self.training
        )

        result = self.sparse_matmul(
            indices = edge_indices,
            values = edge_e_values,
            shape = torch.Size([n_regulons, row_num]),
            mat_2 = w_h
        )
        assert not torch.isnan(result).any()
        # result: row_num x out

        result = result.div(row_e_values_sum)
        # result: row_num x out
        assert not torch.isnan(result).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(result)
        else:
            # if this layer is last layer,
            return result



class Model(nn.Module):
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
        super(Model, self).__init__()
        self.d_model = d_model
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
            if self.attentions is not None:
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

        # Not using multihead attention
        if att_explain is None:
            return last_explain
        else:
            return torch.matmul(last_explain, att_explain)



class Sparse_Model(nn.Module):
    """
    Sparse version of GAT.
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

        :param nhead:int = 1
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
        super(Sparse_Model, self).__init__()
        self.d_model = d_model
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
                Sparse_Graph_Attention_Layer(
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

        # Last output GAT layer
        self.last = Sparse_Graph_Attention_Layer(
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
            x = fea_mats[i].to(self.factory_kwargs['device']).to_dense()
            adj_mat = adj_mats[i].to(self.factory_kwargs['device']).to_dense()
            x = F.dropout(x, self.dropout, training = self.training)
            # Multi-head attention mechanism
            if self.attentions is not None:
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
                a_values.T,
                out_dim = self.n_hidden,
            )

            last_explain = Get_Attention(
                torch.cat(
                    [a.eval()(fea_mat, adj_mat) for a in self.attentions],
                    dim = 1
                ),
                adj_mat.narrow(1, 0, adj_mat.size()[0]),
                self.last.weights,
                self.last.a_values.T,
                out_dim = self.en_dim,
            )
        else:
            last_explain = Get_Attention(
                fea_mat,
                adj_mat,
                self.last.weights,
                self.last.a_values.T,
                out_dim = self.en_dim,
            )

        # Not using multihead attention
        if att_explain is None:
            return last_explain
        else:
            return torch.matmul(last_explain, att_explain)
