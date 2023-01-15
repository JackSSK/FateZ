#!/usr/bin/env python3
"""
Encode GRN with Graph Attention Network (GAT) techniques.

Note: Developing~

author: jy
"""
import os
import glob
import torch
from torch.autograd import Variable
import fatez.model.gat as gat



class Encode(object):
    """
    Encode GRNs with GAT.
    """

    def __init__(self,
        **kwargs
        ):
        super(Encode, self).__init__()
        self.d_model = d_model  # Must be initialized here
        self.en_dim = en_dim
        self.n_hidden = n_hidden
        self.nhead = nhead
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.alpha = alpha

        self.gat_model = gat.GAT(
            d_model = self.d_model,
            en_dim = self.n_enc_fea,
            n_hidden = self.n_hidden ,
            nhead = self.nhead ,
            lr = self.lr,
            weight_decay = self.weight_decay,
            dropout = self.dropout,
            alpha = self.alpha,
        )

        self.binning = None

    def process(self,
        grns:list = None,
        use_gpu:bool = True,
        ):
        genes_mats, adj_mats = self.prepare_input(sample)
        if torch.cuda.is_available() and use_gpu is True:
            self.gat_model.cuda()
            genes_mats = features.cuda()
            adj_mats = adj_mats.cuda()

        genes_mats, adj_mats = Variable(genes_mats), Variable(adj_mats)
        self.gat_model.train()
        output = self.gat_model(genes_mats, adj_mats)
        # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.gat_model.optimizer.step()



    def prepare_input(self, grns):
        for sample in grns:
            print(sample)

    # Use fatez.model.Save and fatez.model.Load
    # def save(self, path):
    #     print('under construction')
    #
    # def load(self, path):
    #     print('under construction')
