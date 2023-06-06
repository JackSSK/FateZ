#!/usr/bin/env python3
"""
Reconstructor for BERT-based model.

author: jy, nkmtmsys
"""
import torch.nn as nn
import fatez.model.mlp as mlp
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe



class Reconstructor(nn.Module):
    """
    Reconstructor of Node feature matrix (and Adjacent matrix)
    """
    def __init__(self,
        rep_embedder = pe.Skip(),
        encoder:transformer.Encoder = None,
        mat_sizes:dict = None,
        train_adj:bool = False,
        dtype:str = None,
        **kwargs
        ):
        """
        :param rep_embedder: = position_embedder.Skip
            Positional embedding method for GNN-encoded representations.

        :param encoder:transformer.Encoder = None
            The Encoder to build pre-train model with.

        :param mat_sizes:dict = None
            Exp

        :param train_adj:bool = False
            Whether reconstructing adjacent matrices or not.
        """
        super(Reconstructor, self).__init__()
        self.rep_embedder = rep_embedder
        self.encoder = encoder
        self.mat_sizes = mat_sizes
        self.recon_node = mlp.Model(
            type = 'RECON',
            d_model = self.encoder.d_model,
            n_layer_set = 1,
            n_class = self.mat_sizes['node_attr'],
            dtype = dtype
        )
        self.recon_adj = None

        if train_adj:
            self.recon_adj = mlp.Model(
                type = 'RECON',
                d_model = self.encoder.d_model,
                n_layer_set = 1,
                n_class = self.mat_sizes['n_node'],
                dtype = dtype
            )
            if self.mat_sizes['edge_attr'] > 1:
                print('ToDo: capable to reconstruc multiple edge attrs')

    def forward(self, input, mask = None):
        output = self.rep_embedder(input)
        embed_rep = self.encoder(output, mask)
        node_mat = self.recon_node(embed_rep)

        if self.recon_adj != None:
            adj_mat = self.recon_adj(embed_rep)
        else:
            adj_mat = None

        return node_mat, adj_mat
