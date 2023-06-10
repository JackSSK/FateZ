#!/usr/bin/env python3
"""
Reconstructor for BERT-based model.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.adapter as adapter
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe



class Reconstructor(nn.Module):
    """
    Reconstructor of Node feature matrix (and Adjacent matrix)
    """
    def __init__(self,
        rep_embedder = pe.Skip(),
        encoder:transformer.Encoder = None,
        adapter:str = None,
        input_sizes:dict = None,
        train_adj:bool = False,
        dtype:str = None,
        **kwargs
        ):
        """
        :param rep_embedder: = position_embedder.Skip
            Positional embedding method for GNN-encoded representations.

        :param encoder:transformer.Encoder = None
            The Encoder to build pre-train model with.

        :param input_sizes:dict = None
            Exp

        :param train_adj:bool = False
            Whether reconstructing adjacent matrices or not.
        """
        super(Reconstructor, self).__init__()
        self.factory_kwargs = {'dtype': dtype}
        self.rep_embedder = rep_embedder
        self.encoder = encoder
        self.freeze_encoder = False
        self.adapter=self._set_adapter(adapter) if adapter is not None else None
        self.input_sizes = input_sizes
        self.recon_node = mlp.Model(
            type = 'RECON',
            d_model = self.encoder.d_model,
            n_layer_set = 1,
            n_class = self.input_sizes['node_attr'],
            dtype = dtype
        )
        self.recon_adj = None

        if train_adj:
            self.recon_adj = mlp.Model(
                type = 'RECON',
                d_model = self.encoder.d_model,
                n_layer_set = 1,
                n_class = self.input_sizes['n_node'],
                dtype = dtype
            )
            if self.input_sizes['edge_attr'] > 1:
                print('ToDo: capable to reconstruc multiple edge attrs')

    def forward(self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            is_causal: bool = None,
            ) -> torch.Tensor:
        out = self.rep_embedder(src)
        if self.adapter is None:
            out = self.encoder(out, mask, src_key_padding_mask, is_causal)
        else:
            out=self.deploy_adapter(out, mask, src_key_padding_mask, is_causal)
        node_mat = self.recon_node(out)
        if self.recon_adj != None:
            adj_mat = self.recon_adj(out)
        else:
            adj_mat = None

        return node_mat, adj_mat

    def deploy_adapter(self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            is_causal: bool = None,
            ) -> torch.Tensor:
        output, args, convert_to_nested = self.encoder.prepare(
            src, mask, src_key_padding_mask, is_causal
        )
        output = self.adapter(
            output,
            args = args,
            encoder_layers = self.encoder.encoder.layers,
            freeze_encoder = self.freeze_encoder,
        )
        if convert_to_nested: output = output.to_padded_tensor(0.)
        return self.encoder.normalize(output)

    def _set_adapter(self, adp_type:str = 'LORA',):
        """
        Set up adapter model accordingly.
        """
        n_dim = self.encoder.d_model
        n_layer = len(self.encoder.encoder.layers)
        if adp_type == 'LORA':
            return adapter.LoRA(
                d_model = n_dim,
                n_layer = n_layer,
                **self.factory_kwargs,
            )
        else:
            raise model.Error('Unknown Adapter Type:', adp_type)
