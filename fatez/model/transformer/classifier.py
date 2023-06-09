#!/usr/bin/env python3
"""
Classifier for BERT-based model.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gnn as gnn
import fatez.model.cnn as cnn
import fatez.model.rnn as rnn
import fatez.model.adapter as adapter
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe



class Classifier(nn.Module):
    """
    Classification model.
    """
    def __init__(self,
            rep_embedder = pe.Skip(),
            encoder:transformer.Encoder = None,
            adapter:str = 'LORA',
            clf_type:str = 'MLP',
            clf_params:dict = {'n_hidden': 2},
            n_class:int = 100,
            dtype:str = None,
            **kwargs
            ):
        """
        :param rep_embedder: = position_embedder.Skip
            Positional embedding method for GNN-encoded representations.

        :param encoder:transformer.Encoder = None
            The Encoder to build fine-tune model with.

        :param classifier = None
            The classification model for making predictions.
        """
        super(Classifier, self).__init__()
        self.factory_kwargs = {'dtype': dtype}
        self.rep_embedder = rep_embedder
        self.encoder = encoder
        self.adapter=self._set_adapter(adapter) if adapter is not None else None
        self.classifier = self._set_classifier(
            n_dim = encoder.d_model,
            n_class = n_class,
            clf_type = clf_type,
            clf_params = clf_params,
        )

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

        out = self.classifier(out)
        return out

    def deploy_adapter(self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            is_causal: bool = None,
            ) -> torch.Tensor:
        output, args, convert_to_nested = self.encoder.prepare(
            src, mask, src_key_padding_mask, is_causal
        )
        output = self.adapter(output, args, self.encoder.encoder.layers)
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

    def _set_classifier(self,
            n_dim:int = 4,
            n_class:int = 2,
            clf_type:str = 'MLP',
            clf_params:dict = {'n_hidden': 2},
            ):
        """
        Set up classifier model accordingly.
        """
        if clf_type.upper() == 'MLP':
            return mlp.Model(
                d_model = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'CNN_1D':
            return cnn.Model_1D(
                in_channels = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'CNN_2D':
            return cnn.Model_2D(
                in_channels = 1,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'CNN_HYB':
            return cnn.Model_Hybrid(
                in_channels = 1,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'RNN':
            return rnn.RNN(
                input_size = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'GRU':
            return rnn.GRU(
                input_size = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'LSTM':
            return rnn.LSTM(
                input_size = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        else:
            raise model.Error('Unknown Classifier Type:', clf_type)
