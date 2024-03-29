#!/usr/bin/env python3
"""
Classifier for BERT-based model.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            input_sizes:list = None,
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

        :param adapter:str = None
            Exp

        :param input_sizes:dict = None
            Exp
        """
        super(Classifier, self).__init__()
        self.input_sizes = input_sizes
        self.dtype = dtype
        self.freeze_encoder = True
        self.rep_embedder = rep_embedder
        self.encoder = encoder
        self.adapter=self._set_adapter(adapter) if adapter is not None else None
        self.relu = nn.ReLU(inplace = True)
        self.classifier = self._set_classifier(
            n_features = self.input_sizes['n_reg'],
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
        # out = self.relu(out)
        out = self.classifier(out)
        return F.softmax(out, dim = -1)

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
        if adp_type.upper() == 'LORA':
            return adapter.LoRA(
                d_model = n_dim,
                n_layer = n_layer,
                dtype = self.dtype,
            )
        else:
            raise model.Error('Unknown Adapter Type:', adp_type)

    def _set_classifier(self,
            n_features:int = 4,
            n_dim:int = 4,
            n_class:int = 2,
            clf_type:str = 'MLP',
            clf_params:dict = {'n_hidden': 2},
            ):
        """
        Set up classifier model accordingly.
        """
        args = {
            'n_features': n_features,
            'n_dim': n_dim,
            'n_class': n_class,
            'dtype': self.dtype,
        }
        if clf_type.upper() == 'MLP':
            return mlp.Model(d_model = n_dim, **args, **clf_params,)
        elif clf_type.upper() == 'CNN_1D':
            return cnn.Model_1D(**args, **clf_params,)
        elif clf_type.upper() == 'CNN_2D':
            return cnn.Model_2D(**args, **clf_params,)
        elif clf_type.upper() == 'CNN_HYB':
            return cnn.Model_Hybrid(**args, **clf_params,)
        elif clf_type.upper() == 'RNN':
            return rnn.RNN(**args, **clf_params,)
        elif clf_type.upper() == 'GRU':
            return rnn.GRU(**args, **clf_params,)
        elif clf_type.upper() == 'LSTM':
            return rnn.LSTM(**args, **clf_params,)
        else:
            raise model.Error('Unknown Classifier Type:', clf_type)
