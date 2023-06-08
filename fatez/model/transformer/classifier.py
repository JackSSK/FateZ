#!/usr/bin/env python3
"""
Classifier for BERT-based model.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe



class Classifier(nn.Module):
    """
    Classification model.
    """
    def __init__(self,
        rep_embedder = pe.Skip(),
        encoder:transformer.Encoder = None,
        adaptor = 1,
        classifier = None,
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
        self.rep_embedder = rep_embedder
        self.encoder = encoder
        self.adaptor = adaptor
        self.classifier = classifier


    def forward(self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            is_causal: bool = None,
            ) -> torch.Tensor:
        output = self.rep_embedder(src)
        if self.adaptor is None:
            output = self.encoder(output, mask, src_key_padding_mask, is_causal)
        else:
            output = self.deploy_adapter(
                output, mask, src_key_padding_mask, is_causal
            )

        output = self.classifier(output)
        return output

    def deploy_adapter(self,
            output: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            is_causal: bool = None,
            ) -> torch.Tensor:
        args, convert_to_nested = self.encoder.prepare(
            output, mask, src_key_padding_mask, is_causal
        )

        for mod in self.encoder.encoder.layers:
            mod.eval()
            output = mod(**args)
            mod.train()

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        return self.encoder.normalize(output)
