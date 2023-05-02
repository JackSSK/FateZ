#!/usr/bin/env python3
"""
Classifier for BERT-based model.

author: jy, nkmtmsys
"""
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
        self.classifier = classifier


    def forward(self, input, mask = None):
        output = self.rep_embedder(input)
        output = self.encoder(output, mask)
        output = self.classifier(output)
        return output
