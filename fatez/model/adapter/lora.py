#!/usr/bin/env python3
"""
Low Rank Adaption (LoRA) Model related stuffs.

author: jy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Model(nn.Module):
    """
    A simple LoRA model.
    """
    def __init__(self,
            d_model:int = 512,
            n_layer:int = 6,
            dtype:str = None,
            **kwargs
            ):
        """
        :param d_model <int = 512>
            Number of expected features in the inputs.

        :param n_layer <int = 6>
            Number of encoder layers.
        """
        super(Model, self).__init__()

        self.model = OrderedDict([])
        for i in range(n_layer):
            self.model.update(
                {f'layer{i}': nn.Linear(d_model, d_model, dtype = dtype)}
            )
        self.model = nn.Sequential(self.model)

    def forward(self, output, args, encoder_layers, freeze_encoder:bool=False):
        for i, mod in enumerate(encoder_layers):
            if freeze_encoder:
                with torch.no_grad(): output = mod(output, **args)
            else:
                output = mod(output, **args)
            output = self.model[i](output)
        return output
