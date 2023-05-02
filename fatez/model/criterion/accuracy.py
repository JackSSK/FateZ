#!/usr/bin/env python3
"""
Accuracy as loss

author: jy
"""
import torch
import torch.nn as nn
import fatez.model.criterion as criterion



class Accuracy(nn.Module):
    """
    Accuracy scoring
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(Accuracy, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs

    def forward(self, input, label, ):
        top_probs, preds = torch.max(input, dim = -1)
        acc = (preds == label).type(torch.float).sum().item() / len(label)

        if not self.requires_grad:
            return acc
        else:
            # Make predictions by solving least square problem
            preds = criterion.probs_to_preds(input, preds)
            return criterion.preds_to_scores(preds, acc)
