#!/usr/bin/env python3
"""
F1 Score as loss

author: jy
"""
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import fatez.model.criterion as criterion



class F1_Score(nn.Module):
    """
    F1 Score
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(F1_Score, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs
        # Suppress warning
        if 'zero_division' not in self.kwargs:
            self.kwargs['zero_division'] = 0
        # weighted as default average option considering imbalance label probs
        if 'average' not in self.kwargs:
            self.kwargs['average'] = 'weighted'

    def forward(self, input, label, ):
        top_probs, preds = torch.max(input, dim = -1)
        f1 = metrics.f1_score(label.cpu(), preds.cpu(), **self.kwargs)

        if not self.requires_grad:
            return f1
        else:
            # Make predictions by solving least square problem
            preds = criterion.probs_to_preds(input, preds)
            return criterion.preds_to_scores(preds, f1)
