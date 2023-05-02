#!/usr/bin/env python3
"""
Silhouette score as loss

author: jy
"""
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import fatez.model.criterion as criterion


class Silhouette(nn.Module):
    """
    Silhouette score

    Note: For clustering
    TBC, not tested yet
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(Silhouette, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs

    def forward(self, input, label, ):
        top_probs, preds = torch.max(input, dim = -1)
        score = metrics.silhouette_score(input.cpu(), label.cpu(), **self.kwargs)

        if not self.requires_grad:
            return score
        else:
            # Make predictions by solving least square problem
            preds = criterion.probs_to_preds(input, preds)
            return criterion.preds_to_scores(preds, score)
