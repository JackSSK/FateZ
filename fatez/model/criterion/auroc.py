#!/usr/bin/env python3
"""
Area Under Receiver Operating Characteristic Curve (AUROC) as loss

author: jy
"""
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import fatez.model.criterion as criterion



class AUROC(nn.Module):
    """
    AUROC scoring
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(AUROC, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs
        # one-vs-rest as default
        if 'multi_class' not in self.kwargs:
            self.kwargs['multi_class'] = 'ovr'


    def forward(self, input, label, ):
        top_probs, preds = torch.max(input, dim = -1)
        score = metrics.roc_auc_score(label.cpu(), preds.cpu(), **self.kwargs)

        if not self.requires_grad:
            return score
        else:
            # Make predictions by solving least square problem
            preds = criterion.probs_to_preds(input, preds)
            return criterion.preds_to_scores(preds, score)
