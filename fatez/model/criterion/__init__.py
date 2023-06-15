#!/usr/bin/env python3
"""
Modules for criterion functions

Why?
1:
Implement Module based precision, recall (and F1 score etc) criterion functions
So we can backward params based on such scores (cheating. yes.)

2:
Find a way to transfer sklearn or xgboost based model's prediction outputs into
loss we can backward

author: jy
"""
import torch
from fatez.model.criterion.accuracy import Accuracy
from fatez.model.criterion.auroc import AUROC
from fatez.model.criterion.f1 import F1_Score
from fatez.model.criterion.silhouette import Silhouette

__all__ = [
    'Accuracy',
    'AUROC',
    'F1_Score',
    'Silhouette',
]

"""
These methods are for linking one tensor to a result which has no grad through
solving least square problems and matmul input tensor with the inverse matrix.
Basically, we just simply whatever process leading to the result as a param mat.
"""
def reps_to_probs(reps, probs,):
    inv = torch.linalg.lstsq(reps, probs,).solution
    return torch.matmul(reps.data, inv)

def probs_to_preds(probs, preds,):
    preds = preds.reshape(probs.shape[0], 1).to(probs.dtype).to(probs.device)
    inv = torch.linalg.lstsq(probs, preds,).solution
    return torch.matmul(probs.data, inv)

def preds_to_scores(preds, scores,):
    scores = torch.Tensor([[scores]],).to(preds.dtype).to(preds.device)
    inv = torch.linalg.lstsq(preds.T, scores).solution.T
    return torch.matmul(inv, preds.data)



# if __name__ == '__main__':
#     import torch.nn as nn
#     # Multiclass case
#     n_sample = 4
#     n_dim = 8
#     layer = nn.Linear(n_dim, n_dim)
#     input = layer(torch.randn(n_sample, n_dim,))
#     label = torch.empty(n_sample).random_(n_sample)
#     probs, preds = torch.max(input, dim = -1)
#     correct = (preds == label).type(torch.float)
#     print(f'Preds:{preds}',  f'Labels:{label}', f'Correct:{correct}',)
#
#
#     test = Accuracy(requires_grad = True)(input, label)
#     test = F1_Score(requires_grad = True,)(input, label)
#     test = AUROC(requires_grad = True,)(input, label)
#     # test = F1_Score(requires_grad = True, average = 'micro')(input, label)
#     # test = F1_Score(requires_grad = True, average = 'macro')(input, label)
#     test.backward()
#
#     # # We can confrim the backward is working here
#     # for num, para in enumerate(list(layer.parameters())):
#     #     print(para)
#     #     print(para.grad)
