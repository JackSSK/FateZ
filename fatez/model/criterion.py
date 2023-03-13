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
import torch.nn as nn
import sklearn.metrics as metrics

"""
These methods are for linking one tensor to a result which has no grad through
solving least square problems and matmul input tensor with the inverse matrix.
Basically, we just simply whatever process leading to the result as a param mat.
"""
def reps_to_probs(reps, probs, device = None):
    inv = torch.linalg.lstsq(reps, probs,).solution
    return torch.matmul(reps, inv)

def probs_to_preds(probs, preds, device = None):
    preds = preds.reshape(probs.shape[0], 1).to(probs.dtype).to(probs.device)
    inv = torch.linalg.lstsq(probs, preds,).solution
    return torch.matmul(probs, inv)

def preds_to_scores(preds, scores, device = None):
    scores = torch.Tensor([[scores]],).to(preds.dtype).to(preds.device)
    inv = torch.linalg.lstsq(preds.T, scores).solution.T
    return torch.matmul(inv, preds)



class Accuracy(nn.Module):
    """
    Accuracy as loss
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(Accuracy, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs

    def forward(self, inputs, labels, ):
        top_probs, preds = torch.max(inputs, dim = -1)
        acc = (preds == labels).type(torch.float).sum().item() / len(labels)

        if not self.requires_grad:
            return acc
        else:
            # Make predictions by solving least square problem
            preds = probs_to_preds(inputs, preds)
            return preds_to_scores(preds, acc)



class F1_Score(nn.Module):
    """
    F1_Score as loss
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

    def forward(self, inputs, labels, ):
        top_probs, preds = torch.max(inputs, dim = -1)
        f1 = metrics.f1_score(labels, preds, **self.kwargs)

        if not self.requires_grad:
            return f1
        else:
            # Make predictions by solving least square problem
            preds = probs_to_preds(inputs, preds)
            return preds_to_scores(preds, f1)



class AUROC(nn.Module):
    """
    AUROC as loss
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(AUROC, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs
        # one-vs-rest as default
        if 'multi_class' not in self.kwargs:
            self.kwargs['multi_class'] = 'ovr'


    def forward(self, inputs, labels, ):
        top_probs, preds = torch.max(inputs, dim = -1)
        score = metrics.roc_auc_score(labels, preds, **self.kwargs)

        if not self.requires_grad:
            return score
        else:
            # Make predictions by solving least square problem
            preds = probs_to_preds(inputs, preds)
            return preds_to_scores(preds, score)



class Silhouette(nn.Module):
    """
    Silhouette as loss

    Note: For clustering
    TBC, not tested yet
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(Silhouette, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs

    def forward(self, inputs, labels, ):
        top_probs, preds = torch.max(inputs, dim = -1)
        score = metrics.silhouette_score(inputs, labels, **self.kwargs)

        if not self.requires_grad:
            return score
        else:
            # Make predictions by solving least square problem
            preds = probs_to_preds(inputs, preds)
            return preds_to_scores(preds, score)



if __name__ == '__main__':
    # Multiclass case
    n_sample = 4
    n_dim = 8
    layer = nn.Linear(n_dim, n_dim)
    input = layer(torch.randn(n_sample, n_dim,))
    labels = torch.empty(n_sample).random_(n_sample)
    probs, preds = torch.max(input, dim = -1)
    correct = (preds == labels).type(torch.float)
    print(f'Preds:{preds}',  f'Labels:{labels}', f'Correct:{correct}',)


    # test = Accuracy(requires_grad = True)(input, labels)
    # test = F1_Score(requires_grad = True,)(input, labels)
    # test = AUROC(requires_grad = True,)(input, labels)
    # test = F1_Score(requires_grad = True, average = 'micro')(input, labels)
    # test = F1_Score(requires_grad = True, average = 'macro')(input, labels)
    test.backward()

    # # We can confrim the backward is working here
    # for num, para in enumerate(list(layer.parameters())):
    #     print(para)
    #     print(para.grad)
