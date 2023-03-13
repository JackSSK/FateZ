#!/usr/bin/env python3
"""
Modules for criterion functions

Why?
1:
Implement Module based precision, recall (and F1 score etc) criterion functions

2:
Find a way to transfer sklearn or xgboost based model's prediction outputs into
loss we can backward

author: jy
"""
import torch
import torch.nn as nn
from sklearn.metrics import f1_score as f1score



class Inverse_Mat_Cal(nn.Module):
    """
    docstring for Inverse_Mat_Cal.
    """

    def __init__(self, **kwargs):
        super(Accuracy, self).__init__()

    def forward(self, inputs, target_loss, feat_import = None):
        n_sample = input.shape[0]
        target_mat = torch.ones(n_sample, 1) * target_loss / n_sample
        inverse_mat = torch.linalg.lstsq(input, target_mat,driver = 'gels').solution


def probs_to_preds(probs, preds):
    preds = preds.reshape(probs.shape[0],1).to(probs.dtype)
    inv = torch.linalg.lstsq(probs, preds,).solution
    return torch.matmul(probs, inv)

def preds_to_value(preds, value):
    inv = torch.linalg.lstsq(preds.T, torch.Tensor([[value]])).solution.T
    return torch.matmul(inv, preds)


class Accuracy(nn.Module):
    """
    Loss as Accuracy.
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(Accuracy, self).__init__()
        self.requires_grad = requires_grad

    def forward(self, inputs, labels, ):
        probs, preds = torch.max(inputs, dim = -1)
        acc = (preds == labels).type(torch.float).sum().item() / len(labels)

        if not self.requires_grad:
            return acc
        else:
            # Make predictions by solving least square problem
            preds = probs_to_preds(inputs, preds)
            acc = preds_to_value(preds, acc)

            return acc



class F1_Score(nn.Module):
    """
    Loss as F1_Score.
    """

    def __init__(self, n_class:int = 2, **kwargs):
        super(F1_Score, self).__init__()
        self.kwargs = kwargs
        self.n_class = n_class


    def forward(self, inputs, labels, ):
        assert max(labels).item() <= self.n_class
        probs, preds = torch.max(inputs, dim = -1)
        inverse_mat = torch.linalg.lstsq(input, preds,).solution
        preds = torch.mm(input, inverse_mat)
        f1 = f1score(labels, preds, **self.kwargs)

        # f1s = [self.f1_score(i, preds, labels) for i in range(self.n_class)]
        # f1 = sum(f1s) / len(labels)
        #
        # f1 = self.f1_score(1, preds, labels,)

        return f1


    def f1_score(self, cls, preds, labels,):
        true_pos = len([
            x for x in range(len(preds)) if preds[x]==cls and labels[x]==cls
        ])
        if true_pos == 0: return 0

        precision = true_pos / (preds == cls).type(torch.float).sum().item()
        truth = (labels == cls).type(torch.float).sum().item()
        recall = true_pos / truth
        result = (2 * precision * recall) / (precision + recall)

        if self.type == 'WEIGHT':
            return truth * result
        else:
            return result


if __name__ == '__main__':
    # Multiclass case
    layer = nn.Linear(4,4)
    input = layer(torch.randn(4,4,))

    labels = torch.empty(4).random_(4)
    probs, preds = torch.max(input, dim = -1)
    correct = (preds == labels).type(torch.float)

    print(f'Preds:{preds}',  f'Labels:{labels}', f'Correct:{correct}',)

    test = Accuracy(requires_grad = True)(input, labels)
    test.backward()
    for num, para in enumerate(list(layer.parameters())):
        print(para)
        print(para.grad)

    # test = F1_Score(n_class = len(input[0]),  type = 'macro')
    # print(f'Macro:{test(input, labels) == f1score(labels, preds, average = "macro")}')

    # test = F1_Score(n_class = len(input[0]),  type = 'micro')
    # print(f'Micro:{test(input, labels) == f1score(labels, preds, average = "micro")}')
    #
    # test = F1_Score(n_class = len(input[0]),  type = 'weight')
    # print(f'Weight:{test(input, labels) == f1score(labels, preds, average = "weighted")}')
    #
    #
    # # Binary case
    # input = abs(torch.randn(4,2))
    # labels = torch.empty(4).random_(2)
    # probs, preds = torch.max(input, dim = -1)
    # correct = (preds == labels).type(torch.float)
    #
    # print(f'Preds:{preds}',  f'Labels:{labels}', f'Correct:{correct}',)
    # test = F1_Score(n_class = len(input[0]),  type = 'origin')
    # print(f'origin:{test(input, labels) == f1score(labels, preds)}')
