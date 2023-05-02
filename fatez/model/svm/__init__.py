#!/usr/bin/env python3
"""
Support vector machine based models

Note:
Since we would have multi-dim latent space, how shall we apply SVMs?
One SMV for each dim and combine outputs?
Fully connect then single SVM?
Use dense layer to project multi-dim into single dim?
YTD

author: jy
"""
import torch.nn as nn
from sklearn import svm



class SVC(nn.Module):
    """
    docstring for SVC.
    """

    def __init__(self, **kwargs):
        super(SVC, self).__init__()
        self.kwargs = kwargs
        self.model = svm.SVC(**self.kwargs)

    def forward(self, input, label):
        self.model.fit(input, label)
        return



class LinearSVC(nn.Module):
    """
    docstring for LinearSVC.
    """

    def __init__(self, **kwargs):
        super(LinearSVC, self).__init__()
        self.kwargs = kwargs
        self.model = svm.LinearSVC(**self.kwargs)

    def forward(self, input, label):
        return



class SVR(nn.Module):
    """
    docstring for SVR.
    """

    def __init__(self, **kwargs):
        super(SVR, self).__init__()
        self.kwargs = kwargs
        self.model = svm.SVR(**self.kwargs)

    def forward(self, input, label):
        return



class LinearSVR(nn.Module):
    """
    docstring for LinearSVR.
    """

    def __init__(self, **kwargs):
        super(LinearSVR, self).__init__()
        self.kwargs = kwargs
        self.model = svm.LinearSVR(**self.kwargs)

    def forward(self, input, label):
        return
