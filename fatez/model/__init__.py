#!/usr/bin/env python3
"""
This folder contains machine learning models.

author: jy, nkmtmsys
"""
import re
import tqdm
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from sklearn import cluster, datasets, mixture



class Error(Exception):
    """
    Error handling
    """
    pass



def Save(model, file_path:str = 'a.model', device:str = 'cpu',):
    """
    Saving a model

    :param model: the model to save
    :param file_path: model output path
    :param device: device to load model
    """
    model_type = str(type(model))

    if (re.search(r'torch.nn.modules.', model_type) or
        re.search(r'fatez.model.bert.', model_type) or
        re.search(r'fatez.model.sparse_gat.Spare_GAT', model_type) or
        re.search(r'fatez.model.gat.GAT', model_type)
        ):
        torch.save(model.cpu(), file_path)
        model = model.to(device)
    else:
        raise Error('Not Supporting Save ' + model_type)
    return model

def Load(file_path:str = 'a.model', mode:str = 'torch', device:str = 'cpu',):
    """
    Loading a model

    :param file_path: path to load model
    :param device: device to load model
    """
    model = None
    # PyTorch Loading method
    if mode == 'torch':
        model = torch.load(file_path)
        model = model.to(device)
    else:
        raise Error('Not Supporting Load Mode ' + mode)
    return model



class Masker(object):
    """
    Make masks for BERT encoder input.
    """
    def __init__(self, ratio, seed = None):
        super(Masker, self).__init__()
        self.ratio = ratio
        self.seed = seed
        self.choices = None

    def make_2d_mask(self, size, ):
        # Set random seed
        if self.seed is not None:
            random.seed(self.seed)
            self.seed += 1
        # Make tensors
        answer = torch.ones(size)
        mask = torch.zeros(size[-1])
        # Set random choices to mask
        choices = random.choices(range(size[-2]), k = int(size[-2]*self.ratio))
        assert choices is not None
        self.choices = choices
        # Make mask
        for ind in choices:
            answer[ind] = mask
        return answer

    def mask(self, input, factory_kwargs = None):
        mask = self.make_2d_mask(input[0].size())
        if factory_kwargs is not None:
            mask = mask.to(factory_kwargs['device'])
        try:
            return torch.multiply(input, mask)
        except:
            raise Error('Something else is wrong')



class Binning_Process(nn.Module):
    """
    Binning data output by GAT.

    Probably we can just multiply data into integers.
    Clustering is way too comp expensive.
    """

    def __init__(self, n_bin, config = None):
        super(Binning_Process, self).__init__()
        self.n_bin = n_bin
        # self.config = config
        # average_linkage = cluster.AgglomerativeClustering(
        #     linkage = 'ward',
        #     affinity = 'euclidean',
        #     n_clusters = self.n_bin,
        #     connectivity = connectivity,
        # )

    def forward(self, input):
        # return int(input * self.n_bin)
        return input



class LR_Scheduler(object):
    """
    Automatically adjust learning rate based on learning steps.
    """

    def __init__(self,
        optimizer,
        n_features:int = None,
        n_warmup_steps:int = None
        ):
        """
        :param optimizer: <Default = None>
            The gradient optimizer.

        :param n_features: <int Default = None>
            The number of expected features in the model input.

        :param n_warmup_steps: <int Default = None>
            Number of warming up steps. Gradually increase learning rate.
        """

        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_learning_rate = np.power(n_features, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
            ]
        )

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_learning_rate * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
