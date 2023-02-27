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
import fatez.model.gat as gat


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
        re.search(r'fatez.*.Model', model_type) or
        re.search(r'fatez.model.bert.', model_type) or
        re.search(r'fatez.model.gat.Sparse_Model', model_type) or
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

def Set_GAT(config:dict = None, factory_kwargs:dict = None):
    """
    Set up GAT model based on given config.
    """
    if config['gat']['type'] == 'GAT':
        return gat.Model(**config['gat']['params'], **factory_kwargs)
    elif config['gat']['type'] == 'SGAT':
        return gat.Sparse_Model(**config['gat']['params'], **factory_kwargs)
    else:
        raise model.Error('Unknown GAT type')



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



class Position_Encoder(nn.Module):
    """
    Absolute positional encoding.
    SAT tested transformer + Random Walk PE, maybe we can try as 
    """

    def __init__(self, n_features, n_dim):
        super(Position_Encoder, self).__init__()
        self.encoder = nn.Embedding(n_features, n_dim)

    def forward(self, x):
        return x + self.encoder(torch.arange(x.shape[1], device = x.device))
