#!/usr/bin/env python3
"""
This folder contains machine learning models.

author: jy, nkmtmsys
"""
import re
import tqdm
import torch
import torch.nn as nn
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
        re.search(r'fatez.*.Model', model_type) or
        re.search(r'fatez.model.bert.', model_type) or
        re.search(r'fatez.model.gat.Sparse_Model', model_type) or
        re.search(r'fatez.model.gat.GAT', model_type)
        ):
        torch.save(model.cpu(), file_path)
    else:
        raise Error('Not Supporting Save ' + model_type)
    return model.to(device)

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
    else:
        raise Error('Not Supporting Load Mode ' + mode)
    return model.to(device)



class Binning_Process(nn.Module):
    """
    Binning data output by GAT.

    Probably we can just multiply data into integers.
    Clustering is way too comp expensive.
    """

    def __init__(self, n_bin = None, config = None, **kwargs):
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
