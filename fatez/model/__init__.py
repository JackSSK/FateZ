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



def Save(model, file_path:str = 'a.model',):
    """
    Saving a model

    :param model: the model to save
    :param file_path: model output path
    :param device: device to load model
    """
    model_type = str(type(model))
    if (re.search(r'fatez.*.Trainer*', model_type) or
        re.search(r'fatez.*.Tuner*', model_type)
        ):
        model.model = model.model.to('cpu')
        dict = {
            'type':model_type,
            'model':model.model.state_dict(),
            'optimizer':model.optimizer.state_dict(),
            'scheduler':model.scheduler.state_dict(),
        }
        torch.save(dict, file_path)
    else:
        raise Error('Not Supporting Save ' + model_type)
    return

def Load(file_path:str = 'a.model', mode:str = 'torch',):
    """
    Loading a model

    :param file_path: path to load model
    """
    model = None
    # PyTorch Loading method
    if mode == 'torch':
        dict = torch.load(file_path)
        print('Loaded:', dict['type'])
        return dict
    else:
        raise Error('Not Supporting Load Mode ' + mode)



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
