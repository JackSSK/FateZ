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

def to_state_dict(model, save_full:bool = False):
    """
    Convert a model to state dict
    """
    model_type = str(type(model))
    if (
            re.search(r'fatez.*.Trainer*', model_type) or
            re.search(r'fatez.*.Tuner*', model_type) or
            re.search(r'fatez.*.Imputer*', model_type)
        ):
        model.model = model.model.to('cpu')
        state_dict = model.model.to_state_dict(save_full)
        state_dict['type'] = model_type
        state_dict['optimizer'] = model.optimizer.state_dict()
        state_dict['scheduler'] = model.scheduler.state_dict()
        return state_dict
    else:
        model = model.to('cpu')
        return model.state_dict()

def Save(model, file_path:str = 'a.model', save_full:bool = False,):
    """
    Saving a model

    :param model: the model to save
    :param file_path: model output path
    """
    torch.save(to_state_dict(model, save_full), file_path)
    return

def Load(file_path:str = 'a.model', mode:str = 'torch',):
    """
    Loading a model

    :param file_path: path to load model
    """
    model = None
    # PyTorch Loading method
    if mode == 'torch':
        state_dict = torch.load(file_path)
        print('Loaded:', state_dict['type'])
        return state_dict
    else:
        raise Error('Not Supporting Load Mode ' + mode)

def Load_state_dict(net, state, load_opt_sch:bool = False):
    if load_opt_sch:
        net.optimizer = state['optimizer']
        net.scheduler = state['scheduler']

    # This part will be depreciated soon
    if 'model' in state:
        net.model.load_state_dict(state['model'])
        return
    net.model.graph_embedder.load_state_dict(state['graph_embedder'])
    net.model.gat.load_state_dict(state['gnn'])
    if state['bert_model'] is not None:
        net.model.bert_model.load_state_dict(state['bert_model'])
    else:
        net.model.bert_model.encoder.load_state_dict(state['encoder'])
        net.model.bert_model.rep_embedder.load_state_dict(state['rep_embedder'])




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
