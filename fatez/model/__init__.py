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



def Save(model, file_path:str = 'a.model', save_full:bool = False,):
    """
    Saving a model

    :param model: the model to save
    :param file_path: model output path
    :param device: device to load model
    """
    model_type = str(type(model))
    if (re.search(r'fatez.*.Trainer*', model_type) or
        re.search(r'fatez.*.Tuner*', model_type) or
            re.search(r'fatez.*.Imputer*', model_type)
        ):
        model.model = model.model.to('cpu')
        if save_full:
            bert_model = model.model.bert_model.state_dict()
            encoder = None
            rep_embedder = None
        else:
            bert_model = None
            encoder = model.model.bert_model.encoder.state_dict()
            rep_embedder = model.model.bert_model.rep_embedder.state_dict()
        dict = {
            'type':model_type,
            'graph_embedder':model.model.graph_embedder.state_dict(),
            'gnn':model.model.gat.state_dict(),
            'encoder':encoder,
            'rep_embedder':rep_embedder,
            'bert_model':bert_model,
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

def Load_state_dict(net, state, load_opt_sch:bool = True):
    if load_opt_sch:
        net.optimizer.load_state_dict(state['optimizer'])
        net.scheduler.load_state_dict(state['scheduler'])

    # This part will be depreciated soon
    if 'model' in state:
        net.model.load_state_dict(state['model'])
        return
    net.model.graph_embedder.load_state_dict(state['graph_embedder'])
    net.model.gat.load_state_dict(state['gnn'])
    if state['bert_model'] is not None:
        net.model.bert_model.load_state_dict(state['bert_model'])
        # net.optimizer.load_state_dict(state['optimizer'])
        # net.scheduler.load_state_dict(state['scheduler'])
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
