#!/usr/bin/env python3
"""
This is the playground for Multiomics rebuilder.

author: jy
"""
import os
import pandas as pd
import torch
import copy
import sys
from torch.utils.data import DataLoader
import torch_geometric.data as pyg_d
import fatez.lib as lib
import fatez.model as model
import fatez.tool.JSON as JSON
import fatez.tool.PreprocessIO as PreprocessIO
import fatez.process.early_stopper as es
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
from fatez.process.scale_network import scale_network
from sklearn.model_selection import train_test_split

import tracemalloc
import warnings
from pkg_resources import resource_filename
import shap
import torch.nn as nn
import fatez.model.mlp as mlp
import fatez.model.gnn as gnn
import fatez.model.criterion as crit
import fatez.model.position_embedder as pe
import fatez.process as process
import fatez.process.worker as worker



data_header = '../data/rd_fake_data/'
node_dir = data_header+'all_node/'
matrix1 = PreprocessIO.input_csv_dict_df(
            node_dir,
            df_type='node', order_cell=False)
hep_edge = pd.read_table(data_header+'hep_edge.txt')
endo_edge = pd.read_table(data_header+'endo_edge.txt')
edge_dict = {'hep':hep_edge,'endo':endo_edge}
for i in list(edge_dict.keys()):
    edge = scale_network(edge_dict[i])
    edge = torch.from_numpy(edge)
    edge = edge.to(torch.float32)
    edge_dict[i] = edge
edge_label = pd.read_table(data_header+'fake_label.txt')
edge_label.index = edge_label['sample']
samples = []
for i in range(len(matrix1)):
    sample_name = list(matrix1.keys())[i]
    m1 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
    label = edge_label['label'][str(sample_name)]
    if sample_name[0:3] == 'hep':
        key_use = 'hep'
    else:
        key_use = 'endo'
    inds, attrs = lib.get_sparse_coo(edge_dict[key_use])
    samples.append(
        pyg_d.Data(
            x=m1,
            edge_index=inds,
            edge_attr=attrs,
            y=label,
            shape=edge_dict[key_use].shape,
        )
    )


"""
preprocess
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

matrix1 = PreprocessIO.input_csv_dict_df(
    data_header + 'endo_node',
    df_type ='node',
    order_cell = False,
)

print(matrix1)

###load edge
matrix2 = PreprocessIO.input_csv_dict_df(
    'D:\\Westlake\\pwk lab\\fatez\\test_0613/edge/',
    df_type ='edge',
    order_cell = False,
)
gene_num = matrix2[list(matrix2.keys())[0]].columns
