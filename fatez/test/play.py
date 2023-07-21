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
