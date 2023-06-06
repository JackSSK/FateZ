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




"""
preprocess
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## preprocess parameters

####load node
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/fine_tune_node_binrna20_atacnor/GSE205117_NMFSM/',df_type ='node')


###load edge
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/edge/',df_type ='edge')
gene_num = matrix2[list(matrix2.keys())[0]].columns


###load label
label_dict = {}
edge_label = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/label/GSE205117_NMFSM.txt')
label_check = edge_label['label'].values
label_set = list(set(label_check))

print(label_set)
edge_label.index = edge_label['sample']
### scale edge
for i in list(matrix2.keys()):
    edge = scale_network(matrix2[i])
    edge = torch.from_numpy(edge)
    edge = edge.to(torch.float32)
    matrix2[i] = edge

labels = torch.tensor([0]*len(matrix1))
### samples and labels
samples = []
# labels = []
for i in range(len(matrix1)):
    sample_name = list(matrix1.keys())[i]
    m1 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
    dict_key = sample_name.split('#')[0]
    #edge_label = label_dict[dict_key]
    label = edge_label['label'][str(sample_name)]
    # labels.append(label)
    edge_name = edge_label.loc[sample_name,'label']
    key_use  = dict_key+'#'+str(edge_name)
    m2 = matrix2[key_use]
    """
    Using PyG Data object
