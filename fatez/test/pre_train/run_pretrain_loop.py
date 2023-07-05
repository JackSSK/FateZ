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
import torch.nn as nn
import numpy as np
import psutil

import fatez.process.worker as worker
"""
hyperparameters
"""
###############################
# General params
batch_size = 10
num_epoch = 1
test_size = 0.3
device = [0]
worker.setup(device)
##############################
data_save = True
epoch = 'epoch1'
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/'
"""
preprocess
"""
def bytes_to_gb(bytes):
    return bytes / (1024**3)
###load edge
edge = ['GSE209610#13.csv','GSE209610#18.csv','GSE209610#4.csv']
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge/',df_type ='edge',cell_use = edge)
gene_num = matrix2[list(matrix2.keys())[0]].columns
print(matrix2[list(matrix2.keys())[0]].shape)
tf = matrix2[list(matrix2.keys())[0]].index
gene = matrix2[list(matrix2.keys())[0]].columns
print(len(np.intersect1d(tf,gene[0:1103])))

print('total edge num: ',len(matrix2))
###load label
label_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pp/label2/'
label_dict = {}
label_list = os.listdir(label_dir)
for i in label_list:
    label_dir2 = '/storage/peiweikeLab/jiangjunyao/fatez/pp/label2/'+i
    edge_label1 = pd.read_table(label_dir2,index_col=None)
    edge_label1.index = edge_label1['sample']
    label_name = i.split('.txt')[0]
    label_dict[label_name] = edge_label1
### scale edge
for i in list(matrix2.keys()):
    edge = scale_network(matrix2[i])
    edge = torch.from_numpy(edge)
    edge = edge.to(torch.float32)
    matrix2[i] = edge


### load idx
pretrain_idx = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/pp/pretrain_idx.txt')
#result = pretrain_idx.loc[pretrain_idx['idx'] == pretrain_idx_use, 'cell'].tolist()

all_idx = list(set(pretrain_idx['idx'].tolist()))
all_idx = [x for x in all_idx if x != 1]
for pretrain_idx_use in all_idx:
    memory = psutil.virtual_memory()
    # result = pretrain_idx.loc[pretrain_idx['idx'] == pretrain_idx_use, 'cell'].tolist()
    result = pretrain_idx.loc[0:2, 'cell'].tolist()
    #result = pretrain_idx.loc[1:200, 'cell'].tolist()
    print('we are in pretrain idex:',pretrain_idx_use)
    print('input node length:',len(result))
    print('ram_use:',bytes_to_gb(memory.used))
    ####load node
    matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/pp/node_bin20/',df_type ='node',cell_use = result)


    samples = []
    ### samples and labels
    for i in range(len(matrix1)):
        sample_name = list(matrix1.keys())[i]
        m1 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
        dict_key = sample_name.split('#')[0]
        edge_label = label_dict[dict_key]
        label = edge_label['label'][str(sample_name)]
        edge_name = edge_label.loc[sample_name,'label']
        key_use  = dict_key+'#'+str(edge_name)
        inds, attrs = lib.get_sparse_coo(matrix2[key_use])
        """
        Using PyG Data object
        """
        samples.append(
            pyg_d.Data(
                x = m1,
                edge_index = inds,
                edge_attr = attrs,
                y = 0,
                shape = matrix2[key_use].shape,
            )
        )


    """
    dataloader
    """
    train_dataloader = DataLoader(
        lib.FateZ_Dataset(samples = samples),
        batch_size = batch_size,
        collate_fn = lib.collate_fn,
        shuffle=True
    )

    print('data loader finish!')


    """
    model define
    """
    config_name = 'config1.json'


    config = JSON.decode('/storage/peiweikeLab/jiangjunyao/fatez/pre_train/config/'+config_name)
    """
    traning
    """
    trainer = pre_trainer.Set(
        config,
        prev_model = model.Load(model_dir),
        dtype = torch.float32,
        device = device
    )
    model_dir = data_save_dir +'model/'+ epoch +'/'+config_name+'_pretrainindex'+str(pretrain_idx_use-1)+'_pre_train.model'

    for j in range(num_epoch):
        print(f"Epoch {j+1}\n-------------------------------")

        report_train = trainer.train(
            train_dataloader,
            report_batch = True
            )

        print(report_train[-1:])

        report_train.to_csv(
            data_save_dir + 'reporch/epoch1/' + config_name + '_pretrainindex' + str(
                pretrain_idx_use) + '.csv',
            mode='a',
            header=False
        )

    if data_save:
        model.Save(
            trainer.model,
            data_save_dir + 'model/epoch1/' + config_name + '_pretrainindex' + str(
                pretrain_idx_use) + '_pre_train.model'
        )
