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


"""
preprocess
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## preprocess parameters
print(device)
####load node
#matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/fine_tune_node_bin20/GSE205117_NMFSM/',df_type ='node')
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/node1/',df_type ='node')


###load edge
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/edge/',df_type ='edge')
gene_num = matrix2[list(matrix2.keys())[0]].columns
print(matrix2[list(matrix2.keys())[0]].shape)
tf = matrix2[list(matrix2.keys())[0]].index
gene = matrix2[list(matrix2.keys())[0]].columns
print(len(np.intersect1d(tf,gene[0:1103])))

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
    inds, attrs = lib.get_sparse_coo(matrix2[key_use])
    """
    Using PyG Data object
    """
    samples.append(
        pyg_d.Data(
            x = m1,
            edge_index = inds,
            edge_attr = attrs,
            y = label,
            shape = matrix2[key_use].shape,
        )
    )

    # samples.append([m1, m2])

###
"""
hyperparameters
"""
###############################
# General params
batch_size = 10
num_epoch = 1
test_size = 0.3

##############################
data_save = True
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/fine_tune_big/'
"""
dataloader
"""
X_train,X_test,y_train,y_test = train_test_split(
    samples,
    labels,
    test_size=test_size,
    train_size = 1-test_size,
    random_state=0
)
train_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = X_train),
    batch_size = batch_size,
    collate_fn = lib.collate_fn,
    shuffle=True
)

test_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=X_test),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=True
)
data_name = 'GSE205117_NMFSM_bin20'


"""
model define
"""
config_name = 'config_big1.json'
for i in range(1):

    config = JSON.decode('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/new_version_config/'+config_name)
    config['fine_tuner']['n_class'] = len(set(label_check))
    print(config['input_sizes'])
    """
    traning
    """
    factory_kwargs = {'device': device, 'dtype': torch.float32, }
    trainer = pre_trainer.Set(config, factory_kwargs)
    trainer.model = nn.DataParallel(trainer.model)
    early_stop = es.Monitor(tolerance=30, min_delta=0.01)
    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}\n-------------------------------")

        report_train = trainer.train(train_dataloader, report_batch = False)

        print(report_train[-1:])


        report_train.to_csv(data_save_dir + 'train-'+config_name+'-'+data_name+'.csv',
                            mode='a',header=False)


if data_save:
    model.Save(
        fine_tuner_model.model,
        data_save_dir +model_name+ config_name+'fine_tune.model'
    )
