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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/model_tf/epoch1/config1.json_pretrainindex4_pre_train.model'
## preprocess parameters

####load node
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/anno/data/GSE130399_cortex/node1/'
                                         ,df_type ='node',
                                         order_cell=False)



###load edge
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/',df_type ='edge',order_cell=False)
gene_num = matrix2[list(matrix2.keys())[0]].columns
print(matrix2[list(matrix2.keys())[0]].shape)
tf = matrix2[list(matrix2.keys())[0]].index
gene = matrix2[list(matrix2.keys())[0]].columns
print(len(np.intersect1d(tf,gene[0:1103])))

###load label
label_dict = {}
edge_label = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/pp/label2/GSE130399_cortex.txt')
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
            y = label-1,
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
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/anno/result/'
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
data_name = 'GSE130399_cortex'
print('Done data')

"""
model define
"""
config_list = ['config1.json']
for config_name in config_list:

    config = JSON.decode('/storage/peiweikeLab/jiangjunyao/fatez/pre_train/tf_config/'+config_name)
    config['fine_tuner']['n_class'] = len(label_set)
    trainer = pre_trainer.Set(config, dtype=torch.float32,
                                      device=device,
                                      prev_model=model.Load(model_dir))
    fine_tuner_model = fine_tuner.Set(config, dtype = torch.float32
                                      ,device=device
                                      ,prev_model=trainer)
    fine_tuner_model.setup()
    early_stop = es.Monitor(tolerance=30, min_delta=0.01)
    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}\n-------------------------------")

        report_train = fine_tuner_model.train(train_dataloader,report_batch = True)
        print(report_train[-1:])

        report_test = fine_tuner_model.test(test_dataloader,report_batch = True)
        print(report_test[-1:])

        report_train.to_csv(data_save_dir + 'train-'+config_name+'-'+data_name+'.csv',
                            mode='a',header=False)
        report_test.to_csv(data_save_dir + 'test-'+config_name+'-'+data_name+'.csv',
                           mode='a',header=False)

        if early_stop(float(report_train[-1:]['Loss']),
                      float(report_test[-1:]['Loss'])):
            print("We are at epoch:", i)
            break
    torch.cuda.empty_cache()

if data_save:
    model.Save(
        fine_tuner_model.model,
        data_save_dir +model_name+ config_name+'fine_tune.model'
    )
