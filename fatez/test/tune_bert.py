import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fatez.lib as lib
import fatez.tool.JSON as JSON
from fatez.tool import PreprocessIO
import fatez.model as model
import fatez.model.transformer as transformer
import fatez.model.gat as gat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.model.position_embedder as pe
import fatez.process.pre_trainer as pre_trainer
from sklearn.model_selection import train_test_split
import fatez.process.early_stopper as es
from fatez.process.scale_network import scale_network
os.chdir("/storage/peiweikeLab/jiangjunyao/fatez/FateZ/fatez/test")
"""
preprocess
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## preprocess parameters

####load node
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/node_test/',df_type ='node')
###load edge
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/edge/',df_type ='edge')
gene_num = matrix2[list(matrix2.keys())[0]].columns
###load label
label_dict = {}
label_list = os.listdir('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/label')
for i in label_list:
    label_dir = '/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/label/'+i
    edge_label1 = pd.read_table(label_dir,index_col=None)
    edge_label1.index = edge_label1['sample']
    label_name = i.split('.txt')[0]
    label_dict[label_name] = edge_label1
### scale edge
for i in list(matrix2.keys()):
    edge = scale_network(matrix2[i])
    edge = torch.from_numpy(edge)
    edge = edge.to(torch.float32)
    matrix2[i] = edge
### samples and labels
samples = []
labels = []
for i in range(len(matrix1)):
    sample_name = list(matrix1.keys())[i]
    m1 = matrix1[sample_name]
    m1 = torch.from_numpy(m1.to_numpy())
    m1 = m1.to(torch.float32)
    dict_key = sample_name.split('#')[0]
    edge_label = label_dict[dict_key]
    label = edge_label['label'][str(sample_name)]
    labels.append(label)
    edge_name = edge_label.loc[sample_name,'label']
    key_use  = dict_key+'#'+str(edge_name)
    m2 = matrix2[key_use]
    print(sample_name)
    print(edge_name)
    samples.append([m1, m2])

labels = torch.tensor([0]*len(matrix1))
labels = labels.long()
labels = labels.to(device)
print(list(matrix1.keys()))
print(labels)
###
"""
hyperparameters
"""
###############################
# General params
batch_size = 10
num_epoch = 10
##############################
data_save = True
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/pre_train_model/'
"""
dataloader
"""
train_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=samples, labels=labels),
    batch_size=batch_size,
    shuffle=False
)



"""
model define
"""
config = JSON.decode('test_config_cnnhyb.json')

config['input_sizes'][0][0] = batch_size
config['input_sizes'][1][0] = batch_size
config['input_sizes'][1][2] = gene_num

"""
Debug area
"""
print(config['input_sizes'])

"""
traning
"""
config["rep_embedder"]['type'] = 'abs'
config['pre_trainer']['train_adj'] = True
rep_type = config["rep_embedder"]['type']
n_layer_list = [6]
nhead_list = [4]
dim_feedforward_list = [512]
d_model_list = [128,256]
for p1 in d_model_list:
    for p2 in dim_feedforward_list:
        for p3 in nhead_list:
            for p4 in n_layer_list:
                config['gnn']['params']['en_dim'] = p1
                config['encoder']['d_model'] = p1
                config["rep_embedder"]['params']["n_dim"] = p1
                config['encoder']['dim_feedforward'] = p2
                config['encoder']['nhead'] = p3
                config['encoder']['n_layer'] = p4
                factory_kwargs = {'device': device, 'dtype': torch.float32, }
                trainer = pre_trainer.Set(config, factory_kwargs)
             
                #fine_tuner_model = fine_tuner.Set(config, factory_kwargs)
                early_stop = es.Monitor(tolerance=30, min_delta=0.01)
                para_name = 'd_model-'+str(p1)+'-dff-'+str(p2)+'-nhead-'+str(p3)+'-n_layer-'+str(p4) + '-rep-' +rep_type + '-batchsize-' + str(batch_size)
                print(para_name)
                for epoch in range(num_epoch):
                    print(f"Epoch {epoch+1}\n-------------------------------")

                    report_train = trainer.train(train_dataloader, report_batch = True)
                    print(report_train[-1:])

                    report_train.to_csv(data_save_dir + 'pre-train-'+para_name+'.csv',
                                        mode='a',header=False)

                    if early_stop(float(report_train[-1:]['Loss']),
                                  float(report_test[-1:]['Loss'])):
                        print("We are at epoch:", i)
                        break
                torch.cuda.empty_cache()

                if data_save:
                    model.Save(
                        trainer,
                        data_save_dir + para_name + '#pre_train.model'
                    )
