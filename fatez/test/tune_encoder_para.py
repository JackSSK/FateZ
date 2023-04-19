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
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_para/nmf_sm_200/node/',df_type ='node')
###load edge
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_para/nmf_sm_200/ct_edge/',df_type ='edge')
edge_label = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/tune_para/nmf_sm_200/label.txt')
edge_label.index = edge_label['sample']
for i in list(matrix2.keys()):
    print(matrix2[i].columns[0:100])
    print(matrix2[i].index[0:100])
    edge = scale_network(matrix2[i])
    edge = torch.from_numpy(edge)
    edge = edge.to(torch.float32)
    edge = edge.to_sparse()
    matrix2[i] = edge
    print(matrix2[i])
### samples and labels
samples = []
sample_name = list(matrix1.keys())[1]
print(matrix1[sample_name].index[0:100])
sample_name = list(matrix1.keys())[2]
print(matrix1[sample_name].index[0:100])
for i in range(len(matrix1)):
    sample_name = list(matrix1.keys())[i]
    m1 = matrix1[sample_name]
    m1 = torch.from_numpy(m1.to_numpy())
    m1 = m1.to(torch.float32)
    edge_name = edge_label.loc[sample_name,'label']
    m2 = matrix2[str(edge_name)]
    m1 = m1.to_sparse()
    print(sample_name)
    print(edge_name)
    samples.append([m1, m2])

### add ladbel
a2=pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/tune_para/nmf_sm_200/label.txt')
a2.index = a2['sample']
labels = a2['label'][name1].values
labels = torch.from_numpy(labels)
labels = labels.long()
labels = labels.to(device)
print(name1)
print(labels)
###
"""
hyperparameters
"""
###############################
# General params
batch_size = 10
num_epoch = 10
test_size = 0.3

##############################
data_save = False
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/tune_para/test_nmd_ct_reorder/'
outgat_dir = data_save_dir+'out_gat/'
#os.makedirs(outgat_dir )
"""
dataloader
"""
X_train,X_test,y_train,y_test = train_test_split(
    samples,labels,test_size=test_size,train_size = 1-test_size,random_state=0)
train_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=X_train, labels=y_train),
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=X_test, labels=y_test),
    batch_size=batch_size,
    shuffle=True
)



"""
model define
"""
config = JSON.decode('test_config_cnnhyb.json')


"""
Debug area
"""
print(config['input_sizes'])


"""
traning
"""
config["rep_embedder"]['type'] = 'abs'
rep_type = config["rep_embedder"]['type']
n_layer_list = [6]
nhead_list = [2]
dim_feedforward_list = [4]
d_model_list = [4]
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
                fine_tuner_model = fine_tuner.Set(config, factory_kwargs)
                early_stop = es.Monitor(tolerance=30, min_delta=0.01)
                para_name = 'd_model-'+str(p1)+'-dff-'+str(p2)+'-nhead-'+str(p3)+'-n_layer-'+str(p4) + '-rep-' +rep_type + '-batchsize-' + str(batch_size)
                print(para_name)
                for epoch in range(num_epoch):
                    print(f"Epoch {epoch+1}\n-------------------------------")

                    report_train = fine_tuner_model.train(train_dataloader,)
                    print(report_train[-1:])

                    report_test = fine_tuner_model.test(test_dataloader,)
                    print(report_test[-1:])

                    report_train.to_csv(data_save_dir + 'train-'+para_name+'.csv',
                                        mode='a',header=False)
                    report_test.to_csv(data_save_dir + 'test-'+para_name+'.csv',
                                       mode='a',header=False)

                    if early_stop(float(report_train[-1:]['Loss']),
                                  float(report_test[-1:]['Loss'])):
                        print("We are at epoch:", i)
                        break
                torch.cuda.empty_cache()

                if data_save:
                    model.Save(
                        fine_tuner_model.model,
                        data_save_dir + 'fine_tune.model'
                    )
