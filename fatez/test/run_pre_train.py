import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import fatez.lib as lib
import fatez.tool.JSON as JSON
from fatez.tool import PreprocessIO
from fatez.tool import model_training
import fatez.model as model
import fatez.model.gat as gat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
from sklearn.model_selection import train_test_split
import time
t1 = time.time()

"""
preprocess
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
## preprocess parameters
cluster_use =[1,4]
###
file_list = os.listdir('/storage/peiweikeLab/jiangjunyao/fatez/sample1000/')
file_list = file_list[0:10000]
matrix1 = {}
for i in file_list:
    sample_name = i.split('#')[0]
    sample = pd.read_csv('/storage/peiweikeLab/jiangjunyao/fatez/sample1000/' + i
                         , header=0, index_col=0)
    matrix1[i] = sample

###
edge=os.listdir('/storage/peiweikeLab/jiangjunyao/fatez/sample1000/edge')
edge_dict = {}
for i in edge:
    matrix2 = pd.read_csv('/storage/peiweikeLab/jiangjunyao/fatez/sample1000/edge/'+i,index_col=0)
    matrix2 = matrix2.replace(np.nan,0)
    m2 = torch.from_numpy(matrix2.to_numpy())
    m2 = m2.to(torch.float32)
    edge_name = i.split('#')[0]
    print(edge_name)
    edge_dict[i.split('#')[0]] = m2

print('----')
### samples and labels
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m1 = m1.to(torch.float32)
    sample_name=list(matrix1.keys())[i]
    sample_name = sample_name.split('#')[0]
    if sample_name in edge_dict.keys():

        m2 = edge_dict[sample_name]
        samples.append([m1, m2])
print(len(samples))
labels = [0]*len(samples)
labels = np.array(labels)
labels = torch.from_numpy(labels)
labels = labels.long()
labels = labels.to(device)


###
"""
hyperparameters
"""
###############################
# General params
num_epoch = 100
test_size = 0.3
pre_train_adj = True
##############################
# config file
config = JSON.decode('test_config.json')

# factory_kwargs should be adjusted upon useage
factory_kwargs = {'device': device, 'dtype': torch.float32,}



##############################
data_save = False
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/model/pre_train/'
outgat_dir = data_save_dir+'out_gat/'
if  pre_train_adj:
    config['pre_trainer']['n_dim_adj'] = matrix2.shape[1]

#os.makedirs(outgat_dir )
"""
dataloader
"""
X_train,X_test,y_train,y_test = train_test_split(
    samples,labels,test_size=test_size,train_size = 1-test_size,random_state=0)
train_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=X_train, labels=y_train),
    batch_size = config['batch_size'],
    shuffle=True
)

test_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=X_test, labels=y_test),
    batch_size = config['batch_size'],
    shuffle=True
)

pretrain_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=samples, labels=labels),
    batch_size = config['batch_size'],
    shuffle=True
)


"""
pre-training
"""
pre_train_model = pre_trainer.Set(config, factory_kwargs)
all_loss = list()
for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loss = pre_train_model.train(pretrain_dataloader, print_log = True)
    print(f"epoch: {epoch+1}, train_loss: {train_loss}")
    all_loss.append(train_loss.tolist())

if data_save:
    # model.Save(
    #     pre_train_model.model.bert_model.encoder,
    #     data_save_dir + 'bert_encoder.model'
    # )
    # # Use this to save whole bert model
    # model.Save(
    #     pre_train_model.model.bert_model,
    #     data_save_dir + 'bert_fine_tune.model'
    # )
    # model.Save(
    #     pre_train_model.model.gat,
    #     data_save_dir + 'gat.model'
    # )
    model.Save(
        pre_train_model.model,
        data_save_dir + 'full_pre_train.model'
    )


print(all_loss)
with open(data_save_dir+'loss.txt', 'w+')as f:
    for i in all_loss:
         f.write(str(i)+'\n')

# Save JSON one might be easier
JSON.encode(config, data_save_dir + 'config.txt')

"""
fine-tune traning
"""

# test_model = fine_tuner.Tuner(
#     gat = pre_train_model.model.gat,
#     encoder = pre_train_model.model.bert_model.encoder ,
#     rep_embedder = pre_train_model.model.rep_embedder,
#     **config['fine_tuner'],
#     **factory_kwargs,
# ).model

# all_loss = list()
# for epoch in range(num_epoch):
#     print(f"Epoch {epoch + 1}\n-------------------------------")
#     out_gat_data,\
#     train_loss,train_acc = model_training.training(train_dataloader,model_gat,
#                                                   test_model,
#                                                   nn.CrossEntropyLoss(),
#                                                   optimizer,device=device)
#     print(
#      f"epoch: {epoch+1}, train_loss: {train_loss}, train accuracy: {train_acc}")
#     all_loss.append(train_loss.tolist())
#     scheduler.step()
#     test_loss,test_acc = model_training.testing(test_dataloader,
#                                                test_model, nn.CrossEntropyLoss()
#                                                , device=device)
#     print(
#         f"epoch: {epoch+1}, test_loss: {test_loss}, test accuracy: {test_acc}")
#
#     if early_stopping(train_loss, test_loss):
#         print("We are at epoch:", i)
#         break
# if data_save:
#     model.Save(
#         test_model.bert_model.encoder,
#         data_save_dir + 'bert_encoder.model'
#     )
#     # Use this to save whole bert model
#     model.Save(
#         test_model.bert_model,
#         data_save_dir + 'bert_fine_tune.model'
#     )
#     model.Save(
#         model_gat,
#         data_save_dir + 'gat.model'
#     )
# print(all_loss)
#
# """
# # You are making a new model with untraiend classficiation MLP
# # So, even if you test it without save and load, it won't perform well.
# # Go check line #228-230
# test = bert.Fine_Tune_Model(test_model.bert_model.encoder, n_class = 2)
# model.Save(test, data_save_dir+'bert_fine_tune.model')
# """
#
# JSON.encode(
#     out_gat_data,
#     outgat_dir + str(epoch) + '.js'
# )
t2 = time.time()
print(t2-t1)
# """
# testing
# """
