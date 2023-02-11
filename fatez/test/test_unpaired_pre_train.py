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
from fatez.tool import EarlyStopping
from fatez.tool import model_training
import fatez.model as model
import fatez.model.gat as gat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
from sklearn.model_selection import train_test_split

"""
preprocess
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## preprocess parameters
pseudo_cell_num_per_cell_type = 5000
cluster_use =[1,4]


matrix1 = PreprocessIO.input_csv_dict_df(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500/node/')
matrix2 = pd.read_csv(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500/edge_matrix.csv'
    ,index_col=0)
m2 = torch.from_numpy(matrix2.to_numpy())
m2 = m2.to(torch.float32)
### samples and labels
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m1 = m1.to(torch.float32)
    samples.append([m1, m2])
labels = torch.from_numpy(np.repeat(range(len(cluster_use))
                                    ,len(matrix1)/len(cluster_use)))
labels = labels.long()
labels = labels.to(device)
###
"""
hyperparameters
"""
###############################
# Not tune-able
n_features = 2
n_class = 2
###############################
# General params
batch_size = 20
num_epoch = 100
lr = 1e-3
test_size = 0.3
early_stop_tolerance = 15
##############################
# GAT params
en_dim = 3                 # Embed dimension output by GAT
gat_n_hidden = 2            # Number of hidden units in GAT
gat_nhead = 0               # Number of attention heads in GAT
##############################
# BERT Encoder params
n_layer = 6                 # Number of Encoder Layers
bert_nhead = 3             # Attention heads
dim_ff = 2                  # Dimension of the feedforward network model.
bert_n_hidden = 2           # Number of hidden units in classification model.
##############################
# BERT pretrain params
masker_ratio = 0.5
n_bin = 100
##############################
data_save = True
data_save_dir = 'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\model/nhead0_endim3_nhidden2_lr-3_epoch100_batch_size_20_lr-3_epoch100/'
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

pretrain_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=samples, labels=labels),
    batch_size=batch_size,
    shuffle=True
)
"""
model define
"""

model_gat = gat.GAT(
    d_model = n_features,
    en_dim = en_dim,
    nhead = gat_nhead,
    device = device,
    n_hidden = gat_n_hidden,
)
bert_encoder = bert.Encoder(
    d_model = model_gat.en_dim,
    n_layer = n_layer,
    nhead = bert_nhead,
    dim_feedforward = dim_ff,
    device = device,
)
test_model = fine_tuner.Model(
    gat = model_gat,
    bin_pro = model.Binning_Process(n_bin = 100),
    bert_model = bert.Fine_Tune_Model(
        bert_encoder,
        n_class = n_class,
        n_hidden = bert_n_hidden,
    ),
    device = device,
)
masker = model.Masker(ratio = masker_ratio)
model_gat.to(device)
bert_encoder.to(device)
pre_train_model = pre_trainer.Model(
    gat = model_gat,
    masker = masker,
    bin_pro = model.Binning_Process(n_bin = n_bin),
    bert_model = bert.Pre_Train_Model(
        bert_encoder, n_bin = n_bin, n_dim_node = n_features
    )
    device = device,
)
### adam and CosineAnnealingWarmRestarts
optimizer = torch.optim.Adam(
    test_model.parameters(),
    lr = lr,
    weight_decay = 1e-3
)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0 = 2,
    T_mult=2,
    eta_min = lr / 50
)

early_stopping = EarlyStopping.EarlyStopping(
    tolerance = early_stop_tolerance,
    min_delta = 10
)
model_gat.to(device)
bert_encoder.to(device)
test_model.to(device)
pre_train_model.to(device)
"""
pre-training
"""
all_loss = list()
for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loss = model_training.pre_training(pretrain_dataloader,
                                                  pre_train_model,
                                                  optimizer,
                                                  device=device)
    print(
     f"epoch: {epoch+1}, train_loss: {train_loss}")
    all_loss.append(train_loss.tolist())
    scheduler.step()


"""
fine-tune traning
"""
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
#     early_stopping(train_loss, test_loss)
#     if early_stopping.early_stop:
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
# """
# testing
# """
