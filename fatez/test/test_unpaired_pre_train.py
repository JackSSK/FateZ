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
from fatez.process import early_stopper
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
batch_size = 10
num_epoch = 20
lr = 1e-4
test_size = 0.3
early_stop_tolerance = 15
pre_train_adj = True
##############################
# config file

config = {
    "batch_size": batch_size,
    "epoch": num_epoch,
    "gat": {
        "type": "GAT",
        "params": {
            "d_model": n_features,  # Feature dim
            "en_dim": 4,            # Embed dimension output by GAT
            "n_hidden": 2,          # Number of hidden units in GAT
            "nhead": 0              # Number of attention heads in GAT
        }
    },
    "encoder": {
        "d_model": 4,               # == gat params en_dim
        "n_layer": 6,               # Number of Encoder Layers
        "nhead": 4,                 # Attention heads
        "dim_feedforward": 4        # Dimension of the feedforward network model.
    },
    "pos_embedder": {
        "type": "Skip",             # Not using any positional embedding method
        # "type": "ABS",              # Absolute positional embedding
        "params": {
            "n_embed": 100,         # Number of TFs
            "n_dim": n_features,
        }
    },
    "masker": {
        "ratio": 0.15
    },
    "pre_trainer": {
        "n_dim_adj": None,
        "lr": lr,
        "weight_decay": 1e-3,
        "sch_T_0": 2,
        "sch_T_mult": 2,
        "sch_eta_min": lr / 50,
    },
    "fine_tuner": {
        "n_class": n_class,
        "clf_type": 'MLP',
        "clf_params": {
            "n_hidden": 2,
        },
        "lr": lr,
        "weight_decay": 1e-3,
        "sch_T_0": 2,
        "sch_T_mult": 2,
        "sch_eta_min": lr / 50,
    }
}

# factory_kwargs should be adjusted upon useage
factory_kwargs = {'device': device, 'dtype': torch.float32,}



##############################
data_save = True
data_save_dir = 'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\pre_train_model/0217_pretrain_adj'
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
model define
Code below should looks more explicit
"""
pre_train_model = pre_trainer.Set(config, factory_kwargs)
test_model = fine_tuner.Tuner(
    gat = pre_train_model.model.gat,
    encoder = pre_train_model.model.encoder,
    pos_embedder = pre_train_model.model.pos_embedder,
    **config['fine_tuner'],
    **factory_kwargs,
).model



#####################################################################
"""
These parts won't be needed later
"""
test_model = fine_tuner.Model(
    gat = pre_train_model.model.gat,
    pos_embedder = pre_train_model.model.pos_embedder,
    bert_model = bert.Fine_Tune_Model(
        encoder = pre_train_model.model.encoder,
        n_class = n_class,
        n_hidden = config['fine_tuner']['n_hidden'],
    ),
    device = device,
)

pre_train_model = pre_trainer.Model(
    gat = pre_train_model.model.gat,
    masker = pre_train_model.model.masker,
    pos_embedder = pre_train_model.model.pos_embedder,
    bert_model = pre_train_model.model.bert_model,
    device = device,
)
### adam and CosineAnnealingWarmRestarts
optimizer = torch.optim.Adam(
    pre_train_model.parameters(),
    lr = lr,
    weight_decay = 1e-3
)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0 = 2,
    T_mult=2,
    eta_min = lr / 50
)

# model_gat.to(device)
# bert_encoder.to(device)
# test_model.to(device)
pre_train_model.to(device)
######################################################################

early_stopping = early_stopper.Monitor(
    tolerance = early_stop_tolerance,
    min_delta = 10
)

"""
pre-training
"""
all_loss = list()
for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loss = model_training.pre_training(
        pretrain_dataloader,
        pre_train_model,
        optimizer,
        device=device
    )
    scheduler.step()

    # Try this one later
    # train_loss = pre_train_model.train(pretrain_dataloader)

    print(f"epoch: {epoch+1}, train_loss: {train_loss}")
    all_loss.append(train_loss.tolist())

if data_save:
    model.Save(
        pre_train_model.bert_model.encoder,
        data_save_dir + 'bert_encoder.model'
    )
    # Use this to save whole bert model
    model.Save(
        pre_train_model.bert_model,
        data_save_dir + 'bert_fine_tune.model'
    )
    model.Save(
        pre_train_model.gat,
        data_save_dir + 'gat.model'
    )

    # model.Save(
    #     pre_train_model,
    #     data_save_dir + 'a.model'
    # )


print(all_loss)
with open(data_save_dir+'loss.txt', 'w+')as f:
    for i in all_loss:
         f.write(str(i)+'\n')

# Save JSON one might be easier
JSON.encode(config, data_save_dir + 'config.txt')
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
# """
# testing
# """
