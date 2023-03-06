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
import fatez.model.transformer as transformer
import fatez.model.gat as gat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.process.position_embedder as pe
from sklearn.model_selection import train_test_split


"""
preprocess
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## preprocess parameters
pseudo_cell_num_per_cell_type = 250
cluster_use =[1,4]


matrix1 = PreprocessIO.input_csv_dict_df(
    'D:\\Westlake\\pwk lab\\fatez\\para_test/node/')
matrix2 = pd.read_csv(
    'D:\\Westlake\\pwk lab\\fatez\\para_test/edge_matrix.csv'
    ,index_col=0)
m2 = torch.from_numpy(matrix2.to_numpy())
matrix2 = matrix2.replace(np.nan,0)
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
labels = labels.cuda()
print(labels.device)
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
num_epoch = 100
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
    "rep_embedder": {
        "type": "Skip",             # Not using any positional embedding method
        # "type": "ABS",              # Absolute positional embedding
        "params": {
            "n_embed": 100,         # Number of TFs
            "n_dim": n_features,
        }
    },
    "graph_embedder": {
        "type": "Skip",             # Not using any positional embedding method
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

##############################
data_save_dir = 'D:\\Westlake\\pwk lab\\fatez\\tune_para\\test1/'
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
factory_kwargs = {'device': device, 'dtype': torch.float32,}
fine_tuner_model = fine_tuner.Set(config, factory_kwargs)
"""
traning
"""
all_loss = list()
for epoch in range(num_epoch):
    print(f"Epoch {epoch+1}\n-------------------------------")

    train_loss, acc = fine_tuner_model.train(train_dataloader, print_log=True)
    print(f"epoch: {epoch+1}, train_loss: {train_loss}, ACC: {acc}")

    test_loss, acc = fine_tuner_model.test(train_dataloader)
    print(f"epoch: {epoch+1}, test_loss: {test_loss}, ACC: {acc}")

    all_loss.append(train_loss.tolist())


if data_save:
    model.Save(
        fine_tuner_model.model,
        data_save_dir + 'fine_tune.model'
    )
print(all_loss)
