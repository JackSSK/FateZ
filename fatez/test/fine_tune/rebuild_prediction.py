import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import fatez.lib as lib
import fatez.test as test
import fatez.model as model
import fatez.tool.JSON as JSON
import fatez.process as process
import fatez.process.worker as worker
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
from pkg_resources import resource_filename
import torch_geometric.data as pyg_d
import fatez.tool.PreprocessIO as PreprocessIO
import fatez.process.early_stopper as es
from fatez.process.scale_network import scale_network
from sklearn.model_selection import train_test_split
import numpy as np
def tensor_cor_atac(tensor1,tensor2):
    all_cor = []
    for i in range(tensor1.shape[0]):
        tensor1_use = tensor1[i]
        tensor2_use = tensor2[i]
        column1 = tensor1_use[:, 1].detach().numpy()
        column2 = tensor2_use[:, 1].detach().numpy()
        correlation1 = np.corrcoef(column1, column2)[0, 1]
        all_cor.append(correlation1)
    return(np.array(all_cor).mean())
def tensor_cor_rna(tensor1,tensor2):
    all_cor = []
    for i in range(tensor1.shape[0]):
        tensor1_use = tensor1[i]
        tensor2_use = tensor2[i]
        column1 = tensor1_use[:, 0].detach().numpy()
        column2 = tensor2_use[:, 0].detach().numpy()
        correlation1 = np.corrcoef(column1, column2)[0, 1]
        all_cor.append(correlation1)
    return(np.array(all_cor).mean())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=10
model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/rebuild_fine_tune.model'
###data
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_endo/node/',df_type ='node',order_cell=False)
edge = pd.read_csv('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/GSE205117#Endothelium.csv',header=0,index_col=0)
edge = scale_network(edge)
edge = torch.from_numpy(edge)
edge = edge.to(torch.float32)
samples = []
samples2 = []
for i in range(len(matrix1)):
    sample_name = list(matrix1.keys())[i]
    m1 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
    m2 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
    m2[:,1] = 0
    inds, attrs = lib.get_sparse_coo(edge)
    samples.append(
        pyg_d.Data(
            x = m2,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )
    samples2.append(
        pyg_d.Data(
            x = m1,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )
pertubation_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = samples),
    batch_size = batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)

result_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = samples2),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)
config_path = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/tf_config/config1.json'
config = JSON.decode(config_path)
batch_size=len(matrix1)
trainer = pre_trainer.Set(config, dtype = torch.float32, device=device,
                          prev_model=model.Load(model_dir))
trainer.setup()
report_batch = True
for x,y in pertubation_dataloader:
    input = [ele.to(trainer.device) for ele in x]
    node_rec, adj_rec = trainer.worker(input)
    y = [result_dataloader.dataset.samples[ele].to(trainer.device) for ele in y]
    node_results = torch.split(
        torch.stack([ele.x for ele in y], 0),
        node_rec.shape[1],
        dim=1
    )[0]
    node_results = nn.LogSoftmax(dim=-2)(node_results)
    cor_atac = tensor_cor_atac(node_rec.cpu(), node_results.cpu())
    print(cor_atac)
    torch.save(node_rec,
               '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/rec.pt')
    torch.save(node_results,
               '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/result.pt')