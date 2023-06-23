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


"""
Initializing
"""
suppressor = process.Quiet_Mode()
dtype = torch.float32
"""
This doesn't seem appropriate.
I will be surprised if it can be compiled.

device = torch.device([0] if torch.cuda.is_available() else 'cpu')
"""
device = [0]
worker.setup(device)
print('Done init')

"""
Data Preparing

I can not test any part of this due to lack of data.
"""
batch_size = 10
test_size = 0.3
data_name = 'GSE205117_NMFSM_bin20'
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/result_dmodel4/'

####load node
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/pp/pertur_erth/node/',df_type ='node')
#matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/node1/',df_type ='node')

###load edge
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/pp/pertur_erth/edge/',df_type ='edge')
gene_num = matrix2[list(matrix2.keys())[0]].columns
print(matrix2[list(matrix2.keys())[0]].shape)
tf = matrix2[list(matrix2.keys())[0]].index
gene = matrix2[list(matrix2.keys())[0]].columns
print(len(np.intersect1d(tf,gene[0:1103])))

### scale edge
for i in list(matrix2.keys()):
    edge = scale_network(matrix2[i])
    edge = torch.from_numpy(edge)
    edge = edge.to(torch.float32)
    matrix2[i] = edge

labels = torch.tensor([0]*len(matrix1))
### samples and labels
samples = []
for i in range(len(matrix1)):
    sample_name = list(matrix1.keys())[i]
    m1 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
    inds, attrs = lib.get_sparse_coo(edge)
    samples.append(
        pyg_d.Data(
            x = m1,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )

X_train,X_test,y_train,y_test = train_test_split(
    samples,
    labels,
    test_size=test_size,
    train_size = 1-test_size,
    random_state=0
)
pertubation_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = X_train),
    batch_size = batch_size,
    collate_fn = lib.collate_fn,
    shuffle=True
)

result_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = X_test),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=True
)
print('Done Data Prepare')


"""
Model Preparing
"""
config_path = '/home/peiweikeLab/jiangjunyao/conda/envs/fatez-test/lib/python3.10/site-packages/fatez/data/config/gat_bert_config.json'
# config_path = '../data/config/gat_bert_config.json'
config = JSON.decode(config_path)
trainer = pre_trainer.Set(config, dtype = dtype, device=device)
report_batch = False
size = trainer.input_sizes

trainer.worker.train(True)
best_loss = 99
loss_all = 0
report = list()

for x,y in pertubation_dataloader:

    # Prepare input data as always
    input = [ele.to(trainer.device) for ele in x]

    # Mute some debug outputs
    suppressor.on()
    node_rec, adj_rec = trainer.worker(input)
    suppressor.off()

    # Prepare pertubation result data using a seperate dataloader
    y = [result_dataloader.dataset.samples[ele].to(trainer.device) for ele in y]
    # Please be noted here that this script is only reconstructing TF parts
    # To reconstruct whole genome, we can certainly add an additionaly layer which takes adj_rec and node_rec to do the job.
    node_results = torch.split(
        torch.stack([ele.x for ele in y], 0),
        node_rec.shape[1],
        dim = 1
    )[0]
    adj_results = lib.get_dense_adjs(
        y, (size['n_reg'],size['n_node'],size['edge_attr'])
    )

    # Get total loss
    loss = trainer.criterion(node_rec, node_results)
    if adj_rec is not None:
        loss += trainer.criterion(adj_rec, adj_results)

    # Some backward stuffs here
    loss.backward()
    nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_norm)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

    # Accumulate
    best_loss = min(best_loss, loss.item())
    loss_all += loss.item()

    # Some logs
    if report_batch: report.append([loss.item()])


trainer.scheduler.step()
report.append([loss_all / len(pertubation_dataloader)])
report = pd.DataFrame(report)
report.columns = ['Loss', ]
print(report)

tuner = pre_trainer.Set(config, prev_model = trainer.model, dtype = dtype, device = device)

# Some new fake data
tuner_dataloader = faker.make_data_loader()

# And the tuning process is also based on input reconstruction as pretraining
suppressor.on()
report = tuner.train(tuner_dataloader, report_batch = False,)
suppressor.off()
print(report)

trainer.model.eval()

for x,_ in tuner_dataloader:
    # Prepare input data as always
    input = [ele.to(trainer.device) for ele in x]
    # Mute some debug outputs
    suppressor.on()
    node_rec, adj_rec = trainer.model(input)
    suppressor.off()
    print(node_rec.shape, adj_rec.shape)
