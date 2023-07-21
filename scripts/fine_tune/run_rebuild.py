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
dtype = torch.float32
"""
This doesn't seem appropriate.
I will be surprised if it can be compiled.

device = torch.device([0] if torch.cuda.is_available() else 'cpu')
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Done init')

"""
Data Preparing

I can not test any part of this due to lack of data.
"""
batch_size = 10
epoch = 2
data_name = 'GSE205117_NMFSM_bin20'
pretrain_model = 'No'
model_name = sys.argv[1]
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/'

model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/model_tf/epoch1/'+model_name
print('pretrain model',model_dir)

####load node
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_endo/node/',df_type ='node',order_cell=False)
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_hep/node/',df_type ='node',order_cell=False)
matrix3 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_ery/node/',df_type ='node',order_cell=False)
matrix4 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_bp/node/',df_type ='node',order_cell=False)

###load edge
edge = pd.read_csv('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/GSE205117#Endothelium.csv',header=0,index_col=0)
edge2 = pd.read_csv('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/GSE205117#Haematoendothelial_progenitors.csv',header=0,index_col=0)
edge3 = pd.read_csv('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/GSE205117#Erythroid1.csv',header=0,index_col=0)
edge4 = pd.read_csv('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/GSE205117#Blood_progenitors_2.csv',header=0,index_col=0)
### scale edge
edge = scale_network(edge)
edge = torch.from_numpy(edge)
edge = edge.to(torch.float32)
edge2 = scale_network(edge2)
edge2 = torch.from_numpy(edge2)
edge2 = edge2.to(torch.float32)
edge3 = scale_network(edge3)
edge3 = torch.from_numpy(edge3)
edge3 = edge3.to(torch.float32)
edge4 = scale_network(edge4)
edge4 = torch.from_numpy(edge4)
edge4 = edge4.to(torch.float32)

labels = torch.tensor([0]*len(matrix1))
### samples and labels
samples = []
samples2 = []
samples3 = []
samples4 = []
samples5 = []
samples6 = []
samples7 = []
samples8 = []
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
for i in range(len(matrix2)):
    sample_name = list(matrix2.keys())[i]
    m1 = torch.from_numpy(matrix2[sample_name].to_numpy()).to(torch.float32)
    m2 = torch.from_numpy(matrix2[sample_name].to_numpy()).to(torch.float32)
    m2[:,1] = 0
    inds, attrs = lib.get_sparse_coo(edge2)
    samples3.append(
        pyg_d.Data(
            x = m2,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )
    samples4.append(
        pyg_d.Data(
            x = m1,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )
for i in range(len(matrix3)):
    sample_name = list(matrix3.keys())[i]
    m1 = torch.from_numpy(matrix3[sample_name].to_numpy()).to(torch.float32)
    m2 = torch.from_numpy(matrix3[sample_name].to_numpy()).to(torch.float32)
    m2[:,1] = 0
    inds, attrs = lib.get_sparse_coo(edge3)
    samples5.append(
        pyg_d.Data(
            x = m2,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )
    samples6.append(
        pyg_d.Data(
            x = m1,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )
for i in range(len(matrix4)):
    sample_name = list(matrix4.keys())[i]
    m1 = torch.from_numpy(matrix4[sample_name].to_numpy()).to(torch.float32)
    m2 = torch.from_numpy(matrix4[sample_name].to_numpy()).to(torch.float32)
    m2[:,1] = 0
    inds, attrs = lib.get_sparse_coo(edge4)
    samples7.append(
        pyg_d.Data(
            x = m2,
            edge_index = inds,
            edge_attr = attrs,
            y = 0,
            shape = edge.shape,
        )
    )
    samples8.append(
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
predict_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = samples3),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)
predict_true_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = samples4),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)
### ery
predict_dataloader2 = DataLoader(
    lib.FateZ_Dataset(samples = samples5),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)
predict_true_dataloader2 = DataLoader(
    lib.FateZ_Dataset(samples = samples6),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)
### bp
predict_dataloader3 = DataLoader(
    lib.FateZ_Dataset(samples = samples7),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)
predict_true_dataloader3 = DataLoader(
    lib.FateZ_Dataset(samples = samples8),
    batch_size=batch_size,
    collate_fn = lib.collate_fn,
    shuffle=False
)

print('Done Data Prepare')
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
"""
Model Preparing
"""
#config_list = os.listdir('/storage/peiweikeLab/jiangjunyao/fatez/tune_bert/new_version_config/generate_config')
config_list = ['config1.json']
for config_name in config_list:

    config_path = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/tf_config/'+config_name
    config = JSON.decode(config_path)

    trainer = pre_trainer.Set(config, dtype = dtype, device=device,prev_model=model.Load(model_dir))
    trainer.setup()
    report_batch = True
    size = trainer.input_sizes
    print('Done model set')


    for i in range(epoch):
        trainer.worker.train(True)
        best_loss = 99
        loss_all = 0
        report = list()
        all_cor_rna = []
        all_cor_atac = []
        print('epoch-------'+str(i+1))
        for x,y in pertubation_dataloader:

            # Prepare input data as always
            input = [ele.to(trainer.device) for ele in x]

            node_rec, adj_rec = trainer.worker(input)


            # Prepare pertubation result data using a seperate dataloader
            y = [result_dataloader.dataset.samples[ele].to(trainer.device) for ele in y]
            # Please be noted here that this script is only reconstructing TF parts
            # To reconstruct whole genome, we can certainly add an additionaly layer which takes adj_rec and node_rec to do the job.
            node_results = torch.split(
                torch.stack([ele.x for ele in y], 0),
                node_rec.shape[1],
                dim = 1
            )[0]
            node_results = nn.LogSoftmax(dim=-2)(node_results)
            adj_results = lib.get_dense_adjs(
                y, (size['n_reg'],size['n_node'],size['edge_attr'])
            )
            cor_atac = tensor_cor_atac(node_rec.cpu(),node_results.cpu())
            cor_rna = tensor_cor_rna(node_rec.cpu(), node_results.cpu())
            all_cor_rna.append(cor_rna)
            all_cor_atac.append(cor_atac)
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
        report = pd.DataFrame({'loss':report})
        report_cor = pd.DataFrame({'cor_rna':all_cor_rna,'cor_atac':all_cor_atac})
        print(report)
        out1 = pd.DataFrame(report)
        out1.to_csv(data_save_dir+model_name+'loss'+config_name+'.csv',mode='a+')
        report_cor.to_csv(
            data_save_dir+model_name +'cor_' + config_name + '.csv',
            mode='a+')
        torch.save(node_rec,
                   data_save_dir+model_name+'_node_rec.pt')
        torch.save(node_results,
                   data_save_dir+model_name+'_node_result.pt')


        ### hap
        predict_all_cor_atac_hap = []
        for x,y in predict_dataloader:
            # Prepare input data as always
            input = [ele.to(trainer.device) for ele in x]
            # Mute some debug outputs
            node_rec, adj_rec = trainer.model(input)
            y = [predict_true_dataloader.dataset.samples[ele].to(trainer.device) for ele
                 in y]
            node_results = torch.split(
                torch.stack([ele.x for ele in y], 0),
                node_rec.shape[1],
                dim=1
            )[0]
            node_results = nn.LogSoftmax(dim=-2)(node_results)
            cor_atac = tensor_cor_atac(node_rec.cpu(), node_results.cpu())
            cor_rna = tensor_cor_rna(node_rec.cpu(), node_results.cpu())
            predict_all_cor_atac_hap.append(cor_atac)

        ### ery
        predict_all_cor_atac_ery = []
        for x,y in predict_dataloader2:
            # Prepare input data as always
            input = [ele.to(trainer.device) for ele in x]
            # Mute some debug outputs
            node_rec, adj_rec = trainer.model(input)
            y = [predict_true_dataloader2.dataset.samples[ele].to(trainer.device) for ele
                 in y]
            node_results = torch.split(
                torch.stack([ele.x for ele in y], 0),
                node_rec.shape[1],
                dim=1
            )[0]
            node_results = nn.LogSoftmax(dim=-2)(node_results)

            cor_atac = tensor_cor_atac(node_rec.cpu(), node_results.cpu())
            cor_rna = tensor_cor_rna(node_rec.cpu(), node_results.cpu())
            predict_all_cor_atac_ery.append(cor_atac)
            ### predict

        ### bp
        predict_all_cor_atac_bp = []
        for x, y in predict_dataloader3:
            # Prepare input data as always
            input = [ele.to(trainer.device) for ele in x]
            # Mute some debug outputs
            node_rec, adj_rec = trainer.model(input)
            y = [predict_true_dataloader3.dataset.samples[ele].to(trainer.device)
                 for ele
                 in y]
            node_results = torch.split(
                torch.stack([ele.x for ele in y], 0),
                node_rec.shape[1],
                dim=1
            )[0]
            node_results = nn.LogSoftmax(dim=-2)(node_results)

            cor_atac = tensor_cor_atac(node_rec.cpu(), node_results.cpu())
            cor_rna = tensor_cor_rna(node_rec.cpu(), node_results.cpu())
            predict_all_cor_atac_bp.append(cor_atac)
        hap = np.array(predict_all_cor_atac_hap).mean()
        ery = np.array(predict_all_cor_atac_ery).mean()
        bp = np.array(predict_all_cor_atac_bp).mean()
        predict_out = pd.DataFrame({'hap':[hap],'ery':[ery],'bp':[bp]})
        predict_out.to_csv(data_save_dir+model_name +'predict.csv',mode='a+',
                           header=False,index=False)

    trainer.model.eval()
    model.Save(
        trainer,
        data_save_dir + 'rebuild_fine_tune.model'
    )
