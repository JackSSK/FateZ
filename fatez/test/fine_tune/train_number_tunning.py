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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_dataset_ct(label,train_number):
    train_all = []
    test_all = []
    for i in set(label['label']):
        label_use = label.loc[label['label']==i]
        train_sample = label_use['sample'][0:train_number].to_list()
        test_sample = label_use['sample'][train_number:len(label)].to_list()
        train_all.extend(train_sample)
        test_all.extend(test_sample)
    return train_all,test_all
"""
Data Preparing

I can not test any part of this due to lack of data.
"""
batch_size = 10
epoch = 10
data_name = 'GSE205117_NMFSM_bin20'
pretrain_model = 'No'
model_name = sys.argv[1]
train_number = int(sys.argv[2])
data_save_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/'

model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/model_tf/epoch1/'+model_name
print('pretrain model',model_dir)

####load node
matrix1 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE231674_kft1_anno/node/',df_type ='node',order_cell=False)


###load edge
matrix2 = PreprocessIO.input_csv_dict_df('/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/',df_type ='edge',order_cell=False)

###load label

### scale edge
for i in list(matrix2.keys()):
    edge = scale_network(matrix2[i])
    edge = torch.from_numpy(edge)
    edge = edge.to(torch.float32)
    matrix2[i] = edge


edge_label = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/pp/label2/GSE231674_kft1_anno.txt')
edge_label.index = edge_label['sample']
train_cell,test_cell = split_dataset_ct(edge_label,train_number)
### samples and labels

samples1 = []
samples2 = []
samples3 = []
samples4 = []

for i in range(len(matrix1)):
    sample_name = list(matrix1.keys())[i]
    m1 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
    m2 = torch.from_numpy(matrix1[sample_name].to_numpy()).to(torch.float32)
    m2[:,1] = 0
    dict_key = sample_name.split('#')[0]
    label = edge_label['label'][str(sample_name)]
    edge_name = edge_label.loc[sample_name, 'label']
    key_use = dict_key + '#' + str(edge_name)
    inds, attrs = lib.get_sparse_coo(matrix2[key_use])
    if sample_name in train_cell:
        samples1.append(
            pyg_d.Data(
                x = m2,
                edge_index = inds,
                edge_attr = attrs,
                y = 0,
                shape = matrix2[key_use].shape,
            )
        )
        samples2.append(
            pyg_d.Data(
                x = m1,
                edge_index = inds,
                edge_attr = attrs,
                y = 0,
                shape = matrix2[key_use].shape,
            )
        )
    else:
        samples3.append(
            pyg_d.Data(
                x = m2,
                edge_index = inds,
                edge_attr = attrs,
                y = 0,
                shape = matrix2[key_use].shape,
            )
        )
        samples4.append(
            pyg_d.Data(
                x = m1,
                edge_index = inds,
                edge_attr = attrs,
                y = 0,
                shape = matrix2[key_use].shape,
            )
        )

print(len(samples1))
print(len(samples3))


pertubation_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = samples1),
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
    batch_size=len(samples3),
    collate_fn = lib.collate_fn,
    shuffle=False
)
predict_true_dataloader = DataLoader(
    lib.FateZ_Dataset(samples = samples4),
    batch_size=len(samples4),
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

    trainer = pre_trainer.Set(
        config,
        dtype = dtype,
        device=device,
        prev_model=model.Load(model_dir)
    )
    trainer = pre_trainer.Set(
        config,
        node_recon_dim = 1,
        dtype = dtype,
        device=device,
        prev_model=trainer
    )
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

            node_results = torch.stack([ele.x for ele in input], 0)

            """
            Need to select training feature here by partioning node_results
            """

            # The input is not LogSoftmax-ed?
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
        out1.to_csv(data_save_dir+model_name+'train_number_'+str(train_number)+'_loss'+config_name+'.csv',mode='a+')
        report_cor.to_csv(
            data_save_dir+model_name +'train_number_'+str(train_number)+ '_cor_' + config_name + '.csv',
            mode='a+')
        torch.save(node_rec,
                   data_save_dir+model_name+'_node_rec.pt')
        torch.save(node_results,
                   data_save_dir+model_name+'_node_result.pt')

    trainer.model.eval()
    ###
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
    report_cor = pd.DataFrame(
        {'predict_cor_rna': [cor_rna], 'predict_cor_atac': predict_all_cor_atac_hap})
    report_cor.to_csv(
        data_save_dir + model_name + 'train_number_' + str(
            train_number) + '_cor_' + config_name + '.csv',
        mode='a+')

    node_rec = node_rec[:,:,1]
    ode_results = ode_results[:, :, 1]
    df1 = pd.DataFrame(node_rec.detach().numpy())
    df2 = pd.DataFrame(node_results.detach().numpy())
    df1.columns = test_cell
    df2.columns = test_cell
    df1.index = list(matrix2.index)
    df2.index = list(matrix2.index)
    df1.to_csv(data_save_dir + model_name + 'train_number_' + str(
            train_number) + '_predict_mt.csv')
    df2.to_csv(data_save_dir + model_name + 'train_number_' + str(
        train_number) + '_true_mt.csv')

    model.Save(
        trainer,
        data_save_dir + model_name+'train_number_'+str(train_number)+ '_rebuild_fine_tune.model'
    )
