import numpy as np
import torch
import fatez.model as model
import fatez.model.gat as gat
import torch.nn as nn
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.process.preprocessor as pre
import pandas as pd
from transformers import AdamW
from torch.utils.data import DataLoader
import fatez.lib as lib
from fatez.tool import PreprocessIO
import fatez.model.mlp as mlp
from sklearn.model_selection import train_test_split
import fatez.tool.JSON as JSON
from fatez.tool import EarlyStopping
from fatez.tool import model_testing
import os
"""
preprocess
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## preprocess parameters
pseudo_cell_num_per_cell_type = 5000
correlation_thr_to_get_gene_related_peak = 0.4
rowmean_thr_to_get_variable_gene = 0.1
cluster_use =[1,4]
# peak_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/atac_AE_Pre10x/')
# rna_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/rna_AE_Pre10x/')
# gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
# tf_db_path = 'E:\\public/TF_target_tss1500.txt.gz'
# network = pre.Preprocessor(rna_path, peak_path, gff_path, tf_db_path, data_type='unpaired')
# network.load_data(matrix_format='10x_unpaired')
# ### qc
# network.rna_qc(rna_min_genes=1, rna_min_cells=1, rna_max_cells=5000000)
# network.atac_qc(atac_min_cells=10, )
# print(network.atac_mt)
# ### select cell type
# atac_cell_type = pd.read_table(
#  'D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/atac_cell_type_AE_Pre.txt',
#   header=None)
# atac_cell_type.index = atac_cell_type[0]
# atac_cell_type = atac_cell_type[1]
# rna_cell_type = pd.read_table(
#  'D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/rna_cell_type_AE_Pre.txt',
#   header=None)
# rna_cell_type.index = rna_cell_type[0]
# rna_cell_type = rna_cell_type[1]
# network.add_cell_label(rna_cell_type,modality='rna')
# network.add_cell_label(atac_cell_type,modality='atac')
# network.annotate_peaks()
# network.make_pseudo_networks(data_type='unpaired',
#                              network_number=pseudo_cell_num_per_cell_type,
#                              network_cell_size = 10)
# network.cal_peak_gene_cor(exp_thr = rowmean_thr_to_get_variable_gene,
#                           cor_thr=correlation_thr_to_get_gene_related_peak)
#
# matrix1 = network.output_pseudo_samples() ### exp count mt
# matrix2 = network.generate_grp() ### correlation mt
# network.extract_motif_score(matrix2)
# matrix2 = np.multiply(network.motif_enrich_score,matrix2)


matrix1 = PreprocessIO.input_csv_dict_df(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data_10000/node/')
matrix2 = pd.read_csv(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data_10000/edge_matrix.csv'
    ,index_col=0)

# train_rate = 0.7
# train_idx, val_test_idx, _, y_validate_test = train_test_split(index, labels, stratify=labels, train_size=train_rate,test_size=1-train_rate,
#                                                  random_state=2, shuffle=True)
# val_idx, test_idx, _, _ = train_test_split(val_test_idx,y_validate_test, train_size=val_rate/(1-train_rate), test_size=1-val_rate/(1-train_rate),
#                                                      random_state=2, shuffle=True)


### samples and labels
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m2 = torch.from_numpy(matrix2.to_numpy())
    m1 = m1.to(torch.float32)
    m2 = m2.to(torch.float32)
    # m1 = m1.to(device)
    # m2 = m2.to(device)
    samples.append([m1, m2])
labels = torch.from_numpy(np.repeat(range(len(cluster_use))
                                    ,len(matrix1)/len(cluster_use)))
labels = labels.long()
labels = labels.to(device)
###
"""
hyperparameters
"""
batch_size = 40
num_epoch = 100
n_hidden = 2
nhead = 0
lr = 1e-3
test_size = 0.3
early_stop_tolerance = 15
data_save = True
data_save_dir = 'D:\\Westlake\\pwk lab\\fatez\\gat_gradient/nhead0_nhidden2_lr-3/'
outgat_dir = data_save_dir+'out_gat/'
os.makedirs(outgat_dir )
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
model_gat = gat.GAT(
    d_model = 2,
    en_dim = 8,
    nhead = nhead,
    device = device,
    n_hidden = n_hidden
)
decison = mlp.Classifier(
    d_model = 8,
    n_hidden = 4,
    n_class = 2,
    device = device,
)
bert_encoder = bert.Encoder(
    d_model = 8,
    n_layer = 6,
    nhead = 8,
    dim_feedforward = 2,
    device = device,
)
test_model = fine_tuner.Model(
    gat = model_gat,
    bin_pro = model.Binning_Process(n_bin = 100),
    bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = 2),
)
### adam and CosineAnnealingWarmRestarts
optimizer = torch.optim.Adam(test_model.parameters(),
                             lr=lr,
                             weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=2,
                                                                 T_mult=2,
                                                                 eta_min=lr/50)

early_stopping = EarlyStopping.EarlyStopping(tolerance=early_stop_tolerance, min_delta=10)
model_gat.to(device)
bert_encoder.to(device)
test_model.to(device)

"""
traning
"""
all_loss = list()
for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    batch_num = 1
    test_model.train()
    train_loss = 0
    train_acc = 0
    out_gat_data = list()
    for x,y in train_dataloader:
        optimizer.zero_grad()
        out_gat = model_gat(x[0], x[1])
        # torch.save(out_gat,
        #            'D:\\Westlake\\pwk lab\\fatez\\out_gat/epoch'+str(epoch)+'_batch'+str(batch_num)+'.pt')
        output = test_model(x[0].to(device), x[1].to(device))
        for ele in out_gat.detach().tolist(): out_gat_data.append(ele)
        loss = nn.CrossEntropyLoss()(
            output, y
        )
        loss.backward()
        optimizer.step()
        acc = (output.argmax(1)==y).type(torch.float).sum()/batch_size
        print(f"batch: {batch_num} loss: {loss} accuracy:{acc}")
        batch_num += 1
        train_loss += loss
        train_acc += acc
    print(
     f"epoch: {epoch+1}, train_loss: {train_loss/175}, train accuracy: {train_acc/175}")
    all_loss.append(train_loss)
    scheduler.step()
    test_loss,test_acc = model_testing.testing(test_dataloader,
                                               test_model, nn.CrossEntropyLoss()
                                               , device=device)
    print(
        f"epoch: {epoch+1}, test_loss: {test_loss}, test accuracy: {test_acc}")
    # if data_save:
    #     JSON.encode(
    #         out_gat_data,
    #         outgat_dir+str(epoch) + '.js'
    #     )
    early_stopping(train_loss, test_loss)
    if early_stopping.early_stop:
        print("We are at epoch:", i)
        break
if data_save:
    model.Save(test_model.bert_model.encoder,
               data_save_dir+'bert_encoder.model')
    # model.Save(test_model,
    #            data_save_dir+'fine_tune.model')
    model.Save(model_gat,
               data_save_dir+'gat.model')
print(all_loss)
test = bert.Fine_Tune_Model(test_model.bert_model.encoder, n_class = 2)
model.Save(test, data_save_dir+'bert_fine_tune.model')
JSON.encode(
    out_gat_data,
    outgat_dir + str(epoch) + '.js'
)
"""
testing
"""
