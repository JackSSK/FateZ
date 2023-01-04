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
"""
preprocess
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
## preprocess parameters
pseudo_cell_num_per_cell_type = 2
correlation_thr_to_get_gene_related_peak = 0.6
rowmean_thr_to_get_variable_gene = 0.1
cluster_use =[1,4]
peak_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/atac_10x/')
rna_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/rna_10x/')
gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
tf_db_path = 'E:\\public/TF_target_tss_1500.txt.gz'
network = pre.Preprocessor(rna_path, peak_path, gff_path, tf_db_path, data_type='unpaired')
network.load_data(matrix_format='10x_unpaired')
### qc
network.rna_qc(rna_min_genes=1, rna_min_cells=1, rna_max_cells=5000000)
network.atac_qc(atac_min_cells=1  , )
### select cell type
atac_cell_type = pd.read_table(
 'D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/atac_cell_type.txt',
  header=None)
atac_cell_type.index = atac_cell_type[0]
atac_cell_type = atac_cell_type[1]
rna_cell_type = pd.read_table(
 'D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/rna_cell_type.txt',
  header=None)
rna_cell_type.index = rna_cell_type[0]
rna_cell_type = rna_cell_type[1]
network.add_cell_label(rna_cell_type,modality='rna')
network.add_cell_label(atac_cell_type,modality='atac')
network.annotate_peaks()
network.make_pseudo_networks(data_type='unpaired',network_number=pseudo_cell_num_per_cell_type)
network.cal_peak_gene_cor(exp_thr = rowmean_thr_to_get_variable_gene,
                          cor_thr=correlation_thr_to_get_gene_related_peak)
matrix1 = network.output_pseudo_samples() ### exp count mt
matrix2 = network.generate_grp() ### correlation mt
### samples and labels
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m2 = matrix2[list(matrix2.keys())[i]]
    #motif_enrich_mt = network.extract_motif_score(m2)
    m2 = torch.from_numpy(m2.to_numpy())
    #m2 = np.multiply(m2,motif_enrich_mt)
    m1 = m1.to(torch.float64)
    m2 = m2.to(torch.float64)
    m1.to(device)
    m2.to(device)
    samples.append([m1, m2])
labels = torch.from_numpy(np.repeat(range(len(cluster_use))
                                    ,pseudo_cell_num_per_cell_type))
labels = labels.long()
"""
model define
"""
model_gat = gat.GAT(d_model = 2, en_dim = 8, nhead = None, device = device)
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
    bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = 2)
)
optimizer = AdamW(test_model.parameters(), lr=0.01)
model_gat.to(device)
bert_encoder.to(device)
test_model.to(device)
"""
traning
"""
batch_size = 10
num_epoch = 30
train_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=samples, labels=labels),
    batch_size=batch_size,
    shuffle=True
)
for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    batch_num = 1
    for x,y in train_dataloader:
        # sample_idx_list =list(x.numpy())
        # sample_use = []
        # for idx in sample_idx_list:
        #     sample_use.append(samples[idx])
        out_gat = model_gat(x)
        out_gat = model_gat.activation(out_gat)
        out_gat = model_gat.decision(out_gat)
        output = test_model(x)
        loss = nn.CrossEntropyLoss()(
            output, y
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (output.argmax(1)==y).type(torch.float).sum()/batch_size
        print(f"batch: {batch_num} loss: {loss} accuracy:{acc}")
        batch_num += 1
model.Save(test_model.bert_model.encoder, '../data/ignore/bert_encoder.model')

for x,y in train_dataloader:
    print(x)
    print(y)
