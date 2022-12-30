import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fatez.model as model
import fatez.model.gat as gat
import fatez.model.sparse_gat as sgat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.process.preprocessor as pre
import pandas as pd
### preprocess parameters
pseudo_cell_num_per_cell_type = 2
correlation_thr_to_get_gene_related_peak = 0.6
rowmean_thr_to_get_variable_gene = 0.1
### preprocess
peak_path = ('../data/mouse/filtered_feature_bc_matrix/')
rna_path = ('../data/mouse/filtered_feature_bc_matrix/')
gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
tf_db_path = 'E:\\public/TF_target_tss_1500.txt.gz'
network = pre.Preprocessor(rna_path, peak_path, gff_path, tf_db_path, data_type='paired')
network.load_data()
### qc
network.rna_qc(rna_min_genes=5, rna_min_cells=250, rna_max_cells=2500)
network.atac_qc(atac_min_features=5, )
### select cell type
cell_type = pd.read_csv(
    'E:\\public\\public data\\10X\\e18_mouse_brain_fresh_5k\\e18_mouse_brain_fresh_5k_analysis\\analysis\\clustering\\gex\\graphclust/clusters.csv')
cell_type.index = cell_type['Barcode']
cell_type = cell_type['Cluster']
cell_type = cell_type[cell_type.isin([1,4])]
network.add_cell_label(cell_type)

network.make_pseudo_networks(data_type='paired',network_number=pseudo_cell_num_per_cell_type)
network.cal_peak_gene_cor(exp_thr = rowmean_thr_to_get_variable_gene,
                          cor_thr=correlation_thr_to_get_gene_related_peak)
matrix1 = network.output_pseudo_samples() ### exp count mt
matrix2 = network.generate_grp() ### correlation mt

k = 8000
top_k = 1000
n = 2
n_class = 2
nhead = None
d_model = 2
en_dim = 8

samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m2 = matrix2[list(matrix2.keys())[i]]
    m2 = torch.from_numpy(m2.to_numpy())
    samples.append(m1)
    samples.append(m2)




adj_mat = torch.randn(top_k, k)
sample = [torch.randn(k, d_model), adj_mat]
samples = [sample]*n
labels = torch.tensor([1]*n)
print('# Fake feat:', k)
print('# Sample:', len(samples))

# print('here')
print('Test plain GAT')

model_gat = gat.GAT(d_model = d_model, en_dim = en_dim, nhead = nhead,)

# Test GAT
out_gat = model_gat(samples)
out_gat = model_gat.activation(out_gat)
out_gat = model_gat.decision(out_gat)



# Need to make sure d_model is divisible by nhead
bert_encoder = bert.Encoder(
    d_model = en_dim,
    n_layer = 6,
    nhead = 8,
    dim_feedforward = 2,
)

test_model = fine_tuner.Model(
    gat = model_gat,
    bin_pro = model.Binning_Process(n_bin = 100),
    bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = n_class)
)
output = test_model(samples)


loss = nn.CrossEntropyLoss()(
    output, labels
)
loss.backward()
print('GAT CEL:', loss)

# loss = F.nll_loss(out_gat, labels)
# print(loss)



# print('Test sparse GAT')
# model_sgat = sgat.Spare_GAT(d_model = d_model, en_dim = en_dim, nhead = nhead,)
# out_sgat = model_sgat(samples)
#
# # Activation and Decision
# out_sgat = model_gat.activation(out_sgat)
# out_sgat = model_gat.decision(out_sgat)
#
# loss = nn.CrossEntropyLoss()(
#     out_sgat, labels
# )
# loss.backward()
# print('SGAT CEL:', loss)

# loss = F.nll_loss(out_sgat, labels)
# print(loss)
