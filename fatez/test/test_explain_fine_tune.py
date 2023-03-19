#!/usr/bin/env python3
"""
Note:

I suppose you want to explain BERT Fine Tune Model, not the decision model

"""
import fatez.lib as lib
from torch.utils.data import DataLoader
from fatez.tool import identify_regulons
import torch
import fatez.process.fine_tuner as fine_tuner
import fatez.model as model
from fatez.tool import PreprocessIO
import pandas as pd
import numpy as np
import torch.nn as nn
import fatez.model.position_embedder as pe
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp_param = {
    'd_model': 8,
    'n_hidden': 4,
    'n_class': 2,
    'device': device,
    'dtype': torch.float32,
}

batch_size = 2
acc_all = 0
explain_use = []
explain_num = 1
output_result = False
model_gat = model.Load('D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\model\\nhead0_endim8_nhidden2_lr-3_epoch50_batch_size_20\\gat.model')
fine_tuning = fine_tuner.Model(
    gat = model.Load('D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\model\\nhead0_endim8_nhidden2_lr-3_epoch50_batch_size_20\\gat.model'),
    rep_embedder = pe.Skip(),
    bert_model = model.Load('D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\model\\nhead0_endim8_nhidden2_lr-3_epoch50_batch_size_20\\bert_fine_tune.model')
)
fine_tuning.to(device)
matrix1 = PreprocessIO.input_csv_dict_df(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500/node/'
)
matrix2 = pd.read_csv(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500/edge_matrix.csv',
    index_col = 0
)
m2 = torch.from_numpy(matrix2.to_numpy())
m2 = m2.to(torch.float32)
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())

    m1 = m1.to(torch.float32)

    samples.append([m1, m2])
labels = torch.from_numpy(np.repeat(range(2)
                                    ,len(matrix1)/2))
labels = labels.long()
labels = labels.to(device)

train_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=samples, labels=labels),
    batch_size=batch_size,
    shuffle=True
)

fine_tuning.eval()
with torch.no_grad():
    for x,y in train_dataloader:
        pred = fine_tuning(x[0].to(device), x[1].to(device))
        test_loss = nn.CrossEntropyLoss()(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        if correct == batch_size:
            explain_use.append(x)
        acc_all+=correct
        if len(explain_use) == explain_num:
            break
print(acc_all)
### sum rank
regulons = identify_regulons.Regulon(explain_use,matrix2)
regulons.explain_model(fine_tuning,batch_size,filter_tf=False)
regulon_count = regulons.sum_regulon_count()
# regulon_count.index = list(matrix2.columns)
# regulon_count = regulon_count[list(matrix2.index)]
regulon_count = regulon_count.sort_values()

### count top
regulon_top = regulons.get_top_regulon_count(20)
regulon_top = pd.Series(regulon_top)
regulon_top = regulon_top.sort_values(ascending=False)

gene_name = pd.read_table('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117\\rna_AE_Pre10x\\features.tsv.gz',header=None)
gene_name.index = gene_name[0]
sum_top = regulons.tf_names[regulon_count.index]
count_top = regulons.tf_names[regulon_top.index]
count = regulon_top.tolist()
rank = regulon_count.tolist()
print('sum rank top 20:')
sum_top_tf = []
tf_rank = []
for i in range(len(sum_top)):
    if sum_top[i] in matrix2.index:
        sum_top_tf.append(sum_top[i])
        count_idx = rank[i]
        tf_rank.append(count_idx)
print(tf_rank)
top_symbol = []
for i in sum_top_tf:
    symbol = gene_name[gene_name[0].isin([i])][1].to_numpy().tolist()[0]
    top_symbol.append(symbol)

print(top_symbol)
if output_result:
    with open('D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\model\\nhead0_nhidden2_lr-3_epoch50_batch_size_20\\sum_rank.txt','w+')as f:
        for i in range(len(top_symbol)):
            a1 = str(top_symbol[i])+'\n'
            f.write(a1)




print('count top top 20:')
count_top_tf = []
tf_count = []
for i in range(len(count_top)):
    if count_top[i] in matrix2.index:
        count_top_tf.append(count_top[i])
        count_idx = count[i]
        tf_count.append(count_idx)
print(tf_count)
top_symbol = []
for i in count_top_tf:
    symbol = gene_name[gene_name[0].isin([i])][1].to_numpy().tolist()[0]
    top_symbol.append(symbol)
print(top_symbol)
if output_result:
    with open('D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\model\\nhead0_nhidden2_lr-3_epoch50_batch_size_20\\count_top_10.txt','w+')as f:
        for i in range(len(top_symbol)):
            a1 = str(top_symbol[i])+'\n'
            f.write(a1)

###grp
grps = regulons.explain_grp(model_gat,1000,ignore_tf='ENSMUSG00000000078')
grps.to_csv('D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_data_new_10000_02_5000_2500\\model\\nhead0_endim8_nhidden2_lr-3_epoch50_batch_size_20\\grp_top_100.txt')
print(grps)
