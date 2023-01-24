#!/usr/bin/env python3
"""
Note:

I suppose you want to explain BERT Fine Tune Model, not the decision model

"""
import fatez.lib as lib
from torch.utils.data import DataLoader
from fatez.tool import identify_regulons
import fatez.tool.JSON as JSON
import fatez.model.mlp as mlp
import torch
import fatez.process.explainer as explainer
import fatez.process.fine_tuner as fine_tuner
import fatez.model.bert as bert
import fatez.model as model
import shap
from fatez.tool import PreprocessIO
import pandas as pd
import numpy as np
import torch.nn as nn
from fatez.tool import model_testing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp_param = {
    'd_model': 8,
    'n_hidden': 4,
    'n_class': 2,
    'device': device,
    'dtype': torch.float32,
}

batch_size = 20
fine_tuning = fine_tuner.Model(
    gat = model.Load('D:\\Westlake\\pwk lab\\fatez\\gat_gradient\\nhead0_nhidden2_lr-3_epoch200_batch_size_20\\gat.model'),
    bin_pro = model.Binning_Process(n_bin = 100,config = None),
    bert_model = model.Load('D:\\Westlake\\pwk lab\\fatez\\gat_gradient\\nhead0_nhidden2_lr-3_epoch200_batch_size_20\\bert_fine_tune.model')
)
fine_tuning.to(device)
matrix1 = PreprocessIO.input_csv_dict_df(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data_10000/node/'
)
matrix2 = pd.read_csv(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data_10000/edge_matrix.csv',
    index_col = 0
)
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
acc_all = 0
explain_use = []
with torch.no_grad():
    for x,y in train_dataloader:
        pred = fine_tuning(x[0].to(device), x[1].to(device))
        test_loss = nn.CrossEntropyLoss()(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        if correct == batch_size:
            explain_use.append(x)
        acc_all+=correct
print(acc_all)
### sum rank
explain_use = explain_use[0:10]
regulons = identify_regulons.Regulon(explain_use)
regulons.explain_model(fine_tuning,batch_size,matrix2)
regulon_count = regulons.sum_regulon_count()
# regulon_count.index = list(matrix2.columns)
# regulon_count = regulon_count[list(matrix2.index)]
regulon_count = regulon_count.sort_values()
print('---sum rank---')
print(regulon_count)

### count top
regulon_top = regulons.get_top_regulon_count(20)
regulon_top = pd.Series(regulon_top)
regulon_top = regulon_top.sort_values(ascending=False)
print('---count top---')
print(regulon_top)
print(regulons.tf_names[regulon_top.index])

gene_name = pd.read_table('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117\\rna_AE_Pre10x\\features.tsv.gz',header=None)
gene_name.index = gene_name[0]
sum_top = regulons.tf_names[regulon_count.index][0:20]
count_top = regulons.tf_names[regulon_top.index][0:20]

print('sum rank top 20:')
top_symbol = []
for i in sum_top:
    symbol = gene_name[gene_name[0].isin([i])][1].to_numpy().tolist()[0]
    top_symbol.append(symbol)
print(top_symbol)


print('count top top 20:')
top_symbol = []
for i in count_top:
    symbol = gene_name[gene_name[0].isin([i])][1].to_numpy().tolist()[0]
    top_symbol.append(symbol)
print(top_symbol)

