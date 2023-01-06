import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import fatez.tool.JSON as JSON
import fatez.model as model
import fatez.model.gat as gat
import fatez.model.mlp as mlp
import fatez.model.sparse_gat as sgat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.process.preprocessor as pre
import pandas as pd
from transformers import AdamW
from torch.utils.data import DataLoader
import random
import fatez.lib as lib
### preprocess parameters
pseudo_cell_num_per_cell_type = 200
correlation_thr_to_get_gene_related_peak = 0.6
rowmean_thr_to_get_variable_gene = 0.1
cluster_use =[1,4]
### preprocess
"""
↓peak_pathとrna_pathはdata？
"""
peak_path = ('../data/mouse/filtered_feature_bc_matrix/')
rna_path = ('../data/mouse/filtered_feature_bc_matrix/')
gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'

# tf_db_path = '../data/ignore/TF_target_tss_1500.txt.gz'
# cell_type_path = '..data/ignore/e18_mouse_brain_fresh_5k/analysis/clustering/gex/graphclust/clusters.csv'


#↑のpath使ってください
tf_db_path = 'E:\\public/TF_target_tss_1500.txt.gz'
cell_type_path = 'E:\\public\\public data\\10X\\e18_mouse_brain_fresh_5k\\e18_mouse_brain_fresh_5k_analysis\\analysis\\clustering\\gex\\graphclust/clusters.csv'

network = pre.Preprocessor(rna_path, peak_path, gff_path, tf_db_path,
                           data_type='paired')
network.load_data(matrix_format='10x_paired')
### qc
network.rna_qc(rna_min_genes=3, rna_min_cells=250, rna_max_cells=2500)
network.atac_qc(atac_min_features=3, )
### select cell type
cell_type = pd.read_csv(cell_type_path)
cell_type.index = cell_type['Barcode']
cell_type = cell_type['Cluster']
cell_type = cell_type[cell_type.isin(cluster_use)]
network.add_cell_label(cell_type)
network.annotate_peaks()
network.make_pseudo_networks(data_type='paired',
                             network_number=pseudo_cell_num_per_cell_type)
network.cal_peak_gene_cor(exp_thr = rowmean_thr_to_get_variable_gene,
                          cor_thr=correlation_thr_to_get_gene_related_peak)
matrix1 = network.output_pseudo_samples() ### exp count mt
matrix2 = network.generate_grp() ### correlation mt
"""
samples, labels, and iter
"""
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m2 = matrix2[list(matrix2.keys())[i]]
    #motif_enrich_mt = network.extract_motif_score(m2)
    m2 = torch.from_numpy(m2.to_numpy())
    #m2 = np.multiply(m2,motif_enrich_mt)
    m1 = m1.to(torch.float32)
    m2 = m2.to(torch.float32)
    print(m1.shape)
    print(m2.shape)
    samples.append([m1, m2])
sample_idx = torch.tensor(range(len(samples)))
# Parameters
batch_size = 20
num_epoch = 300
###iter
def data_iter(batch_size,mt1,labels):
    num_examples = len(mt1)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield mt1.index_select(0,j),labels.index_select(0,j)

labels = torch.from_numpy(np.repeat(range(len(cluster_use))
                                    ,pseudo_cell_num_per_cell_type))
labels = labels.long()
"""
model define
"""
model_gat = gat.GAT(
    d_model = 2,
    en_dim = 8,
    nhead = None,
    device = device,
)
decison = mlp.Classifier(
    d_model = 8,
    n_hidden = 4,
    n_class = 2,
    device = device,
)


# Need to make sure d_model is divisible by nhead
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
optimizer = torch.optim.Adam(test_model.parameters(),
                             lr=0.0001,
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=2,
                                                                 T_mult=2,
                                                                 eta_min=1e-4/50)

"""
traning
"""
batch_size = 20
num_epoch = 500
train_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=samples, labels=labels),
    batch_size=batch_size,
    shuffle=True
)
for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    batch_num = 1
    test_model.train()
    out_gat_data = list()
    for x,y in train_dataloader:
        optimizer.zero_grad()
        out_gat = model_gat(x[0], x[1])
        # Save GAT output without decison layers
        out_gat_data.append(out_gat.detach().tolist())
        # torch.save(out_gat,
        #            'D:\\Westlake\\pwk lab\\fatez\\out_gat/epoch'+str(epoch)+'_batch'+str(batch_num)+'.pt')
        output = test_model(x[0], x[1])
        loss = nn.CrossEntropyLoss()(
            output, y
        )
        loss.backward()
        optimizer.step()
        acc = (output.argmax(1)==y).type(torch.float).sum()/batch_size
        print(f"batch: {batch_num} loss: {loss} accuracy:{acc}")
        batch_num += 1
    scheduler.step()
    # Saving GAT outputs by epoch
    JSON.encode(
        out_gat_data,
        'D:\\Westlake\\pwk lab\\fatez\\out_gat/epoch' + str(epoch) + '.pt'
    )
model.Save(test_model.bert_model.encoder, '../data/ignore/bert_encoder.model')
