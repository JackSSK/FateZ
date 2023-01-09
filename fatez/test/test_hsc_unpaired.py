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
"""
preprocess
"""
device='cpu'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## preprocess parameters
pseudo_cell_num_per_cell_type = 400
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
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_origi_label_mt/node/')
matrix2 = pd.read_csv(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_origi_label_mt/edge_matrix.csv'
    ,index_col=0)


### samples and labels
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m2 = torch.from_numpy(matrix2.to_numpy())
    m1 = m1.to(torch.float32)
    m2 = m2.to(torch.float32)
    samples.append([m1, m2])
labels = torch.from_numpy(np.repeat(range(len(cluster_use))
                                    ,len(matrix1)/len(cluster_use)))
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
### adam and CosineAnnealingWarmRestarts
optimizer = torch.optim.Adam(test_model.parameters(),
                             lr=0.0001,
                             weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=2,
                                                                 T_mult=2,
                                                                 eta_min=0.0001/50)
model_gat.to(device)
bert_encoder.to(device)
test_model.to(device)

"""
traning
"""
batch_size = 20
num_epoch = 1500
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
    epoch_loss = 0
    for x,y in train_dataloader:
        optimizer.zero_grad()
        out_gat = model_gat(x[0], x[1])
        # torch.save(out_gat,
        #            'D:\\Westlake\\pwk lab\\fatez\\out_gat/epoch'+str(epoch)+'_batch'+str(batch_num)+'.pt')
        output = test_model(x[0], x[1])
        out_gat_data.append(out_gat.detach().tolist())
        loss = nn.CrossEntropyLoss()(
            output, y
        )
        loss.backward()
        optimizer.step()
        acc = (output.argmax(1)==y).type(torch.float).sum()/batch_size
        print(f"batch: {batch_num} loss: {loss} accuracy:{acc}")
        batch_num += 1
        epoch_loss += loss
    print(f"epoch: {epoch}, loss: {epoch_loss}")
    scheduler.step()
    # JSON.encode(
    #     out_gat_data,
    #     'D:\\Westlake\\pwk lab\\fatez\\out_gat_hsc/epoch' + str(epoch) + '.pt'
    # )
model.Save(test_model.bert_model.encoder, '../data/ignore/unpaired_bert_encoder.model')
