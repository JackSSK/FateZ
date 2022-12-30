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

## preprocess parameters
pseudo_cell_num_per_cell_type = 2
correlation_thr_to_get_gene_related_peak = 0.6
rowmean_thr_to_get_variable_gene = 0.1
### preprocess
peak_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/atac_10x/')
rna_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/rna_10x/')
gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
tf_db_path = 'E:\\public/TF_target_tss_1500.txt.gz'
network = pre.Preprocessor(rna_path, peak_path, gff_path, tf_db_path, data_type='paired')
network.load_data(matrix_format='10x_unpaired')
### qc
network.rna_qc(rna_min_genes=2, rna_min_cells=0, rna_max_cells=2500)
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