import numpy as np
import fatez.process.preprocessor as pre
import pandas as pd
import os
from fatez.tool import PreprocessIO
os.chdir("D:\\Westlake\\pwk lab\\fatez\\FateZ\\fatez\\test")
"""
preprocess
"""
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## preprocess parameters
pseudo_cell_num_per_cell_type = 5000
correlation_thr_to_get_gene_related_peak = 0.6
rowmean_thr_to_get_variable_gene = 0.1
cluster_use =[1,4]
peak_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/atac_AE_Pre10x/')
rna_path = ('D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/rna_AE_Pre10x/')
gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
# Let's fix the nomenclature to 'TF_target_tss_1500.txt.gz'
tf_db_path = 'E:\\public/TF_target_tss1500.txt.gz'
network = pre.Preprocessor(rna_path, peak_path, gff_path, tf_db_path, data_type='unpaired')
network.load_data(matrix_format='10x_unpaired')
### qc
network.rna_qc(rna_min_genes=1, rna_min_cells=1, rna_max_cells=5000000)
network.atac_qc(atac_min_cells=10, )
print(network.atac_mt)
### select cell type
atac_cell_type = pd.read_table(
 'D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/atac_cell_type_AE_Pre.txt',
  header=None)
atac_cell_type.index = atac_cell_type[0]
atac_cell_type = atac_cell_type[1]
rna_cell_type = pd.read_table(
 'D:\\Westlake\\pwk lab\\HSC development\\data\\GSE137117/rna_cell_type_AE_Pre.txt',
  header=None)
rna_cell_type.index = rna_cell_type[0]
rna_cell_type = rna_cell_type[1]
network.add_cell_label(rna_cell_type,modality='rna')
network.add_cell_label(atac_cell_type,modality='atac')
network.annotate_peaks()
network.make_pseudo_networks(data_type='unpaired',
                             network_number=pseudo_cell_num_per_cell_type,
                             network_cell_size = 10)
### pseudo network check
print(rna_cell_type[list(network.rna_mt[network.pseudo_network['AE0']['rna']].obs_names)])
print(atac_cell_type[list(network.atac_mt[network.pseudo_network['AE0']['atac']].obs_names)])
print(rna_cell_type[list(network.rna_mt[network.pseudo_network['Pre-HE0']['rna']].obs_names)])
print(atac_cell_type[list(network.atac_mt[network.pseudo_network['Pre-HE0']['atac']].obs_names)])

###output model
network.cal_peak_gene_cor(exp_thr = rowmean_thr_to_get_variable_gene,
                          cor_thr=correlation_thr_to_get_gene_related_peak)
matrix1 = network.output_pseudo_samples() ### exp count mt
matrix2 = network.generate_grp() ### correlation mt
network.extract_motif_score(matrix2)
matrix2 = np.multiply(network.motif_enrich_score,matrix2)

matrix2.to_csv('D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data_10000/edge_matrix.csv')

PreprocessIO.output_csv_dict_df(matrix1,
                        'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data_10000/node/')
# matrix1 = PreprocessIO.input_csv_dict_df(
#     'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_origi_label_mt/node/')

