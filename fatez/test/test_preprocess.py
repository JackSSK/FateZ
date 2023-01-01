import fatez.process.preprocessor as pre
import os
import time
import pandas as pd
import numpy as np
import fatez.tool.sequence as seq
os.chdir("D:\\Westlake\\pwk lab\\fatez\\FateZ\\fatez\\test")
if __name__ == '__main__':
    peak_path = ('../data/mouse/filtered_feature_bc_matrix/')
    rna_path = ('../data/mouse/filtered_feature_bc_matrix/')
    gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
    tf_db_path = 'E:\\public/TF_target_tss_1500.txt.gz'

    ### load data
    network = pre.Preprocessor(rna_path, peak_path, gff_path,tf_db_path, data_type='paired')
    network.load_data(matrix_format='10x_paired')
    #print(network.rna_mt)
    #print(network.atac_mt)
    #print(network.tf_target_db.keys())
    ### qc
    network.rna_qc(rna_min_genes=5,rna_min_cells=250,rna_max_cells=2500)
    network.atac_qc(atac_min_features=5,)
    #print(network.rna_mt)
    #print(network.atac_mt)
    #print(network.peak_count.keys())
    ### merge & fix peak regions

    #network.merge_peak() wtf????
    #print(network.peak_count.keys())


    ### load cell type
    cell_type = pd.read_csv('E:\\public\\public data\\10X\\e18_mouse_brain_fresh_5k\\e18_mouse_brain_fresh_5k_analysis\\analysis\\clustering\\gex\\graphclust/clusters.csv')
    cell_type.index = cell_type['Barcode']
    cell_type = cell_type['Cluster']
    cell_type = cell_type[cell_type.isin([1,4])]
    network.add_cell_label(cell_type)
    print(len(network.pseudo_network))


    ### pseudo cell
    t1 = time.time()
    network.make_pseudo_networks(data_type='paired',network_number=2)
    t2 = time.time()
    print(t2 - t1)
    print(network.pseudo_network.keys())
    #print(network.pseudo_network)

    ### load gene peak annotation

    network.annotate_peaks()
    #print(network.peak_annotations.keys())

    ### calculate correlation between gene and peak
    ### select peak with top correlation
    t1 = time.time()
    network.cal_peak_gene_cor(exp_thr = 0.1)
    matrix1 = network.output_pseudo_samples()
    print(matrix1['10'])
    t2 = time.time()
    print(t2-t1)
    t1 = time.time()
    matrix2 = network.generate_grp()
    t2 = time.time()
    print(t2-t1)
    print('network network')
    print(len(matrix1))
    print(len(matrix2))
    ### construct grp with expressed genes and its tfs with motif enrichment in target promoter



