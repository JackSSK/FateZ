# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:52:04 2023

@author: jjy
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import celloracle as co
import anndata as an

rna_path = '/storage/peiweikeLab/jiangjunyao/public/public_data/GSE117089/rna/'
cluster_path = '/storage/peiweikeLab/jiangjunyao/public/public_data/GSE117089/cluster.txt'
fatez_gene_path = '/storage/peiweikeLab/jiangjunyao/public/fatez_ens_symbol.csv'
fatez_tf_path = '/storage/peiweikeLab/jiangjunyao/public/fatez_tf_use.txt'
output_path = '/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge/'
sample_name = ''
ncores = 10

def read_mtx(rna_path):
    mtx = an.read_mtx(rna_path + 'matrix.mtx.gz')
    gene = pd.read_table(rna_path + 'features.tsv.gz',
                                     header=None)
    barcode = pd.read_table(rna_path + 'barcodes.tsv.gz', header=None)
    mtx.obs_names = list(gene[0])
    mtx.var_names = list(barcode[0])
    rna_mt = mtx.T
    return rna_mt

def add_cell_label(rna_mt,cell_types,modality:str = 'rna'):
    for i, name in enumerate(cell_types['label']):
        if ' ' in str(name):
            cell_types['label'][i] = cell_types['label'][i].replace(' ', '-')
        if '/' in str(name):
            cell_types['label'][i] = cell_types['label'][i].replace('/', '-')
    if modality == 'rna':
        rna_mt = rna_mt[
            np.intersect1d(cell_types.index, rna_mt.obs_names)
        ]
        cell_types = cell_types.loc[list(rna_mt.obs_names)]['label']
        rna_mt.obs['cell_types'] = list(cell_types)
    return rna_mt

def format_grp(grp,gene_path,tf_use_path):
    gene_corr = pd.read_csv(gene_path)
    tf_all = pd.read_table(tf_use_path,header=None)
    tf_all = tf_all[0]
    gene_all = gene_corr['ENS']
    gene_corr.index = gene_corr['symbol'].to_list()

    if grp.index[0].startswith('ENS'):
        
        index_intersect = list(np.intersect1d(grp.index,
                                              list(gene_corr['ENS'])))
        column_intersect = list(np.intersect1d(grp.columns,
                                               list(gene_corr['ENS'])))
        grp = grp.loc[index_intersect, column_intersect]
        grp = grp.loc[list(np.intersect1d(tf_all, grp.index))]
        
    else:
        
        index_intersect = list(np.intersect1d(grp.index,
                                              list(gene_corr['symbol'])))
        column_intersect = list(np.intersect1d(grp.columns,
                                               list(gene_corr['symbol'])))
        grp = grp.loc[index_intersect, column_intersect]
        gene_corr = pd.Series(gene_corr['ENS'].to_list(),
                              index=gene_corr['symbol'])
        gene_corr = gene_corr[[not i for i in gene_corr.index.duplicated()]]
        grp.index = gene_corr[grp.index].values
        grp.columns = gene_corr[grp.columns].values
        grp = grp.loc[list(np.intersect1d(tf_all,grp.index))]
        
    tf_diff = list(np.setdiff1d(tf_all, list(grp.index)))
    zero_mt = pd.DataFrame(np.zeros((len(tf_diff), len(grp.columns))))
    zero_mt.index = tf_diff
    zero_mt.columns = list(grp.columns)
    grp = pd.concat([grp, zero_mt])
    gene_diff = list(np.setdiff1d(gene_all, list(grp.columns)))
    zero_mt = pd.DataFrame(np.zeros((len(grp.index), len(gene_diff))))
    zero_mt.index = list(grp.index)
    zero_mt.columns = gene_diff
    grp = pd.concat([grp, zero_mt], axis=1)
    
    return grp

###preprocess
adata = read_mtx(rna_path)
cluster = pd.read_table(cluster_path,header=0)
cluster.index = cluster['sample']
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all')
filter_result = sc.pp.filter_genes_dispersion(adata.X,
                                              flavor='cell_ranger',
                                              n_top_genes=4000,
                                              log=False)
adata = adata[:, filter_result.gene_subset]
sc.pp.normalize_per_cell(adata)
adata.raw = adata
adata.layers["raw_count"] = adata.raw.X.copy()
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')
sc.tl.louvain(adata, resolution=0.8)
adata =add_cell_label(adata,cluster)
sc.tl.paga(adata, groups='cell_types')

###infer grn
oracle = co.Oracle()
adata.X = adata.layers["raw_count"].copy()
oracle.import_anndata_as_raw_count(adata=adata,
                                   cluster_column_name="cell_types",
                                   embedding_name="X_diffmap")
base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()
oracle.import_TF_data(TF_info_matrix=base_GRN)
oracle.perform_PCA()
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
n_cell = oracle.adata.shape[0]
k = int(0.025*n_cell)
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=ncores)
links = oracle.get_links(cluster_name_for_GRN_unit="cell_types", alpha=10,
                         verbose_level=10)

###output
for i in links.links_dict:
    df1 = links.links_dict[i]
    df2 = df1.pivot(index='source', columns='target', values='-logp')
    df2 = df2.fillna(0)
    df2 = format_grp(df2,fatez_gene_path,fatez_tf_path)
    df2.to_csv(output_path+sample_name+i+'.csv')
    



















