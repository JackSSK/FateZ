#!/usr/bin/env python3
"""
Generate data loader object for training and testing.
Using DDP for multi-gpu training.

author: jy
"""
import os
import torch
from scipy.sparse import csr_matrix, lil_matrix, hstack
import numpy as np
import pandas as pd
import fatez.lib as lib
from fatez.lib.grn import load_grn_dict
import fatez.tool.JSON as JSON
from time import time
import anndata as ad, hdf5plugin
import torch_geometric.data as pyg_d
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def make_data_loader(
        corpus:ad.AnnData = None,
        peak_layer:str = '5k_bps',
        ens_ids:dict = None,
        grn_dict:dict = None,
        batch_size:int = 5,
        dtype = torch.float32,
    ):
    """
    :return:
        torch.utils.data.DataLoader
    """
    samples = [None] * len(corpus.obs.index)
    ind_order = pd.Index(corpus.var.index).get_indexer(ens_ids['order'].keys())

    for i,k in enumerate(corpus.obs.index):
        gex = corpus.X[i]
        peak = corpus.layers[peak_layer][i]
        grn = grn_dict[corpus.obs['grn'].iloc[i]].X.to_memory().tocoo()

        # Convert indices and data to PyTorch tensors
        edge_attr = torch.FloatTensor(grn.data)
        if len(edge_attr.shape) == 1:
            edge_attr = torch.reshape(edge_attr, (len(edge_attr), 1))

        samples[i] = pyg_d.Data(
            x = np.vstack(
                (gex.toarray()[:,ind_order], peak.toarray()[:,ind_order])
            ).transpose(),
            edge_index = torch.LongTensor(np.vstack((grn.row, grn.col))),
            edge_attr = edge_attr,
            y = corpus.obs['batch'].iloc[i],
            shape = grn.shape,
        ).to_dict()
        print(samples[i])
        samples[i] = pyg_d.Data(**samples[i])
        print(samples[i])
        print(f'processed {i}: {k}')
        

    return DataLoader(
        lib.FateZ_Dataset(
            samples,
            tf_boundary = ens_ids['tf_boundary'],
            gene_list = ens_ids['order'].keys(),
        ),
        batch_size = batch_size,
        collate_fn = lib.collate_fn,
        shuffle=True,
    )

def integrate_corpus(
        gex_corpus = None,
        promoter_corpus = None,
        tss_corpus = None,
        gene_dict = None,
        layer_name:str = '5k_bps',
        promoter_levels:int = 5,
        save_path:str = None,
    ):

    # Init
    temp_mat = lil_matrix(gex_corpus.X.shape)
    tss_down_peaks = {k: None for k in tss_corpus.var.index}
    promoter_peaks = {k: None for k in promoter_corpus.var.index}
    
    # Start iterations on each gene for making the TSS downstream matrix
    index_temp = list()
    for i, gene in enumerate(gex_corpus.var.index):
        gene_rec = gene_dict['ens_id'][gene]['tss_region']
        if len(gene_rec) == 1 and gene_rec[0] in tss_down_peaks:
            index_temp.append(gene_rec[0])
        else:
            raise Exception(f'{gene} not found TSS downstream {gene_rec[0]}')
    # Reorder tss_corpus.X
    temp_mat = tss_corpus[:, index_temp].to_memory().X
    assert temp_mat.shape == gex_corpus.shape
    gex_corpus.layers[layer_name] = temp_mat.tolil()

    os.system(f'echo "Finished TSS downstream matrix"')

    
    for level in range(promoter_levels):
        temp_index = list()
        corpus_index = list()
        for i, gene in enumerate(gex_corpus.var.index):
            gene_rec = gene_dict['ens_id'][gene]['promoter']
            # Get promoter_counts
            if level < len(gene_rec) and gene_rec[level] in promoter_peaks:
                temp_index.append(i)
                corpus_index.append(gene_rec[level])
        
        os.system(f'echo "Curating for level {level}"') 
        temp_mat = lil_matrix(gex_corpus.X.shape)
        temp_mat[:, temp_index] = promoter_corpus[:, corpus_index].to_memory().X.tolil()
        gex_corpus.layers[layer_name] = np.add(gex_corpus.layers[layer_name], temp_mat,)
        os.system(f'echo "processed level {level}"')

    gex_corpus.layers[layer_name] = gex_corpus.layers[layer_name].tocsr()

    # Save corpus
    if save_path is not None:
        gex_corpus.write_h5ad(
            save_path,
            compression = hdf5plugin.FILTERS["zstd"],
        )

    return gex_corpus


if __name__ == '__main__':
    corpus = None
    backed = True
    dir = "/data/core-genlmu/e-gyu/data/scRNA_scATAC/"
    integrate_corpus_path = dir + "adata/ptest.integrated.5k_bps.Corpus.h5ad"
    unified_grns_path = dir + "grn/unify_hfilter/"
    ens_ids = '/data/core-genlmu/e-gyu/FateZ/fatez/data/ens_ids.hfilter.json.gz'

    # Integrate corpus

    # Load in data
    gene_dict = JSON.decode('../../fatez/data/gene_dict/main.json.gz')
    gex_corpus = ad.read_h5ad(
        dir + 'adata/gex.harmonized.Corpus.h5ad',
        # dir + 'adata/gex.hfilter.Corpus.h5ad',
        backed = backed
    )
    promoter_corpus = ad.read_h5ad(
        dir + 'adata/promoter.harmonized.Corpus.h5ad',
        backed = backed
    )
    tss_corpus = ad.read_h5ad(
        dir + 'adata/tss_nk.harmonized.Corpus.h5ad',
        backed = backed
    )

    print('Start Integrating...')
    t1 = time() 
    corpus = integrate_corpus(
        gex_corpus = gex_corpus,
        promoter_corpus = promoter_corpus,
        tss_corpus = tss_corpus,
        gene_dict = gene_dict,
        layer_name = '5k_bps',
        save_path = integrate_corpus_path,
    )
    t2 = time() 
    print(f'Executed in {(t2-t1):.4f}s') 
    print('Finished Integrating.')

    # Load integrated corpus
    if corpus is None:
        corpus = ad.read_h5ad(
            integrate_corpus_path,
            backed = backed,
        )

    # Load cluster map
    corpus.obs['grn'] = pd.DataFrame.from_dict(
        JSON.decode(dir + 'grn/cluster_map.json'),
        orient='index',
        columns=['cluster']
    )
    print('Loaded GRN cluster map')
    
    # Load unified grn
    grn_dict = load_grn_dict(
        dir + 'grn/grns.json.gz',
        cache_path = unified_grns_path,
        load_cache = True,
        backed = backed,
    )
    print('Loaded unified GRN dict')

    # data_loader = make_data_loader(
    #     corpus = corpus,
    #     peak_layer = '5k_bps',
    #     ens_ids = JSON.decode(ens_ids),
    #     grn_dict = grn_dict,
    #     batch_size = 5,
    #     dtype = torch.float32,
    # )
    # print('made data_loader')