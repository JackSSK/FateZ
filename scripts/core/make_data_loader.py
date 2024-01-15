#!/usr/bin/env python3
"""
Generate data loader object for training and testing.
Using DDP for multi-gpu training.

author: jy
"""
import torch
from scipy.sparse import csr_matrix, lil_matrix, hstack
import numpy as np
import pandas as pd
import fatez.tool.JSON as JSON
import fatez.lib as lib
from fatez.lib.grn import load_grn_dict
import anndata as ad, hdf5plugin
import torch_geometric.data as pyg_d
from torch.utils.data import DataLoader

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
    index_list = pd.Index(corpus.var.index).get_indexer(ens_ids['order'].keys())

    for i,k in enumerate(corpus.obs.index):
        gex = corpus.X[i]
        peak = corpus.layers[peak_layer][i]
        grn = grn_dict[corpus.obs['grn'].iloc[i]].X.to_memory().tocoo()

        # Convert indices and data to PyTorch tensors
        values = torch.FloatTensor(grn.data)
        if len(values.shape) == 1:
            values = torch.reshape(values, (len(values), 1))


        samples[i] = pyg_d.Data(
            x = np.vstack(
                (gex.toarray()[:,index_list], peak.toarray()[:,index_list])
            ).transpose(),
            edge_index = torch.LongTensor(np.vstack((grn.row, grn.col))),
            edge_attr = values,
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
        genes_dict = None,
        layer_name:str = '5k_bps',
        save_path:str = None,
    ):
    def _get_counts(adata = None, var_list = None, answer = None,):
        for var in var_list:
            try:
                count = adata[:, var].to_memory().X
            except:
                # Should be euqiavlent to:
                # if var not in adata.var.index:
                # But runs faster since not searching the whole index
                continue
            if answer is None:
                answer = count
            else:
                answer += count
        return answer

    # Init
    temp_mat = lil_matrix(gex_corpus.X.shape)
    
    print('start processing')
    for i,gene in enumerate(gex_corpus.var.index):
        temp_mat[:, i] = _get_counts(
            adata = promoter_corpus,
            var_list = genes_dict['ens_id'][gene]['promoter'],
            answer = _get_counts(
                adata = tss_corpus,
                var_list = genes_dict['ens_id'][gene]['tss_region'],
            ),
        )
        print(f'processed {i}: {gene}')
    
    gex_corpus.layers[layer_name] = temp_mat.tocsr()
    gex_corpus.write_h5ad(save_path, compression = hdf5plugin.FILTERS["zstd"],)
    return gex_corpus


if __name__ == '__main__':
    dir = "/data/core-genlmu/e-gyu/data/scRNA_scATAC/"
    integrate_corpus_path = dir + "adata/integrated.5k_bps.Corpus.h5ad"
    unify_path = dir + "grn/unify_hfilter/"
    backed = True

    # # Integrate corpus
    # corpus = integrate_corpus(
    #     gex_corpus = ad.read_h5ad(
    #         dir + 'adata/gex.tpm.Corpus.h5ad',
    #         # dir + 'adata/gex.hfilter.Corpus.h5ad',
    #         backed = backed
    #     ),
    #     promoter_corpus = ad.read_h5ad(
    #         dir + 'adata/promoter.harmonized.Corpus.h5ad',
    #         backed = backed
    #     ),
    #     tss_corpus = ad.read_h5ad(
    #         dir + 'adata/tss_nk.harmonized.Corpus.h5ad',
    #         backed = backed
    #     ),
    #     genes_dict = JSON.decode('../../fatez/data/genes.json.gz'),
    #     layer_name = '5k_bps',
    #     save_path = integrate_corpus_path,
    # )

    ens_ids = '/data/core-genlmu/e-gyu/FateZ/fatez/data/ens_ids.hfilter.json.gz'
    integrate_corpus_path = dir + "adata/test.integrated.5k_bps.Corpus.h5ad"

    # Load integrated corpus
    corpus = ad.read_h5ad(integrate_corpus_path, backed = backed)

    # Load cluster map
    corpus.obs['grn'] = pd.DataFrame.from_dict(
        JSON.decode(dir + 'grn/cluster_map.json'),
        orient='index',
        columns=['cluster']
    )
    
    # Load unified grn
    grn_dict = load_grn_dict(
        dir + 'grn/grns.json.gz',
        cache_path = unify_path,
        load_cache = True,
        backed = backed,
    )
    print('loaded unified grn_dict')

    data_loader = make_data_loader(
        corpus = corpus,
        peak_layer = '5k_bps',
        ens_ids = JSON.decode(ens_ids),
        grn_dict = grn_dict,
        batch_size = 5,
        dtype = torch.float32,
    )
    print('made data_loader')