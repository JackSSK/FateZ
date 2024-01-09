#!/usr/bin/env python3
"""
Generate data loader object for training and testing.
Using DDP for multi-gpu training.

author: jy
"""
import pandas as pd
import fatez.tool.JSON as JSON
from fatez.lib.grn import load_grn_dict
import anndata as ad, hdf5plugin
import torch_geometric.data as pyg_d
from torch.utils.data import DataLoader

# def make_data_loader(self,
#         simpler_samples:bool = True,
#         n_sample:int = 10,
#         batch_size:int = 5
#     ):
#     """
#     Generate random data with given parameters and
#     set up a PyTorch DataLoader.

#     :return:
#         torch.utils.data.DataLoader
#     """
#     assert self.config['fine_tuner']['n_class'] == 2
#     samples = list()

#     def rand_sample():
#         input_sz = self.config['input_sizes']
#         fea_m = torch.abs(torch.randn(
#             (input_sz['n_node'], input_sz['node_attr']),
#             dtype = self.factory_kwargs['dtype']
#         ))
#         adj_m = torch.randn(
#             (input_sz['n_reg'], input_sz['n_node'], input_sz['edge_attr']),
#             dtype = self.factory_kwargs['dtype']
#         )
#         # Zero all features if making simple samples
#         if simpler_samples:
#             fea_m = fea_m * 0 + 1
#             adj_m = adj_m * 0 + 1
#         # Take last two features as testing features
#         fea_m[-2:] *= 0
#         fea_m[-2:] += 1
#         adj_m[:,-2:] *= 0
#         return fea_m, adj_m

#     def append_sample(samples, fea_m, adj_m, label):
#         inds, attrs = lib.get_sparse_coo(adj_m)
#         samples.append(
#             pyg_d.Data(
#                 x = fea_m,
#                 edge_index = inds,
#                 edge_attr = attrs,
#                 y = label,
#                 shape = adj_m.shape,
#             )
#         )

#     # Prepare type_0 samples
#     for i in range(int(n_sample / 2)):
#         fea_m, adj_m = rand_sample()
#         fea_m[-1] += 8
#         adj_m[:,-1] += 9
#         append_sample(samples, fea_m, adj_m, label = 0)

#     # Prepare type_1 samples
#     for i in range(n_sample - int(n_sample / 2)):
#         fea_m, adj_m = rand_sample()
#         adj_m[:,-1] += 1
#         append_sample(samples, fea_m, adj_m, label = 1)

#     return DataLoader(
#         lib.FateZ_Dataset(samples),
#         batch_size=batch_size,
#         collate_fn = lib.collate_fn,
#         shuffle=True,
#     )


if __name__ == '__main__':
    dir = "/data/core-genlmu/e-gyu/data/scRNA_scATAC/"
    unify_path = dir + "grn/unify/"


    gex_corpus = ad.read_h5ad(
        dir + 'adata/gex.hfilter.Corpus.h5ad',
        backed=True
    )

    gex_corpus = ad.read_h5ad(
        dir + 'adata/gex.harmonized.Corpus.h5ad',
        backed=True
    )

    promoter_corpus = ad.read_h5ad(
        dir + 'adata/promoter.harmonized.Corpus.h5ad',
        backed=True
    )

    tss_corpus = ad.read_h5ad(
        dir + 'adata/tss_nk.harmonized.Corpus.h5ad',
        backed=True
    )

    # Load cluster map
    gex_corpus.obs['grn'] = pd.DataFrame.from_dict(
        JSON.decode(dir + 'grn/cluster_map.json'),
        orient='index',
        columns=['cluster']
    )

    # Load unified grn
    grn_dict = load_grn_dict(
        dir + 'grn/grns.json.gz',
        cache_path = unify_path,
        load_cache = True,
    )
    print(grn_dict.keys())
    print('loaded unified grn_dict')
