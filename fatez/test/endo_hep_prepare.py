#!/usr/bin/env python3
"""
This is the playground for Multiomics rebuilder.

author: jy
"""
import torch
from torch.utils.data import DataLoader
import torch_geometric.data as pyg_d
import pandas as pd
import fatez.lib as lib
import fatez.tool.PreprocessIO as PreprocessIO
from fatez.process.scale_network import scale_network


def get_config():
    """
    Get a hard-coded config here
    """
    return {
        "input_sizes": {
            "n_reg": 100,
            "n_node": 100,
            "node_attr": 2,
            "edge_attr": 1
        },
        "graph_embedder": {
            "type": "skip",
            "params": {
                "n_dim": -2
            }
        },
        "gnn": {
            "type": "GATv2",
            "params": {
                "en_dim": 32,
                "n_hidden": 4,
                "nhead": 1
            }
        },
        "rep_embedder": {
            "type": "ABS",
            "params": {
                "n_embed": 1103,
                "n_dim": 32
            }
        },
        "encoder": {
            "d_model": 32,
            "n_layer": 6,
            "nhead": 4,
            "dim_feedforward": 512
        },
        "pre_trainer": {
            "train_adj": False,
            "masker_params": {
                "ratio": 0.15
            }
        },
        "fine_tuner": {
            "n_class": 2,
            "weight_decay": 0.0001,
            "sch_T_0": 2,
            "sch_T_mult": 2,
            "sch_eta_min": 2e-06,
            "adapter": "Lora",
            "clf_type": "CNN_HYB",
            "clf_params": {
                "n_layer_set": 1,
                "conv_kernel_num": 8,
                "verti_kernel_size": 8,
                "horiz_kernel_size": 3,
                "maxpool_kernel_size": 2,
                "densed_size": 128
            },
            "lr": 0.0001
        }
    }



def get_dataloaders(
    train_ratio:float = 0.8,
    batch_size:int = 5,
    data_header:str = '../data/rd_fake_data/',
    node_dir:str = 'all_node/',
    dtype = torch.float32,
    ):
    """
    Make FateZ_Dataset based on partioned endo&hep cell data
    """
    # Rescale Adj mat into 0-1 scale
    def rescale_edge_mat(mat):
        return torch.from_numpy(scale_network(mat)).to(torch.float32)

    # Load in nodes
    node_dict = PreprocessIO.input_csv_dict_df(
        data_header + node_dir,
        df_type = 'node',
        order_cell = False
    )
    # Load in adj mats
    edge_dict = {
        'hep':rescale_edge_mat(pd.read_table(data_header + 'hep_edge.txt')),
        'end':rescale_edge_mat(pd.read_table(data_header + 'endo_edge.txt')),
    }
    # Making label look-up dict
    label_dict = {'end':1, 'hep':0,}

    # Make PyG Data object sample list
    samples = list()
    for key,v in node_dict.items():
        inds, attrs = lib.get_sparse_coo(edge_dict[key[:3]])
        samples.append(
            pyg_d.Data(
                x = torch.from_numpy(v.to_numpy()).to(dtype),
                edge_index = inds,
                edge_attr = attrs,
                y = label_dict[key[:3]],
                shape = edge_dict[key[:3]].shape,
            )
        )

    # Make train/test data sets
    train_size = int(train_ratio * len(samples))
    train_dataset, test_dataset = torch.utils.data.random_split(
        lib.FateZ_Dataset(samples),
        [train_size, len(samples) - train_size]
    )

    # And then data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        collate_fn = lib.collate_fn,
        shuffle = True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = lib.collate_fn,
        shuffle = True,
    )

    return train_dataloader, test_dataloader

# if __name__ == '__main__':
#     train_dataloader, test_dataloader = get_dataloaders()
#     config = get_config()
