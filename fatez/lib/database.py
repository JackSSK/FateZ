#!/usr/bin/env python3
"""
This file contains basic database objects for storing data.

author: jy
"""
import torch
from torch.utils.data import Dataset

def collate_fn(batch):
    """
    The collate function for DataLoader to become capable to handle PyG Data.
    """
    samples = [ele for ele in batch]
    labels = torch.tensor([ele.y for ele in batch])
    return samples, labels


class FateZ_Dataset(Dataset):
    """
	Basic Dataset object for DataLoader.
    Saving data as PyG Data objects.
	"""
    def __init__(self, samples, tf_boundary:int = 0, gene_list = None, **kwargs):
        self.samples = samples
        self.tf_boundary = tf_boundary
        self.gene_list = gene_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = [sample.x, sample.edge_index, sample.edge_attr]
        return self.samples[idx]



# if __name__ == '__main__':
