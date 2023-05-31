#!/usr/bin/env python3
"""
This file contains basic database objects for storing data.

author: jy
"""
from torch.utils.data import Dataset



class FateZ_Dataset(Dataset):
    """
	Basic Dataset object for DataLoader
	"""
    def __init__(self, samples, **kwargs):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = [sample.x, sample.edge_index, sample.edge_attr]
        return data, sample.y



# if __name__ == '__main__':
