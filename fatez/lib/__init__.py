#!/usr/bin/env python3
"""
This folder contains basic objects for FateZ.

author: jy
"""
from torch.utils.data import Dataset



class Error(Exception):
    pass



class GRN_Basic(object):
	"""
	Basic object for GRN components.
	"""

	def __init__(self, id:str = None, **kwargs):
		super(GRN_Basic, self).__init__()
		self.id = id
        # if there are other args
		for key in kwargs: setattr(self, key, kwargs[key])

	def as_dict(self):
		"""
		Cast object into dict type.

		:return: <class 'dict'>
		"""
		return self.__dict__

	def add_attr(self, key, value):
		"""
		Add new attribute to the object.

		:param key: <str>
			The key of new attribute.

		:param value: <>
			The value of new attribute.
		"""
		setattr(self, key, value)
		return


class FateZ_Dataset(Dataset):
    """
	Basic Dataset object for DataLoader
	"""
    def __init__(self, samples, labels):
        assert len(samples) == len(labels)
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
