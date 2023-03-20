#!/usr/bin/env python3
"""
This folder contains basic objects for FateZ.

author: jy
"""
from fatez.lib.database import FateZ_Dataset
from fatez.lib.database import Adj_Mat

__all__ = [
    'Adj_Mat',
    'FateZ_Dataset',
    'GRN_Basic',
]



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
