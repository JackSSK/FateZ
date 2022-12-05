#!/usr/bin/env python3
"""
This folder contains basic objects for FateZ.

author: jy
"""


class Error(Exception):
    pass



class GRN_Basic(object):
	"""
	Basic object for GRN components.
	"""

	def __init__(self, id:str = None):
		super(GRN_Basic, self).__init__()
		self.id = id

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
		setattr(self, key, kwargs[key])
		return
