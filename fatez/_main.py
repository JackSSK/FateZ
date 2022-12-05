#!/usr/bin/env python3
"""
Template for the main control panel

author: jy
"""

import os
import sys
import copy
import threading
from pkg_resources import resource_filename


class Launch:
	"""
	Guess we will have a general object for one-line usage
	"""
	def __init__(self, 		# leave self at first line
		path:str = None, 	# No need to add comment here
		thread_num:int = 1, #
		**kwargs,			# leave the bracket to a new line
		):
		"""
		Pipeline to launch FateZ.

		:param path: <str Default = None>
			Comments will be here, like "Path to datasets".

		:param thread_num: <int Default = 1>
			Number of threads for multithreading. By default, no multithreading.

		Attributes:
			:self.path: Path to datasets
		"""
		# If variables are not expected to be used out of current object,
		# name it with _XXX.
		# Same rule is applied for functions.
		# Note: if functions named as __XXX, they WON't be inherited

		self.path = path
		self._thread_num = thread_num

	def save_result(self,
		savepath:str = None,
		**kwargs,
		):
		"""
		To save running results

		:param kwargs: arguments

		:return: None
		"""
		print('Saving')
		return

	def _proto_default(self, **kwargs):
		"""
		Default processing Protocol which does NOT use multithreading

		:param kwargs: arguments

		:return: None
		"""
		assert self._thread_num == 1
		print('Check')
		return

	def _proto_multi(self, **kwargs):
		"""
		Protocol MULTI for multithreading

		:param kwargs: arguments

		:return: None
		"""
		assert self._thread_num > 1
		print(self._thread_num)
		return
