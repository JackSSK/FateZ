#!/usr/bin/env python3
"""
Gene Expression Matrix related tools

ToDo: make SQL Database based on folder/file
Find a way to deal with seurat objects
author: jy, nkmtmsys
"""
import re
import pandas as pd
import fatez.tool as tool



class Reader(object):
	"""
	Class to read in RNA-seq based Gene Expression Matrices.
	Only suppordt .cvs and .txt for now.
	"""
	def __init__(self,
		path:str = None,
		handle_repeat:str = 'sum',
		**kwargs
		):

		# Decide which seperation mark to use
		if re.search(r'csv', path):
			self.sep = ','
		elif re.search(r'txt', path):
			self.sep = '\t'

		try:
			self.data = pd.read_csv(path, sep = self.sep, **kwargs)
		except Exception as GEM_Reader_Error:
			raise tool.Error('Unsupported File Type: ', path)

		# Just in case, repeated genes need to be solved
		if handle_repeat == 'first':
			self.data = self.data[~self.data.index.duplicated(keep = 'first')]
		elif handle_repeat == 'sum':
			self.data = self.data.groupby(self.data.index).sum()
