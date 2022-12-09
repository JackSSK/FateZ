#!/usr/bin/env python3
"""
Read in MEX format Single Cell Expression Data

author: jy, nkmtmsys
"""
import csv
import gzip
import scipy.io
import pandas as pd
from warnings import warn


class Reader(object):
	"""
	Object to read in MEX data.
	"""
	def __init__(self,
		matrix_path:str = None,
		features_path:str = None,
		barcodes_path:str = None,
		):
		"""
		:param matrix_path: <str Default = None>
			Path to the matrix file.

		:param features_path: <str Default = None>
			Path to the feature (gene) file.

		:param barcodes_path: <str Default = None>
			Path to the barcode file.
		"""
		self.matrix_path = matrix_path

		# Process features
		self.features_path = features_path
		with gzip.open(self.features_path, 'rt') as features:
			self.features = pd.DataFrame(
				data = [x for x in csv.reader(features, delimiter = '\t')]
			)
		assert self.features.shape[1] >= 2

		# Process barcodes
		self.barcodes_path = barcodes_path
		with gzip.open(self.barcodes_path, 'rt') as barcodes:
			self.barcodes = [
				rec[-1] for rec in csv.reader(barcodes, delimiter = '\t')
			]

	def read_matrix(self):
		"""
		Read in matrix at once.
		"""
		return pd.DataFrame(scipy.io.mmread(self.matrix_path).toarray())

	def read_matrix_sparse(self):
		"""
		Read in files without loading full matrix.
		Go through matrix line by line and populate dataframe if there is a read

		Developing~
		"""
		return None

	def seperate_matrices(self, sparse_read:bool = False):
		"""
		Seperate matrices in MEX by data type.
		Useful for handling multiomic data.

		:param sparse_read:bool = False
			Whether load in matrix with sparse method or not.
		"""
		matrices = dict()
		assert self.features.shape[1] >= 3
		types = list(set(self.features[2]))
		# Load data
		if sparse_read:
			data = self.read_matrix_sparse()
			print('Developing~')
		else:
			data = self.read_matrix()
			for type in types:
				matrices[type] = data[self.features[2] == type]
		return matrices

	def get_gem(self,
		save_path:str = None,
		handle_repeat:str = 'sum',
		sparse_read:bool = False
		):
		"""
		Obtain GEM data frame from processed MEX file.

		:param save_path:str = None

		:param handle_repeat:str = 'sum'

		:return: Pandas.DataFrame
		"""
		if sparse_read:
			data = self.read_matrix_sparse()
		else:
			data = self.read_matrix()
		data.index = self.features[0]
		data.columns = self.barcodes
		# sum up data sharing same gene name if any
		if len(data.columns) != len(list(set(data.columns))):
			warn('Found repeated barcodes in MEX! Merging.')
			if handle_repeat == 'first':
				data=data[~data.columns.duplicated(keep='first')]
			elif handle_repeat == 'sum':
				data = data.groupby(data.columns).sum()

		# summ up data sharing same barcode if any
		if len(data.index) != len(list(set(data.index))):
			warn('Found repeated genes in MEX! Merging.')
			if handle_repeat == 'first':
				data = data[~data.index.duplicated(keep='first')]
			elif handle_repeat == 'sum':
				data = data.groupby(data.index).sum()

		# Check for repeats
		assert len(data.columns) == len(list(set(data.columns)))
		assert len(data.index) == len(list(set(data.index)))

		# Remove Gfp counts
		if 'bGH' in data.index:
			data.drop('bGH')
		if 'eGFP' in data.index:
			data.drop('eGFP')

		# save GEM if specified path to save
		if save_path is not None:
			data.to_csv(save_path)

		# Return df purned records only have 0 counts
		return data.loc[~(data==0).all(axis=1)]
