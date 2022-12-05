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
	Object to read in scRNA-seq MEX data.
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
		# Process features
		with gzip.open(features_path, 'rt') as features:
			self.features = [
				{'id':x[0], 'name':x[1]} for x in csv.reader(
					features, delimiter = '\t'
				)
			]

		# Process matrix
		self.data = pd.DataFrame(scipy.io.mmread(matrix_path).toarray())

		# Process barcodes
		with gzip.open(barcodes_path, 'rt') as barcodes:
			self.data.columns = [
				rec[-1] for rec in csv.reader(barcodes, delimiter = '\t')
			]

	def get_gem(self, save_path:str = None, handle_repeat:str = 'sum',):
		"""
		Obtain GEM data frame from processed MEX file.

		:param save_path:str = None

		:param handle_repeat:str = 'sum'
		
		:return: Pandas.DataFrame
		"""
		self.data.index = [x['id'] for x in self.features]

		# sum up data sharing same gene name if any
		if len(self.data.columns) != len(list(set(self.data.columns))):
			warn('Found repeated barcodes in MEX! Merging.')
			if handle_repeat == 'first':
				self.data=self.data[~self.data.columns.duplicated(keep='first')]
			elif handle_repeat == 'sum':
				self.data = self.data.groupby(self.data.columns).sum()

		# summ up data sharing same barcode if any
		if len(self.data.index) != len(list(set(self.data.index))):
			warn('Found repeated genes in MEX! Merging.')
			if handle_repeat == 'first':
				self.data = self.data[~self.data.index.duplicated(keep='first')]
			elif handle_repeat == 'sum':
				self.data = self.data.groupby(self.data.index).sum()

		# Check for repeats
		assert len(self.data.columns) == len(list(set(self.data.columns)))
		assert len(self.data.index) == len(list(set(self.data.index)))

		# Remove Gfp counts
		if 'bGH' in self.data.index:
			self.data.drop('bGH')
		if 'eGFP' in self.data.index:
			self.data.drop('eGFP')

		# save GEM if specified path to save
		if save_path is not None:
			self.data.to_csv(save_path)

		# Return df purned records only have 0 counts
		return self.data.loc[~(self.data==0).all(axis=1)]


# if __name__ == '__main__':
#     gem = Reader(
#         matrix_path = 'GSM4085627_10x_5_matrix.mtx.gz',
# 		features_path = 'GSM4085627_10x_5_genes.tsv.gz',
#         barcodes_path = 'GSM4085627_10x_5_barcodes.tsv.gz'
#     )
#     gem.get_gem(save_path = '../pfA6w.csv')
