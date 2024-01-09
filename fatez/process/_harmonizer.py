#!/usr/bin/env python3
"""
Just testing how to readin files without extrations.

author: jy
"""
import re
import gzip
import h5py
import scipy
import tarfile
from functools import lru_cache, cache
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, lil_matrix, hstack
import fatez.tool.JSON as JSON



def Stratify_Common(genes, peaks):
	"""
	EZPZ
	"""
	overlap = list(set(genes.obs_names).intersection(set(peaks.obs_names)))
	return genes[overlap, :], peaks[overlap, :]



class Peak_Harmonizer(object):
	"""
	Harmonize peak count AnnData objects.
	"""

	def __init__(self, adata = None,):
		super(Peak_Harmonizer, self).__init__()
		self.adata = adata

	def Process(self, peaks_db, chr_ind_dict, return_overlap_dict:bool = False):
		# Make an ID map based on which peaks are overlapping
		overlap_dict = self.Mapping_Peaks(
			query_data = self.adata.var.index,
			peaks_db = peaks_db,
			chr_ind_dict = chr_ind_dict
		)
		print('Done Mapping')

		result = list()
		for idx, peak in enumerate(peaks_db.keys()):
			if peak in overlap_dict and len(overlap_dict[peak]) >= 1:
				record = sum(self.adata[:, p].X for p in overlap_dict[peak])
			else:
				record = csr_matrix(
					(self.adata.X.shape[0], 1),
					dtype = self.adata.X.dtype
				)
			result.append(record)

		# Make result actually adata
		result = ad.AnnData(hstack(result, format='csr'))
		result.obs.index = self.adata.obs.index
		result.var.index = peaks_db.keys()

		if return_overlap_dict:
			return result, overlap_dict
		else:
			return result

	def Mapping_Peaks(self, query_data, peaks_db, chr_ind_dict, sorted_db=True):
		"""
		Mapping peaks to references by seeing whether are there overlaps
		"""

		def __orphan_scaffold_name_solver(chr_id, ref_db):
			"""
			Solve some weird orphan scaffold nomenclature stuffs
			"""
			if chr_id in ref_db:
				return chr_id
			elif re.search(r'chrUn_', chr_id) or re.search(r'_random', chr_id):
				answer = chr_id.split('_')[1] + '.1'
				if answer in ref_db: return answer
				elif answer[:-1] + '2' in ref_db: return answer[:-1] + '2'
				elif answer[:-1] + '3' in ref_db: return answer[:-1] + '3'
				else: return None
			else:
				return None

		answer = dict()
		overlap_dict = dict()
		for query_peak_id in query_data:
			query_peak = self.Peak_Parser(query_peak_id)

			# Weird query peak
			if query_peak is None:
				raise Exception(f'Weird peak format{query_peak_id}')

			# Initialize
			query_chr = query_peak[0]
			overlap_dict[query_peak_id] = []

			# Fix chr nomenclature if dealing with a weird one
			if query_chr not in chr_ind_dict:
				query_chr = __orphan_scaffold_name_solver(query_chr, chr_ind_dict)
				if query_chr is None:
					# raise Exception(f'Weird peak format {query_peak_id}')
					continue

			# Searching through peaks
			ind_range = chr_ind_dict[query_chr]
			for peak_id in list(peaks_db.keys())[ind_range[0]-1:ind_range[1]]:
				peak_info = self.Peak_Parser(peak_id)

				# Since the ref db should be fully sorted based on start pos
				if sorted_db and peak_info[1] >= query_peak[2]: break

				if peak_info and self.Are_Peaks_Overlap(query_peak, peak_info):
					overlap_dict[query_peak_id].append(peak_id)

		# Reversal the dict that makes values being keys
		for key, values in overlap_dict.items():
			for peak in values:
				if peak not in answer: answer[peak] = list()
				answer[peak].append(key)

		return answer

	@lru_cache(maxsize = 100000)
	def Peak_Parser(self, peak_id):
		"""
		Parses the peak to extract the chromosome, start, and end positions.
		:param peak_id: A string identifier of the peak
		:return: Tuple of chromosome, start position, and end position
		"""
		try:
			chrom, positions = peak_id.split(':')
			start, end = map(int, positions.split('-'))
			return chrom, start, end
		except Exception as e:
			print(f"Error parsing peak: {peak_id}. Error: {e}")
			return None

	def Are_Peaks_Overlap(self, peak1, peak2):
		"""
		Checks if two peaks overlap.
		:param peak1: Tuple of (chromosome, start, end) for the first peak
		:param peak2: Tuple of (chromosome, start, end) for the second peak
		:return: Boolean indicating whether the peaks overlap
		"""
		chr1, start1, end1 = peak1
		chr2, start2, end2 = peak2
		return chr1 == chr2 and max(start1, start2) <= min(end1, end2)



class GEX_Harmonizer(object):
	"""
	Harmonize gene expression AnnData objects.
	"""
	def __init__(self, adata = None, name_type:str=None, **kwargs):
		super(GEX_Harmonizer, self).__init__()
		self.adata = adata
		# Auto sign name_type if not specified
		if name_type is None and adata is not None:
			self.name_type = self.GeneName_Or_ENSID(adata.var.index)
		else:
			self.name_type = name_type

	def Process(self, order, map_dict:dict=None, ):
		# Map each var from the original index to its position
		if self.name_type == 'NAME':
			index_map = dict()
			for i, gene in enumerate(self.adata.var_names):
				if gene in map_dict:
					for id in map_dict[gene]: index_map[id] = i
				else:
					continue
		elif self.name_type == 'ENS':
			index_map = {v: i for i, v in enumerate(self.adata.var_names)}
		else:
			raise Exception('What the Fake is this')

		# Create a zero-filled CSR matrix with the desired shape
		result = lil_matrix((len(self.adata.obs_names), len(order)))

		# Update the zero-filled matrix with the data from the CSR matrix
		for idx, label in enumerate(order):
			original_position = index_map.get(label)
			if original_position is not None:
				result[:, idx] = self.adata[:, original_position].X
		# Make result actually adata
		result = ad.AnnData(result.tocsr())

		# # Should work better, not tested, so commented out for now.
		# result = list()
		# for idx, peak in enumerate(order):
		# 	original_position = index_map.get(label)
		# 	if original_position is not None:
		# 		# Polulate with corresponding data
		# 		record = self.adata[:, original_position].X
		# 	else:
		# 		# Inject an empty sparse matrix to hold place
		# 		record = csr_matrix(
		# 			(self.adata.X.shape[0], 1),
		# 			dtype = self.adata.X.dtype
		# 		)
		# 	result.append(record)
		#
		# # Make result actually adata
		# result = ad.AnnData(hstack(result, format='csr'))

		result.obs.index = self.adata.obs.index
		result.var.index = order
		return result

	def GeneName_Or_ENSID(self, vars):
		"""
		Check Gene Exp Var nomenclature
		"""
		count = sum(1 for i in vars if re.compile(r'^EN[A-Z]+\d+$').match(i))
		if count == len(vars):
			return 'ENS'
		elif count <= 10:
			return 'NAME'
		else:
			raise Exception('What the Fake is this')
