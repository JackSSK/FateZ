#!/usr/bin/env python3
"""
Objects

author: jy
"""

import copy
import pandas as pd
import networkx as nx
from fatez.lib import GRN_Basic
import fatez.tool.JSON as JSON



class CRE(GRN_Basic):
	"""
	Class to store information of a cis-regulatory element (CRE).
	"""
	def __init__(self,
		chr:str = None,
		position:list = [0, 0],
		strand:str = '+',
		peaks:float = 0.0,
		**kwargs,
		):
		"""
		:param chr: <str Default = None>
			Chromosomal location of the CRE

		:param position: <list Default = [0.0]>
			Genomic position of the CRE.
			[Start position, End position]

		:param strand: <str Default = '+'>
			Which strand the CRE is located.

		:param peaks: <float Default = None>
			Peak-calling result of the CRE.
		"""
		super(CRE, self).__init__()
		self.id = chr + ':' + str(position[0]) + '-' + str(position[1])
		self.chr = chr
		self.position = position
		self.strand = strand
		self.peaks = peaks
		# if there are other args
		for key in kwargs: setattr(self, key, kwargs[key])



class Gene(GRN_Basic):
	"""
	Class to store information of a Gene.
	"""
	def __init__(self,
		id:str = None,
		# symbol:str = None,
		# type:str = 'Gene',
		# gff_coordinates:list = list(),
		rna_exp:float = 0.0,
		peaks:float = 0.0,
		promoter_peaks:float = 0.0,
		# cre_regions:list = list(),
		**kwargs
		):
		"""
		:param id: <str Default = None>
			Ensembl ID.

		:param symbol: <str Default = None>
			Gene Symbol.

		:param type: <str Default = 'Gene'>
			If the gene is known to be Transcription Factor, change it to 'TF'.

		:param gff_coordinates: <list Default = None>
			The locations of corresponding record in the refernece GFF file.

		:param rna_exp: <float Default = 0.0>
			Transcriptomic expression of gene.

		:param peaks: <float Default = 0.0>
			Peak-calling result of the gene.

		:param promoter_peaks: <float Default = 0.0>
			Peak-calling result of the promoter region of gene.

		:param cre_regions: <list[CRE_IDS] Default = Empty List>
			List of CREs interacting with the gene.
			Note: Based on database like 4DGenome.
		"""
		super(Gene, self).__init__()
		self.id = id
		# self.symbol = symbol
		# self.type = type
		# self.gff_coordinates = gff_coordinates
		self.rna_exp = rna_exp
		self.peaks = peaks
		self.promoter_peaks = promoter_peaks
		# self.cre_regions = cre_regions
		# if there are other args
		for key in kwargs: setattr(self, key, kwargs[key])



class GRP(GRN_Basic):
	"""
	Class to store information of a Gene Regulatory Pathway(GRP).
	"""
	def __init__(self,
		reg_source:Gene = Gene(),
		reg_target:Gene = Gene(),
		reversable:bool = False,
		exp_cor:float = 0.0,
		motif_enrich:float = 0.0,
		**kwargs
		):
		"""
		:param reg_source: <class fatez.lib.grn.Gene>
			The object of propective regulatory source gene.

		:param reg_target: <class fatez.lib.grn.Gene>
			The object of propective regulatory target gene.

		:param reversable: <bool Default = False>
			Whether the target gene can also regulate expression of source gene.
			Note: The reverse GRP is expected to be a seperate object!

		:param exp_cor: <float Default = 0.0>
			The expression correlation between source gene and target gene.

		:param motif_enrich: <float Default = 0.0>
			Enrichment of source gene's motifs to regulate target gene.
		"""
		super(GRP, self).__init__()
		if type(reg_source) == str and type(reg_target) == str:
			self.id = reg_source + '_' + reg_target
		else:
			self.id = reg_source.id + '_' + reg_target.id
		self.reg_source = reg_source
		self.reg_target = reg_target
		self.reversable = reversable
		self.exp_cor = exp_cor
		self.motif_enrich = motif_enrich
		# if there are other args
		for key in kwargs: setattr(self, key, kwargs[key])

	def as_dict(self):
		"""
		Cast object into dict type.

		:return: <class 'dict'>
		"""
		answer = copy.deepcopy(self.__dict__)
		answer['reg_source'] = self.reg_source.id
		answer['reg_target'] = self.reg_target.id
		return answer



class GRN(GRN_Basic):
	"""
	Class for representing Gene Regulatory Network(GRN).

	Attributes:
		:self.genes: Dictionary of Gene() objects in the GRN
		:self.grps: Dictionary of GRP() objects in the GRN
	"""
	def __init__(self, id:str = None, **kwargs):
		"""
		:param id: <str Default = None>
			The ID for GRN object.
		"""
		super(GRN, self).__init__()
		self.id = id
		self.genes = dict()
		self.grps = dict()
		# if there are other args
		for key in kwargs: setattr(self, key, kwargs[key])

	def add_gene(self, gene:Gene = None):
		"""
		Add a new gene into GRN.

		:param gene: <class fatez.lib.grn.Gene>
			The Gene object to add.
		"""
		assert gene.id not in self.genes
		self.genes[gene.id] = gene
		return

	def add_grp(self, grp:GRP = None):
		"""
		Add a new GRP into GRN.

		:param grp: <class fatez.lib.grn.GRP>
			The GRP object to add.
		"""
		assert grp.id not in self.grps
		self.grps[grp.id] = grp
		return

	def as_dict(self):
		"""
		Cast object into dict type.

		:return: <class 'dict'>
		"""
		answer = {
			'genes': {id:record.as_dict() for id, record in self.genes.items()},
			'grps': {id:record.as_dict() for id, record in self.grps.items()},
		}
		for key in self.__dict__:
			if key == 'genes' or key == 'grps':
				continue
			elif key == 'cres':
				answer[key] = {
					id:record.as_dict() for id, record in self.cres.items()
				}
			else:
				answer[key] = self.__dict__[key]
		return answer

	def as_digraph(self):
		"""
		Cast object into networkx DiGraph type.

		:return: <class 'networkx.classes.digraph.DiGraph'>
		"""
		graph = nx.DiGraph()
		for grp_id in self.grps.keys():
			source = self.grps[grp_id].reg_source
			target = self.grps[grp_id].reg_target

			# add regulatory source and target genes to nodes
			if not graph.has_node(source.id):
				graph.add_node(source.id, **source.as_dict())
			if not graph.has_node(target.id):
				graph.add_node(target.id, **target.as_dict())

			# add GRP as an edge
			graph.add_edge(source.id, target.id, **self.grps[grp_id].as_dict())
		return graph

	def save(self, save_path:str = None, indent:int = 4):
		"""
		Save current GRN into a JSON file.

		:param save_path: <str Default = 'out.js'>
	        Path to save the output file.

	    :param indent: <int Default = 4>
	        Length of each indent.
		"""
		JSON.encode(data=self.as_dict(), save_path=save_path, indent=indent)
		return

	def load(self, load_path:str = None):
		"""
		Load GRN from a given JSON file.

		:param save_path: <str Default = 'out.js'>
	        Path to save the output file.

	    :param indent: <int Default = 4>
	        Length of each indent.
		"""
		data = JSON.decode(path = load_path)
		self.id = data['id']
		# Load genes first
		for id, rec in data['genes'].items():
			self.genes[id] = Gene(**rec)
		# Then load GRPs
		for id, rec in data['grps'].items():
			self.grps[id] = GRP(**rec)
			# Change reg source and target back to objects
			self.grps[id].reg_source = self.genes[self.grps[id].reg_source]
			self.grps[id].reg_target = self.genes[self.grps[id].reg_target]
		# Check CREs
		if 'cres' in data:
			self.cres = dict()
			for id, rec in data['cres'].items():
				self.cres[id] = CRE(**rec)
		# Check genetic regions
		if 'regions' in data:
			self.regions = dict()
			for chr, recs in data['regions'].items():
				self.regions[chr] = [x for x in recs]
		# Load other attrs if there is any
		for k in data:
			if k not in ['id','genes','grps','cres','regions']:
				setattr(self, k, data[k])
		return
