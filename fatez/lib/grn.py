#!/usr/bin/env python3
"""
Objects

author: jy
"""

import copy
import pandas as pd
import networkx as nx
import os
import warnings
import numpy as np
import anndata as ad, hdf5plugin
from scipy.sparse import csr_matrix
from fatez.process._harmonizer import GEX_Harmonizer
from fatez.lib import GRN_Basic
import fatez.tool.JSON as JSON

def make_cluster_map(folder_path):
    data_dict = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(
                file_path,
                sep=' ',
                header=None,
                names=['cluster_id', 'cell_id']
            )

            prefix = os.path.basename(file_path).replace('.txt', '.')

            for i, row in df.iterrows():
                if row['cell_id'] not in data_dict:
                    data_dict[row['cell_id']] = prefix + str(row['cluster_id'])
                else:
                    raise Exception('WTF')
            print('Processed: ', file_name)
        else:
            print('Skipped: ', file_name)
    return data_dict

def make_grn_dict(folder_path):
    data_dict = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            key = os.path.splitext(file_name)[0]

            df = pd.read_csv(file_path, index_col=0)
            csr = csr_matrix(
                df.astype(pd.SparseDtype("float64",0)).sparse.to_coo()
            )

            # Record data
            data_dict[key] = {
                'data': csr.data.tolist(),
                'indices': csr.nonzero()[0].tolist(),
                'indptr': csr.nonzero()[1].tolist(),
                'ind_name': df.index.tolist(),
                'col_name': df.columns.tolist(),
            }

            print('Processed:', key)
        else:
            print('Skipped:', file_name)

    return data_dict

def load_grn_dict(
        filepath:str = None,
        cache_path:str = None,
        load_cache:bool = False,
        backed:bool = True,
    ):

    # Init
    answer = dict()
    if cache_path is not None:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        # Check if cache is empty
        if len(os.listdir(cache_path)) == 0 and load_cache:
            warnings.warn('Cache is empty, switching to load_cache = False')
            load_cache = False
    elif load_cache and cache_path is None:
        warnings.warn('Null cache_path, switching to load_cache = False')
        load_cache = False

    # Load from original file
    if not load_cache:
        for key,rec in JSON.decode(filepath).items():
            adata = ad.AnnData(
                X = csr_matrix(
                    (rec['data'], (rec['indices'], rec['indptr'])),
                    shape = (len(rec['ind_name']), len(rec['col_name']))
                ),
                obs = pd.DataFrame(index = rec['ind_name']),
                var = pd.DataFrame(index = rec['col_name']),
            )

            # Backed in cache path
            if cache_path is not None:
                cache_filename = os.path.join(cache_path, key+'.h5ad')
                with open(cache_filename, 'w') as f:
                    adata.write_h5ad(
                        cache_filename,
                        compression = hdf5plugin.FILTERS["zstd"],
                    )
                answer[key] = ad.read_h5ad(cache_filename, backed = backed)
            else:
                # Add the rec
                answer[key] = adata

    # Load from cache
    else:
        for file in os.listdir(cache_path):
            # Only load h5ad files
            if file.endswith(".h5ad"):
                answer[os.path.splitext(file)[0]] = ad.read_h5ad(
                    os.path.join(cache_path, file),
                    backed = backed
                )
    return answer

def get_uni_grn(
        tf_list:pd.DataFrame, 
        genes_dict:dict, 
        grn_dict:dict, 
        missed_genes:pd.DataFrame, 
        valid_genes:pd.Index,
        cache_path:str = None,
        ):
   
    """
    """
    def __get_ens_id(gene_list,):
        """
        """
        answer_dict = dict()
        for gene in gene_list:
            # Successfully found gene rec in genes_dict
            if gene in genes_dict['name']:
                ens_id = genes_dict['name'][gene]
                for ele in ens_id:
                    if ele not in answer_dict:
                        answer_dict[ele] = None
            # Expand search in missed_genes
            elif gene in missed_genes.index:
                print('Found in missed_genes: ', gene)
                ens_id = missed_genes.loc[gene].values[0]
                if str(type(ens_id)) == "<class 'str'>": 
                    if ens_id in genes_dict['ens_id']:
                        answer_dict[ens_id] = None
                    # Skip the pseudogenes, predicted_genes, lncRNA, etc.
                    else:
                        print('Not recorded in gene_dict: ', ens_id)
                # Multiple ENS IDs
                elif str(type(ens_id)) == "<class 'numpy.ndarray'>":
                    for ele in missed_genes.loc[gene].values:
                        ens_id = ele[0]
                        if ens_id[0] in genes_dict['ens_id']:
                            answer_dict[ens_id[0]] = None
                        # Skip the pseudogenes, predicted_genes, lncRNA, etc.
                        else:
                            print('Not recorded in gene_dict: ', ens_id)
            # No can do
            else:
                warnings.warn(f'Not found even in missed_genes: {gene}')
        return answer_dict
    
    # Init
    if cache_path is not None and not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # Transfer valid_genes to dict for faster search
    valid_genes = {ele:None for ele in valid_genes}

    # Need to get valid TFs based on which genes are in the corpus
    tf_ens_ids = {
        ele:None for ele in __get_ens_id(tf_list.index,) if ele in valid_genes
    }
    
    # Get all the genes involved in the grn_dict
    grn_gene_names = np.array([])
    for k,v in grn_dict.items():
        grn_gene_names = np.union1d(grn_gene_names, v.var.index.values)

    # Transfer gene names to Ensemble IDs
    gene_ens_ids = {
        ele:None for ele in __get_ens_id(grn_gene_names,) if ele in valid_genes
    }

    # Unify the GRNs
    for k,v in grn_dict.items():
        # Harmonize the GRN with TF first then genes
        v = GEX_Harmonizer(
            adata = GEX_Harmonizer(
                adata = v.to_memory().T,
                name_type = 'NAME',
            ).Process(
                order = tf_ens_ids,
                map_dict = genes_dict['name'],
            ).T,
            name_type = 'NAME',
        ).Process(
            order = gene_ens_ids,
            map_dict = genes_dict['name'],
        )
        assert v.X.nnz>0
        
        # Update cache to unified GRN
        if cache_path is not None:
            cache_filename = os.path.join(cache_path, k + '.h5ad')
            v.write_h5ad(
                cache_filename,
                compression = hdf5plugin.FILTERS["zstd"],
            )
            grn_dict[k] = ad.read_h5ad(cache_filename, backed = True)
        else:
            grn_dict[k] = v
        print('Unified: ', k, grn_dict[k])

    return grn_dict



def Reverse_Peaks_Ann(annotations):
    """
    Make annotation dict with gene IDs as keys.

    :param annotations:<dict Default = None>
        The annotation dict returned from Reconstruct.annotate_peaks()
    """
    answer = dict()
    for peak, rec in annotations.items():
        if rec is None: continue

        id = rec['id']
        if id not in answer:
            answer[id] = dict()

        if rec['cre']:
            answer[id][peak] = {'overlap':rec['cre']}
        else:
            answer[id][peak]={'overlap':rec['gene'], 'promoter':rec['promoter']}
    return answer



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

# def test_grn():
#     toy = GRN(id = 'toy')
#     # fake genes
#     gene_list = [
#         Gene(id = 'a', symbol = 'AAA'),
#         Gene(id = 'b', symbol = 'BBB'),
#         Gene(id = 'c', symbol = 'CCC'),
#     ]
#     # fake grps
#     grp_list = [
#         GRP(reg_source = gene_list[0], reg_target = gene_list[1]),
#         GRP(reg_source = gene_list[0], reg_target = gene_list[2]),
#         GRP(reg_source = gene_list[1], reg_target = gene_list[2]),
#     ]
#     # populate toy GRN
#     for gene in gene_list:
#         toy.add_gene(gene)
#     for grp in grp_list:
#         toy.add_grp(grp)
#
#     toy.as_dict()
#     toy.as_digraph()
#     toy.save('../data/toy.grn.js.gz')
#     # Load new GRN
#     del toy
#     toy_new = GRN()
#     toy_new.load('../data/toy.grn.js.gz')
#     for id, rec in toy_new.grps.items():
#         print(rec.reg_source.symbol)
