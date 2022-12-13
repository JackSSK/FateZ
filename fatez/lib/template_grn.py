#!/usr/bin/env python3
"""
This file contains objects and functions to make template.

ToDo: load and associate CREs with Genes in template

author: jy
"""
import pandas as pd
import fatez.lib.grn as grn
import fatez.tool.gff as gff
from pkg_resources import resource_filename


class Template_GRN(grn.GRN):
    """
    Object for template GRN, which would made by setting:
        1. Default gene set from GFF
        2. Default CRE set...Developing...
    """

    def load_genes(self,
        gff_path:str = None,
        promoter_upstream:int = 1000,
        promoter_downstream:int = 500,
        **kwargs):
        """
        Load in genes from given GFF file.

        :param gff_path: <str Default = None>
			Path to the GFF file.

        :param kwargs:
			Key word arguments for fatez.tool.gff.Reader.get_genes_gencode()
        """
        self.ref_gff = gff_path
        gff_obj = self.get_gff()
        self.genes = gff_obj.get_genes_gencode(**kwargs)
        self._add_promoter_info(promoter_upstream, promoter_downstream)
        gff_obj.close()
        return

    def _add_promoter_info(self, upstream:int = 1000, downstream:int = 500):
        for k, v in self.genes.items():
            # For genes on pos strand
            if v.strand == '+':
                t = (max(v.position[0]-upstream, 0), v.position[0] + downstream)
            # For genes on negative strand
            elif v.strand == '-':
                t = (max(v.position[1]-downstream, 0), v.position[1] + upstream)
            v.promoter = t
        return

    def load_cres(self, path:str = None):
        self.cres = dict()
        return

    def get_gff(self):
        """
        Read in GFF file based on pre-set path.
        """
        return gff.Reader(self.ref_gff)

    def get_genetic_regions(self,):
        """
        Make a dictionary storing locations for every genetic feature.
        """
        self.regions = dict()
        self._get_gene_loc(self.genes,)
        if hasattr(self, 'cres'): self._get_cre_loc(rec_dict = self.cres,)
        # Sort regions by start position
        for k,v in self.regions.items(): v.sort(key = lambda x:x['pos'][0])
        return

    def _get_gene_loc(self, rec_dict,):
        """
        Populate the region dict with genes.
        """
        for id, rec in rec_dict.items():
            data = {'pos':rec.position, 'promoter':rec.promoter, 'id':id,}
            if rec.chr not in self.regions:
                self.regions[rec.chr] = [data]
            else:
                self.regions[rec.chr].append(data)
        return

    def _get_cre_loc(self, rec_dict):
        """
        Populate the region dict with CRE info.
        """
        for id, rec in rec_dict.items():
            data = {'pos':rec.position, 'promoter':False, 'id':id,}
            if rec.chr not in self.regions:
                self.regions[rec.chr] = [data]
            else:
                self.regions[rec.chr].append(data)
        return
