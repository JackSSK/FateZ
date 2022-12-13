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
        promoter_upstream:int = 250,
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
        self._add_promoter_info(promoter_upstream = promoter_upstream)
        gff_obj.close()
        return

    def _add_promoter_info(self, promoter_upstream:int = 250,):
        for k, v in self.genes.items():
            # For genes on pos strand
            if v.strand == '+':
                v.promoter=(max(v.start_pos-promoter_upstream, 0), v.start_pos)
            # For genes on negative strand
            elif v.strand == '-':
                v.promoter = (v.end_pos, v.end_pos + promoter_upstream)
        return

    def load_cres(self, path:str = None):
        self.cres = dict()
        return

    def get_gff(self):
        """
        Read in GFF file based on pre-set path.
        """
        return gff.Reader(self.ref_gff)

    def get_genetic_regions(self, promoter_upstream:int = 250):
        """
        Make a dictionary storing locations for every genetic feature.
        """
        self.regions = dict()
        self._get_gene_loc(self.genes, promoter_upstream)
        if hasattr(self, 'cres'): self._get_cre_loc(rec_dict = self.cres,)
        # Sort regions by start position
        for k,v in self.regions.items(): v.sort(key = lambda x:x['pos'][0])
        return

    def _get_gene_loc(self, rec_dict, promoter_upstream:int = 250,):
        """
        Populate the region dict with genes.
        """
        for id, rec in rec_dict.items():
            promoter=list(set([rec.start_pos, rec.end_pos]) & set(rec.promoter))
            assert len(promoter) == 1
            data = {
                'pos':(rec.start_pos, rec.end_pos),
                'promoter_pos':promoter[0],
                'id':id,
            }
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
            data = {
                'pos':(rec.start_pos, rec.end_pos), 'promoter_pos':-1, 'id':id,
            }
            if rec.chr not in self.regions:
                self.regions[rec.chr] = [data]
            else:
                self.regions[rec.chr].append(data)
        return




# Main function for testing purpose
if __name__ == '__main__':
    gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
    template_id = 'GRCm38_template'
    out_path = '../data/mouse/template.js.gz'
    gff_path = '../data/human/gencode.v42.basic.annotation.gff3.gz'
    template_id = 'GRCh38_template'
    out_path = '../data/human/template.js.gz'

    template = Template_GRN(id = template_id)
    template.load_genes(gff_path = gff_path)
    template.load_cres()
    template.get_genetic_regions(promoter_upstream = 250)
    template.save(save_path = out_path)
