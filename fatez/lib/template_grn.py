#!/usr/bin/env python3
"""
This file contains objects and functions to make template.

ToDo: load and associate CREs with Genes in template

author: jy
"""
import fatez.lib.grn as grn
import fatez.tool.gff as gff
from pkg_resources import resource_filename


class Template_GRN(grn.GRN):
    """
    Object for template GRN, which would made by setting:
        1. Default gene set from GFF
        2. Default CRE set...Developing...
    """

    def load_genes(self, gff_path, **kwargs):
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
        gff_obj.close()
        return

    def load_cres(self, path:str = None):
        self.cres = dict()
        return

    def get_gff(self):
        """
        Read in GFF file based on pre-set path.
        """
        return gff.Reader(self.ref_gff)

    def get_genetic_regions(self):
        """
        Make a dictionary storing locations for every genetic feature.
        """
        self.regions = dict()
        self._populate_regions(rec_dict = self.genes, type = 'gene')
        if hasattr(self, 'cres'):
            self._populate_regions(rec_dict = self.cres, type = 'cre')
        # Sort regions by start position
        for k,v in self.regions.items(): v.sort(key = lambda x:x['pos'][0])
        return

    def _populate_regions(self, rec_dict, type):
        """
        Populate the region dict with gene records or CRE records.
        """
        for id, rec in rec_dict.items():
            data = {'pos':(rec.start_pos, rec.end_pos), 'id':id, 'type':type}
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
    template.get_genetic_regions()
    template.save(save_path = out_path)
