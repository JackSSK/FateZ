#!/usr/bin/env python3
"""
Objects

author: jjy
"""
import pandas as pd
import pyranges as pr
import scanpy as sc
import anndata as ad
import fatez.tool.gff as gff

class Process():
    """
    Preprocessing the scRNA-seq and scATAC-seq data to get
    linkages between peaks and genes.
    Criteria: (1) Peak region overlap Gene region
    """
    def __int__(self,
        rna_path:str = None,
        atac_path:str = None,
        gff_path:str = None
        ):
        self.rna_path = rna_path
        self.atac_path = atac_path
        self.gff_path = gff_path
        self.rna_mt = list()
        self.peak_region_list = list()
        self.gene_region_list = list()

    """
    :param rna_path: <class fatez.lib.preprocess.preprocess>
    	The object of propective regulatory source gene.

    :param atac_path: <class fatez.lib.preprocess.preprocess>
    	The object of propective regulatory target gene.
    """
    def load_data(self):
        sc.settings.verbosity = 3
        sc.logging.print_header()
        sc.settings.set_figure_params(dpi=80, facecolor='white')
        self.rna_mt = sc.read_10x_mtx(
            self.rna_path,
            var_names = 'gene_symbols',
            cache = True
        )

        gff = gff.Reader(gff_path)
        gff_template = gff.get_genes_gencode(id='GRCm38_template')

        symbol_list = []
        gene_chr_list = []
        gene_start_list = []
        gene_end_list = []
        for i in list(mm10_template.genes.keys()):

            symbol_list.append(gff_template.genes[i].symbol)
            gene_chr_list.append(gff_template.genes[i].chr)
            gene_start_list.append(gff_template.genes[i].start_pos)
            gene_end_list.append(gff_template.genes[i].end_pos)
            
        if self.rna_mt[0][0:3] == 'ENS':
            row_name_list = gff_template.genes.keys()
        else:
            row_name_list = symbol_list

        atac_h5ad = ad.read(self.atac_path)

        peak_names = list(atac_h5ad.var_names)
        chr_list = list()
        start_list = list()
        end_list = list()
        ### extract chromosome start and end
        for i in peak_names:
            peak = i.split('_')
            chr_list.append(peak[0])
            start_list.append(peak[1])
            end_list.append(peak[2])

        self.peak_region_list = list(chr_list,start_list,end_list)
        self.gene_region_list = list(gene_chr_list, gene_start_list, gene_end_list)

