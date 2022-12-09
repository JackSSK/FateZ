#!/usr/bin/env python3
"""
Objects

author: jjy
"""
import pandas as pd
import pyranges as pr
import scanpy as sc
import anndata as ad
import numpy as np

class Process():
    """
    Preprocessing the scRNA-seq and scATAC-seq data to get
    linkages between peaks and genes.
    Criteria: (1) Peak region overlap Gene region
    """
    def __int__(self,
        rna_path:str = None,
        atac_path:str = None,
        ):
        self.rna_path = rna_path
        self.atac_path = atac_path
        self.rna_mt = list()
        self.peak_dict = dict()
        self.peak_pr = list()

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

        ### create pyranges object
        self.peak_dict = {
            'Chromosome': chr_list,
            'Start': start_list,
            'End': end_list,
        }
        self.peak_pr = pr.from_dict(peak_dict)
