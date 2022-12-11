#!/usr/bin/env python3
"""
Objects

Note: Try to keep line length within 81 Chars

author: jjy
"""
import pandas as pd
import random
import scanpy as sc
import anndata as ad
from scipy import stats
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
        self.atac_mt = list()
        self.peak_region_df = pd.DataFrame()
        self.gene_region_df = pd.DataFrame()
        self.pseudo_network = list()

    """
    :param rna_path: <class fatez.lib.preprocess.preprocess>
    	The object of prospective regulatory source gene.

    :param atac_path: <class fatez.lib.preprocess.preprocess>
    	The object of prospective regulatory target gene.
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

        self.atac_h5ad = ad.read(self.atac_path)

        peak_names = list(self.atac_h5ad.var_names)
        chr_list = list()
        start_list = list()
        end_list = list()
        ### extract chromosome start and end
        for i in peak_names:
            peak = i.split('_')
            chr_list.append(peak[0])
            start_list.append(peak[1])
            end_list.append(peak[2])

        self.peak_region_df = pd.DataFrame(
            {'chr' = chr_list,'start' = start_list,'end' = end_list},
            index = peak_names
        )
        self.gene_region_df = pd.DataFrame(
            {
                'chr' = gene_chr_list,
                'start' = gene_start_list,
                'end' = gene_end_list
            },
            index=row_name_list
        )


    def make_pseudo_networks(self,
        network_cell_size:int = 10,
        data_type:str = None,
        network_number:int = 10
        ):
        ### sample cells
        for i in range(network_number):
            if data_type == 'paired':
                rna_cell_use = self.rna_mt.obs_names[
                    random.sample(
                        range(len(self.rna_mt.obs_names)),
                        network_size
                    )
                ]
                atac_cell_use = rna_cell_use

            if data_type == 'unpaired':
                rna_cell_use = self.rna_mt.obs_names[
                    random.sample(
                        range(len(self.rna_mt.obs_names)),
                        network_size
                    )
                ]
                atac_cell_use = self.atac_mt.obs_names[
                    random.sample(
                        range(len(self.atac_mt.obs_names)),
                        network_size)]
            self.pseudo_network.append()



    def find_linkages(self, overlap_size=250, cor_thr = 0.6):
        ### find overlap
        gene_chr_type = list(set())
        gene_overlapped_peaks = {}
        for i in gene_chr_type:
            ### match chr
            gene_df_use = self.gene_region_df[self.gene_region_df['chr'] == i]
            peak_df_use = self.self.gene_region_df[self.gene_region_df['chr'] == i]

            for row in self.gene_region_df.index:
                ### narrow the range
                gene_start = int(gene_df_use.loc[row]['start'])
                peak_df_use = peak_df_use[int(peak_df_use['end'])<gene_start]
                peak_df_use = peak_df_use[int(peak_df_use['start']) < gene_start+overlap_size]

                peak_overlap = []
                for j in peak_df_use.index:
                    ### overlap
                    peak_start = int(peak_df_use.loc[j]['start'])
                    peak_end = int(peak_df_use.loc[j]['end'])

                    if self.__is_overlapping(gene_start,gene_start+overlap_size,
                                     peak_start,peak_end)):
                    ### calculate correlation between gene count and peak count
                        for network in self.pseudo_network:

                            if stats.pearsonr(, ) > cor_thr:
                                peak_overlap.append(True)




                gene_overlapped_peaks[self.gene_region_df.index] = peak_df_use.index[peak_overlap]





        ###
    def find_motif_enrichment(self,)



    def __is_overlapping(self,x1, x2, y1, y2):
        return max(x1, y1) <= min(x2, y2)
