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
from pkg_resources import resource_filename
import numpy as np
import pandas as pd
from Bio import motifs
import fatez.tool.gff as gff
import fatez.tool.transfac as transfac

class Preprocess():
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
        sc.pp.filter_cells(self.rna_mt, min_genes=200)
        sc.pp.filter_genes(self.rna_mt, min_cells=3)
        gff = gff.Reader(self.gff_path)
        gff_template = gff.get_genes_gencode(id='GRCm38_template')

        symbol_list = []
        gene_chr_list = []
        gene_start_list = []
        gene_end_list = []
        for i in list(gff_template.genes.keys()):

            symbol_list.append(gff_template.genes[i].symbol)
            gene_chr_list.append(gff_template.genes[i].chr)
            gene_start_list.append(gff_template.genes[i].position[0])
            gene_end_list.append(gff_template.genes[i].position[1])

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

        self.peak_region_df = pd.DataFrame({'chr':chr_list,'start':start_list,'end':end_list},index=peak_names)
        self.gene_region_df = pd.DataFrame({'chr':gene_chr_list, 'start':gene_start_list,
                                                                        'end':gene_end_list},index=row_name_list)


    def make_pseudo_networks(self,
        network_cell_size:int = 10,
        data_type:str = None,
        network_number:int = 10
        ):
        ### sample cells

        for i in range(network_number):
            ### sample cells
            if data_type == 'paired':
                rna_cell_use = self.rna_mt.obs_names[random.sample(range(len(self.rna_mt.obs_names)),
                                                                   network_cell_size)]
                atac_cell_use = rna_cell_use

            if data_type == 'unpaired':
                rna_cell_use = self.rna_mt.obs_names[random.sample(range(len(self.rna_mt.obs_names)),
                                                                   network_cell_size)]
                atac_cell_use = self.atac_mt.obs_names[random.sample(range(len(self.atac_mt.obs_names)),
                                                           network_cell_size)]
            rna_pseudo_netowrk = self.rna_mt[rna_cell_use]
            atac_pseudo_network = self.atac_mt[atac_cell_use]
            self.pseudo_network.append([rna_pseudo_netowrk,atac_pseudo_network])





    def find_linkages(self, overlap_size=250, cor_thr = 0.6):

        ### find overlap
        gene_chr_type = list(set())
        gene_overlapped_peaks = {}
        gene_related_peak = []
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
                peak_cor = []
                for j in peak_df_use.index:
                    ### overlap
                    peak_start = int(peak_df_use.loc[j]['start'])
                    peak_end = int(peak_df_use.loc[j]['end'])

                    if self.__is_overlapping(gene_start,gene_start+overlap_size,
                                     peak_start,peak_end)):
                    ### calculate correlation between gene count and peak count
                        rna_count = rna_network[:,row].X.todense()
                        atac_count = atac_network[:,j].X.todense()

                        pg_cor = stats.pearsonr(rna_count.transpose().A[0],
                                 atac_count.transpose().A[0])
                        if  pg_cor > cor_thr:
                            peak_overlap.append(j)
                            peak_cor.append(pg_cor)

                pg_cor_se = pd.Series(peak_cor,index=peak_overlap)
                ### select peak with the largest correlation

                if len(pg_cor_se) >1:
                    pg_cor_se = pg_cor_se[pg_cor_se==max(pg_cor_se)]

                gene_related_peak.append(pg_cor_se.index[0])

        gene_related_peak_region = self.peak_region_df.loc[gene_related_peak]
        gene_related_peak_region.index = self.gene_region_df.index
        ### modify the self.gene_region_df to let it contain gene related peak
        self.gene_region_df = pd.concat([self.gene_region_df,gene_related_peak_region],
                                axis=1,join='outer')




        ###
    def find_motifs_binding(self, specie:str = 'mouse', region_use, gene,peak):
        ### load tf motif relationships
        path = resource_filename(
            __name__, '../data/' + specie +'/Transfac201803_MotifTFsF.txt.gz'
        )
        ### make TFs motifs dict
        tf_motifs = transfac.Reader(path = path).get_tfs()
        motifs_use = tf_motifs[gene]['motif']

        # motif_db = pd.read_table(path)
        # TF_motif_dict = {}
        # for i in motif_db.index:
        #     TFs = motif_db.iloc[i, :][3]
        #     TF_list = TFs.split(';')
        #     Motif = motif_db.iloc[i, :][0]
        #     for i  in TF_list:
        #         if i in TF_motif_dict.keys():
        #             TF_motif_dict[i].append(Motif)
        #         else:
        #             TF_motif_dict[i] = [Motif]

        ### load TRANSFAC PWM
        pwm_path = resource_filename(
            __name__, '../data/' + specie +'/Transfac_PWM.txt'
        )
        handle = open(pwm_path)
        record = motifs.parse(handle, "TRANSFAC")
        handle.close()
        score_all = 0
        ### motif discovering
        for i in motifs_use:
            score_dict[i] = []
            motif_use_name = i[1:5]
            motif = record[int(motif_use_name)]
            pwm = motif.counts.normalize()
            pssm = pwm.log_odds()
            for position, score in pssm.search(peak, threshold=3.0):
                score_all += score
        return [gene,score_all]

    def  __generate_grp(self):
        ### grp
        df_gene = pd.DataFrame(np.zeros((len(self.rna_mt.var_names), len(self.rna_mt.var_names))))
        df_gene.columns = self.rna_mt.var_names
        df_gene.index = self.rna_mt.var_names
        gene_melt = pd.melt(df_gene)
        rownames = pd.Series(list(self.rna_mt.var_names)*len(self.rna_mt.var_names))
        gene_melt = pd.concat([rownames,gene_melt ],axis=1,join='outer')
        gene_melt = gene_melt.iloc[:,[0,1]]
        gene_melt.columns = ['source','target']
        gene_melt.iloc[:,[0]].tolist()
        ### This part needed to be fixed
    def __is_overlapping(self,x1, x2, y1, y2):
        return max(x1, y1) <= min(x2, y2)
