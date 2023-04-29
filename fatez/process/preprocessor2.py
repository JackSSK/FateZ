#!/usr/bin/env python3
"""
Objects

Note: Try to keep line length within 81 Chars

author: jjy
"""

import anndata
import pandas as pd
import random
import re
import scanpy as sc
import anndata as ad
from scipy import stats
from scipy.sparse import vstack
from pkg_resources import resource_filename
import numpy as np
import pandas as pd
import fatez.tool.gff as gff1
import fatez.tool.transfac as transfac
import fatez.lib.template_grn as tgrn
import fatez.process.grn_reconstructor as grn_recon
import warnings
from sklearn.preprocessing import MinMaxScaler



class Preprocessor():
    """
    Preprocessing the scRNA-seq and scATAC-seq data to get
    linkages between peaks and genes.
    Criteria: (1) Peak region overlap Gene region
    """
    def __init__(self,
        rna_path:str = None,
        atac_path:str = None,
        gff_path:str = None,
        data_type:str = None
        ):
        """
        :param rna_path: <class fatez.lib.preprocess.preprocess>
        	The object of prospective regulatory source gene.

        :param atac_path: <class fatez.lib.preprocess.preprocess>
        	The object of prospective regulatory target gene.
        """
        self.rna_path = rna_path
        self.atac_path = atac_path
        self.gff_path = gff_path
        self.data_type = data_type
        self.rna_mt = list()
        self.atac_mt = list()
        self.peak_annotations = dict()
        self.peak_region_df = pd.DataFrame()
        self.gene_region_df = pd.DataFrame()
        self.gff_gene = list()
    def load_data(self,
        matrix_format:str = '10x_paired',
        debug_mode:bool = False
        ):
        """
        Load in data

        ToDo:
            Init a DF directly, no need to use lists for chr and position maybe.
        """
        sc.settings.verbosity = 0
            # verbosity: errors (0), warnings (1), info (2), hints (3)
        if debug_mode:
            sc.settings.verbosity = 3
            sc.logging.print_header()
            sc.settings.set_figure_params(dpi=80, facecolor='white')

        chr_list = list()
        start_list = list()
        end_list = list()

        ### 10x paired data
        if matrix_format == '10x_paired':
            self.rna_mt = sc.read_10x_mtx(
                self.rna_path,
                var_names = 'gene_ids',
                cache = True
            )

            self.atac_mt = sc.read_10x_mtx(
                self.atac_path,
                var_names = 'gene_ids',
                cache = True,
                gex_only = False
            )

            self.atac_mt = self.atac_mt[
                :, len(self.rna_mt.var_names):(len(self.atac_mt.var_names) - 1)
            ]
            atac_array = self.atac_mt.X.toarray().T

        ### load unparied data or ???

        elif matrix_format == 'text_unpaired':
            self.rna_mt = sc.read_text(self.rna_path)
            self.atac_mt = sc.read_text(self.atac_path)
            atac_array = self.atac_mt.X.T
        elif matrix_format == 'paired':
            ###
            if self.rna_path[-1] != '/': self.rna_path += '/'
            mtx = anndata.read_mtx(self.rna_path + 'matrix.mtx.gz')
            gene = pd.read_table(self.rna_path + 'features.tsv.gz',
                                 header=None)
            barcode = pd.read_table(
                self.rna_path + 'barcodes.tsv.gz', header=None)
            mtx.obs_names = list(gene[0])
            mtx.var_names = list(barcode[0])
            self.rna_mt = mtx.T

            ###
            if self.atac_path[-1] !='/': self.atac_path += '/'
            mtx_atac = anndata.read_mtx(self.atac_path +'matrix.mtx.gz').T
            peak = pd.read_table(self.atac_path+'features.tsv.gz', header=None)
            barcode = pd.read_table(
                self.atac_path + 'barcodes.tsv.gz', header = None
            )
            mtx_atac.obs_names = list(barcode[0])
            mtx_atac.var_names = list(peak[0])
            self.atac_mt = mtx_atac
            atac_array = self.atac_mt.X.toarray().T


        # ### extract chromosome start and end
        # Assertion here?



        self.atac_mt.obs_names_make_unique()
        ###remove nan
        self.rna_mt = self.rna_mt[:,~self.rna_mt.var_names.isnull()]
        self.atac_mt = self.atac_mt[:,~self.atac_mt.var_names.isnull()]



    def rna_qc(self,
        rna_min_genes:int = 3,
        rna_min_cells:int = 200,
        rna_max_cells:int = 2000,
        rna_mt_counts:int = 5
        ):
        sc.pp.filter_cells(self.rna_mt, min_genes = rna_min_genes)
        sc.pp.filter_genes(self.rna_mt, min_cells = rna_min_cells)
        ###
        self.rna_mt.var['mt'] = self.rna_mt.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            self.rna_mt,
            qc_vars = ['mt'],
            percent_top = None,
            log1p = False,
            inplace = True
        )
        self.rna_mt = self.rna_mt[
            self.rna_mt.obs.n_genes_by_counts < rna_max_cells, :
        ]
        self.rna_mt = self.rna_mt[
            self.rna_mt.obs.pct_counts_mt < rna_mt_counts, :
        ]


    def atac_qc(self,
        atac_min_features:int = 3,
        atac_min_cells:int = 100,
        atac_max_cells:int = 2000
        ):
        sc.pp.filter_cells(self.atac_mt, min_genes = atac_min_features)
        sc.pp.filter_genes(self.atac_mt, min_cells = atac_min_cells)
        #self.atac_mt = self.atac_mt[self.atac_mt.obs.n_genes_by_counts < atac_max_cells, :]


    def merge_peak(self, width = 250):
        """
        This function merge peaks and fix the peak length as 500bp,
        which is easier for downstream analysis.
        """
        peak_df = self.peak_region_df
        ### set query peak. The first peak is query peak
        chr1 = peak_df.iloc[0,0]
        start1 = int(peak_df.iloc[0,1])
        end1 = int(peak_df.iloc[0,2])
        atac_array = self.atac_mt.X.toarray().T
        peak_count1 = atac_array[0]
        ###
        final_chr = []
        final_start = []
        final_end = []
        row_name_list = []
        peak_count_dict = {}
        for i in range(len(peak_df.index)):

            ### set target peak
            chr2 = peak_df.iloc[i,0]
            start2 = int(peak_df.iloc[i,1])
            end2 = int(peak_df.iloc[i,2])
            peak_count2 = atac_array[i]
            ### merge peak
            # 読めないから自分でやってください
            if (chr1 == chr2) & (((start1>=start2) & (start1<=end2)) |
                                 ((end1>=start2) & (end1<=end2)) |
                               ((start1<=start2) & (end1 >end2)) |
                                 ((start2>end1) & ((end2-start1) < width*2) ) |
                               ((start1>end2) & ((end1-start2) < width*2) ) ):
                new_region = sorted([start1,end1,start2,end2])
                start1 = new_region[0]

                end1 = new_region[3]

                ### merge peak count
                peak_count1 = peak_count1 + peak_count2
            else:
                ### fix peak region
                new_start = start1 + int((end1-start1)/2) - width
                new_end = start1 + int((end1-start1)/2) + width
                if new_start<1:
                    new_start = 1
                    new_end = width*2
                row_name = chr1 + ':' + str(new_start) + '-' + str(new_end)
                row_name_list.append(row_name)
                final_chr.append(chr1)
                final_start.append(new_start)
                final_end.append(new_end)
                peak_name = peak_df.index[i]
                peak_count_dict[peak_name] = peak_count1
                ### keep iterating
                chr1 = chr2
                start1 =start2
                end1 = end2
                peak_count1 = peak_count2
        ### add the final row
        # final_chr.append(peak_df.iloc[len(peak_df.index)-1,0])
        # final_start.append(peak_df.iloc[len(peak_df.index)-1,1])
        # final_end.append(peak_df.iloc[len(peak_df.index)-1,2])
        # peak_count_dict[peak_df.index[len(peak_df.index)]] = atac_array[len(peak_df.index)]
        self.gene_region_df = pd.DataFrame(
            {'chr':final_chr, 'start':final_start, 'end':final_end},
            index = row_name_list
        )

    def annotate_peaks(self,tss_region=[2000,1000]):
        template = tgrn.Template_GRN(id='gff')
        template.load_genes(gff_path=self.gff_path)
        self.gff_gene = list(template.genes.keys())
        ### get symbol
        ### id change
        symbol_list = list()
        for i in template.genes.keys():
            symbol_list.append(template.genes[i].symbol)
        id_table = pd.Series(self.gff_gene,index=symbol_list)
        if self.rna_mt.var_names[0][0:3]!='ENS':
            gff_intersect_gene = list(np.intersect1d(symbol_list,
                                                     list(
                                                         self.rna_mt.var_names)))
            self.rna_mt = self.rna_mt[:, gff_intersect_gene]
            ### check duplicated
            id_use = id_table[self.rna_mt.var_names]
            id_use = pd.Series(id_use.index,index=id_use)
            dup_id = id_use[id_use.duplicated()].values
            id_use = id_use[-id_use.duplicated()]
            id_use = pd.Series(id_use.index,index=id_use)
            print('The following genes are duplicated when change symbol to'
                  ' ENS')
            print(dup_id)

            self.rna_mt.var_names = id_use[self.rna_mt.var_names].values

        else:
            gff_intersect_gene = list(np.intersect1d(self.gff_gene,
                                                     list(
                                                         self.rna_mt.var_names)))

            self.rna_mt = self.rna_mt[:, gff_intersect_gene]


        template.load_cres()
        template.get_genetic_regions()
        annotations = dict()
        for id in list(self.atac_mt.var_names):
            cur_chr = None
            cur_index = 0
            skip_chr = False
            # Check peaks IDs
            if re.search(r'.*:\d*\-\d*', id):
                annotations[id] = None
                temp = id.split(':')
                chr = temp[0]

                # Skip weitd Chr
                if chr not in template.regions: continue
                # Reset pointer while entering new chr
                if chr != cur_chr:
                    cur_chr = chr
                    cur_index = 0
                    skip_chr = False
                # Skip rest of peaks in current chr
                elif chr == cur_chr and skip_chr: continue

                # Get peak position
                start = int(temp[1].split('-')[0])
                end = int(temp[1].split('-')[1])
                assert start <= end
                peak = pd.Interval(start, end, closed = 'both')

                # cur_index = 0
                while cur_index < len(template.regions[chr]):
                    ele = template.regions[chr][cur_index]

                    # Load position
                    position = pd.Interval(
                        ele['pos'][0]-tss_region[0],
                        ele['pos'][0]+tss_region[1],
                        closed = 'both',
                    )
                    # Load promoter
                    if ele['promoter']:
                        promoter = pd.Interval(
                            ele['promoter'][0],
                            ele['promoter'][1],
                            closed = 'both',
                        )

                    # Check overlaps
                    if peak.overlaps(position):
                        overlap_region = [
                            max(peak.left, position.left),
                            min(peak.right, position.right),
                        ]

                        annotations[id] = {
                            'id':ele['id'],
                            'gene':False,
                            'cre':False,
                            'promoter':False,
                        }
                        # Check whether there is promoter count
                        if ele['promoter']:
                            annotations[id]['gene'] = overlap_region
                            if peak.overlaps(promoter):
                                annotations[id]['promoter'] = [
                                    max(peak.left, promoter.left),
                                    min(peak.right, promoter.right),
                                ]
                        # If not having promoter, it should be a CRE
                        else:
                            annotations[id]['cre'] = overlap_region
                        break

                    # What if peak only in promoter region
                    elif ele['promoter'] and peak.overlaps(promoter):
                        annotations[id] = {
                            'id':ele['id'],
                            'gene':False,
                            'cre':False,
                            'promoter':[
                                max(peak.left, promoter.left),
                                min(peak.right, promoter.right),
                            ],
                        }
                        break

                    # No need to check others if fail to reach minimum value of
                    # current record
                    if peak.right <= min(position.left, promoter.left):
                        break

                    cur_index += 1
                    # Everything comes next will not fit, then skip this chr
                    if cur_index == len(template.regions[chr]):
                        if peak.left >= max(position.right, promoter.right):
                            skip_chr = True
                        else:
                            cur_index -= 1
                            break
        self.peak_annotations = grn_recon.Reverse_Peaks_Ann(annotations)
    def generate_feature_mt(self):

        fea_mt_dict = {}
        gff_gene = np.array(list(self.peak_annotations.keys()))
        rna_gene = list(self.rna_mt.var.index)
        atac_count = self.atac_mt.X.todense()
        for cell in list(self.rna_mt.obs.index):

            fea_mt = pd.DataFrame(
                {
                    'gene_mean_exp': [0] * len(self.gff_gene),
                    'peak_mean_count': [0] * len(self.gff_gene)
                },
                index=list(self.gff_gene),
                dtype='float64'
            )

            for i in list(gff_gene):

                if i in rna_gene:
                    all_overlap_peaks = list(self.peak_annotations[i].keys())
                    gene_count = int(self.rna_mt[cell,i].X.todense())
                    gene_peak_count = 0
                    for j in all_overlap_peaks:
                        peak_count = \
                            int(self.atac_mt[cell,j].X.todense())
                        gene_peak_count += peak_count
                else:
                    gene_peak_count = 0
                    gene_count = 0

                fea_mt.loc[i][0] = gene_count
                fea_mt.loc[i][1] = gene_peak_count

            fea_mt_dict[cell] = fea_mt
        return fea_mt_dict



    def __update_psNetwork(self, key, rna_cell_use, atac_cell_use):
        assert key not in self.pseudo_network
        self.pseudo_network[key] = {'rna': rna_cell_use, 'atac': atac_cell_use}

    def format_grp(grp):
        path = resource_filename(
            __name__, '../data/' + '/fatez_ens_symbol.csv'
        )
        gene_corr = pd.read_csv(path)
        path = resource_filename(
            __name__, '../data/' + '/fatez_tf_use.txt'
        )
        tf_all = pd.read_table(path, header=None)
        tf_all = tf_all[0]
        gene_all = gene_corr['ENS']
        gene_corr.index = gene_corr['symbol'].to_list()
        if grp.index[0][0:3] != 'ENS':
            index_intersect = list(np.intersect1d(grp.index,
                                                  list(gene_corr['symbol'])))
            column_intersect = list(np.intersect1d(grp.columns,
                                                   list(gene_corr['symbol'])))
            grp = grp.loc[index_intersect, column_intersect]
            gene_corr = pd.Series(gene_corr['ENS'].to_list(),
                                  index=gene_corr['symbol'])
            gene_corr = gene_corr[[not i for i in gene_corr.index.duplicated()]]
            grp.index = gene_corr[grp.index].values
            grp.columns = gene_corr[grp.columns].values
            grp = grp.loc[list(np.intersect1d(tf_all, grp.index))]
        else:
            index_intersect = list(np.intersect1d(grp.index,
                                                  list(gene_corr['ENS'])))
            column_intersect = list(np.intersect1d(grp.columns,
                                                   list(gene_corr['ENS'])))
            grp = grp.loc[index_intersect, column_intersect]
            grp = grp.loc[list(np.intersect1d(tf_all, grp.index))]

        ### add tf zero matrix
        tf_diff = list(np.setdiff1d(tf_all, list(grp.index)))
        zero_mt = pd.DataFrame(np.zeros((len(tf_diff), len(grp.columns))))
        zero_mt.index = tf_diff
        zero_mt.columns = list(grp.columns)
        grp = pd.concat([grp, zero_mt])

        ### add gene zero matrix
        gene_diff = list(np.setdiff1d(gene_all, list(grp.columns)))
        zero_mt = pd.DataFrame(np.zeros((len(grp.index), len(gene_diff))))
        zero_mt.index = list(grp.index)
        zero_mt.columns = gene_diff
        grp = pd.concat([grp, zero_mt], axis=1)

        return grp


