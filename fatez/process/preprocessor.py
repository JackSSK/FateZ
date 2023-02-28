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
        tf_target_db_path:str = None,
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
        self.tf_target_db_path = tf_target_db_path
        self.data_type = data_type
        self.rna_mt = list()
        self.atac_mt = list()
        self.peak_count = dict()
        self.peak_annotations = dict()
        self.peak_region_df = pd.DataFrame()
        self.gene_region_df = pd.DataFrame()
        self.pseudo_network = dict()
        self.peak_gene_links = dict()
        self.tf_target_db = dict()
        self.motif_enrich_score = dict()
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
        elif matrix_format == '10x_unpaired':
            self.rna_mt = sc.read_10x_mtx(self.rna_path, var_names = 'gene_ids')
            if self.atac_path[-1] !='/': self.atac_path += '/'
            mtx = anndata.read_mtx(self.atac_path+'matrix.mtx.gz').T
            peak = pd.read_table(self.atac_path+'features.tsv.gz', header=None)
            barcode = pd.read_table(
                self.atac_path + 'barcodes.tsv.gz', header = None
            )
            mtx.obs_names = list(barcode[0])
            mtx.var_names = list(peak[0])
            self.atac_mt = mtx
            atac_array = self.atac_mt.X.toarray().T

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
        for i, name in enumerate(list(self.atac_mt.var_names)):
            temp = name.split(':')
            positions = temp[1].split('-')
            chr_list.append(temp[0])
            start_list.append(positions[0])
            end_list.append(positions[1])
            self.peak_count[name] = atac_array[i]


        # gff = gff1.Reader(self.gff_path)
        # gff_template = gff.get_genes_gencode()
        #
        # symbol_list = []
        # gene_chr_list = []
        # gene_start_list = []
        # gene_end_list = []
        # for i in list(gff_template.genes.keys()):
        #
        #     symbol_list.append(gff_template.genes[i].symbol)
        #     gene_chr_list.append(gff_template.genes[i].chr)
        #     gene_start_list.append(gff_template.genes[i].position[0])
        #     gene_end_list.append(gff_template.genes[i].position[1])
        #
        # if self.rna_mt[0][0:3] == 'ENS':
        #     row_name_list = gff_template.genes.keys()
        # else:
        #     row_name_list = symbol_list
        #
        # self.atac_h5ad = ad.read(self.atac_path)
        #
        #

        self.peak_region_df = pd.DataFrame(
            {'chr': chr_list, 'start': start_list, 'end': end_list},
            index = list(self.atac_mt.var_names)
        )

        ###remove nan
        self.rna_mt = self.rna_mt[:,~self.rna_mt.var_names.isnull()]
        self.atac_mt = self.atac_mt[:,~self.atac_mt.var_names.isnull()]
        ### load tf target db
        """
        If following method would only be used here, then just add codes here.
        """
        self.__get_target_related_tf()


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
        self.peak_region_df = self.peak_region_df.loc[
            list(self.atac_mt.var_names)
        ]
        ### peak_count dict
        atac_array = self.atac_mt.X.toarray().T
        for i, name in enumerate(list(self.atac_mt.var_names)):
            self.peak_count[name] = atac_array[i]

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
        self.peak_count = peak_count_dict
        self.gene_region_df = pd.DataFrame(
            {'chr':final_chr, 'start':final_start, 'end':final_end},
            index = row_name_list
        )

    def add_cell_label(self,cell_types,modality:str = 'rna'):
        if modality == 'rna':
            self.rna_mt = self.rna_mt[
                np.intersect1d(cell_types.index, self.rna_mt.obs_names)
            ]
            cell_types = cell_types[list(self.rna_mt.obs_names)]
            self.rna_mt.obs['cell_types'] = list(cell_types)
        elif modality == 'atac':
            self.atac_mt = self.atac_mt[cell_types.index]
            cell_types = cell_types[list(self.atac_mt.obs_names)]
            self.atac_mt.obs['cell_types'] = list(cell_types)
        else:
            print('input correct modality')

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

            self.rna_mt.var_names = self.rna_mt[:, gff_intersect_gene]


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

    def make_pseudo_networks(self,
        network_cell_size:int = 10,
        data_type:str = 'paired',
        same_cell_type = True,
        specific_network_number = None,
        ):
        """
        Because of the sparsity of single cell data and data asymmetry,
        pick up several cells to make pseudo cell help to increase model
        accuracy, and to significantly reduce running time.
        This function makes pseudo cell with 'sample_cells'
        """
        #pseudo_sample_dict = {}
        ### sample cells
        if same_cell_type:
            ### intersect cell types
            if data_type == 'unpaired':
                cell_type_use = np.intersect1d(
                    list(set(self.rna_mt.obs.cell_types)),
                    list(set(self.atac_mt.obs.cell_types))
                )
            elif data_type == 'paired':
                cell_type_use = set(self.rna_mt.obs.cell_types)

            for i in cell_type_use:
                rna_data = self.rna_mt[self.rna_mt.obs.cell_types == i]
                # Good habit to init var even may not be used in all cases
                atac_data = None
                if data_type == 'unpaired':
                    atac_data = self.atac_mt[self.atac_mt.obs.cell_types == i]

                ### set pseudo sample num. when cell number of this cell type
                ### < 1000, pseudo sample number is set as 1000. when >1000,
                ### pseudo sample number = cell number of this cell type
                if specific_network_number == None:
                    if data_type == 'unpaired':
                        if  max(len(atac_data.obs),len(rna_data.obs)) <1000:
                            network_number = 1000
                        else:
                            network_number = max(len(atac_data), len(rna_data))
                    elif data_type == 'paired':
                        if  len(rna_data.obs) <1000:
                            network_number = 1000
                        else:
                            network_number = len(rna_data)
                else:
                    network_number = specific_network_number

                for j in range(network_number):
                    key = str(i) +'_'+ str(j)
                    rna_cell_use = random.sample(
                        list(range(len(rna_data))), network_cell_size
                    )
                    cell_name = list(rna_data[rna_cell_use].obs_names)
                    # Too long
                    rna_cell_use = [list(self.rna_mt.obs_names).index(x) for x in cell_name]

                    if data_type == 'paired':
                        # rna_cell_use = self.rna_mt.obs_names[random.sample(range(len(self.rna_mt.obs_names)),
                        #                                                  network_cell_size)]
                        self.__update_psNetwork(key, rna_cell_use, rna_cell_use)

                    if data_type == 'unpaired':
                        atac_cell_use = random.sample(
                            list(range(len(atac_data))), network_cell_size
                        )
                        cell_name = list(atac_data[atac_cell_use].obs_names)
                        self.__update_psNetwork(
                            key,
                            rna_cell_use,
                            [list(self.atac_mt.obs_names).index(x) for x in cell_name]
                        )

        else:
            for i in range(specific_network_number):
                #pseudo_sample_dict[i] = {'rna':[],'atac':[]}
                ### sample cells
                if data_type == 'paired':
                    #rna_cell_use = self.rna_mt.obs_names[random.sample(range(len(self.rna_mt.obs_names)),
                    #                                                  network_cell_size)]
                    rna_cell_use = random.sample(
                        list(range(len(self.rna_mt.obs_names))),
                        network_cell_size
                    )
                    self.__update_psNetwork(i, rna_cell_use, rna_cell_use)

                elif data_type == 'unpaired':
                    #rna_cell_use = self.rna_mt.obs_names[random.sample(range(len(self.rna_mt.obs_names)),
                    #                                                   network_cell_size)]
                    #atac_cell_use = self.atac_mt.obs_names[random.sample(range(len(self.atac_mt.obs_names)),
                    #                                           network_cell_size)]
                    rna_cell_use = random.sample(
                        list(range(len(self.rna_mt.obs_names))),
                        network_cell_size
                    )
                    atac_cell_use = random.sample(
                        list(range(len(self.atac_mt.obs_names))),
                        network_cell_size
                    )
                    self.__update_psNetwork(i, rna_cell_use, atac_cell_use)
        #         ### output gene and peak matrixs
        #
        #         rna_pseudo = self.rna_mt[rna_cell_use]
        #         rna_pseudo_network = pd.DataFrame(rna_pseudo.X.todense().T)
        #         rna_pseudo_network.index = list(rna_pseudo.var_names.values)
        #         rna_pseudo_network.columns = rna_pseudo.obs_names.values
        #         pseudo_sample_dict[i]['rna'] = rna_pseudo_network
        #
        #         atac_pseudo = self.atac_mt[atac_cell_use]
        #         atac_pseudo_network = pd.DataFrame(atac_pseudo.X.todense().T)
        #         atac_pseudo_network.index = list(atac_pseudo.var_names.values)
        #         atac_pseudo_network.columns = atac_pseudo.obs_names.values
        #         pseudo_sample_dict[i]['atac'] = atac_pseudo_network
        # return pseudo_sample_dict

    def slidewindow(self):
        print('under development')

    def cal_peak_gene_cor(self, exp_thr = 0, cor_thr = 0.6):
         warnings.filterwarnings('ignore')
         for network in list(self.pseudo_network.keys()):
             print(network)
             ### extract rna pseudo network
             rna_cell_use = self.pseudo_network[network]['rna']
             rna_pseudo = self.rna_mt[rna_cell_use]
             rna_pseudo_network = pd.DataFrame(rna_pseudo.X.todense().T)
             rna_pseudo_network.index = list(rna_pseudo.var_names.values)
             rna_pseudo_network.columns = rna_pseudo.obs_names.values
             rna_row_mean = rna_pseudo_network.mean(axis=1)

             ### select expressed genes to reduce computional cost
             rna_row_mean = rna_row_mean[rna_row_mean > exp_thr]

             ### atac cell
             atac_cell_use = self.pseudo_network[network]['atac']
             ### overlapped genes
             mt_gene_array = np.array(rna_row_mean.index)
             gff_gene_array = np.array(list(self.peak_annotations.keys()))
             gene_use = np.intersect1d(mt_gene_array,gff_gene_array)
             self.peak_gene_links[network] = {}
             for i in list(gene_use):

                ### load target gene related tfs
                ### then refine the list by gene_use
                related_tf = np.array(self.tf_target_db[i]['motif'])
                #tf_use = np.intersect1d(related_tf,gene_use)

                self.peak_gene_links[network][i] = {}
                all_overlap_peaks = list(self.peak_annotations[i].keys())
                peak_cor = []
                peak_use = []
                ###
                for j in all_overlap_peaks:
                    rna_count = rna_pseudo_network.loc[i,]
                    atac_count = self.peak_count[j][atac_cell_use]

                    pg_cor = stats.pearsonr(list(rna_count),list(atac_count))
                    if abs(pg_cor[0]) > cor_thr:
                        peak_cor.append(pg_cor[0])
                        peak_use.append(j)
                # gene_mean_count = rna_row_mean[i]
                if len(peak_use) > 0:
                    peak_series = pd.Series(peak_cor,index=peak_use)
                    peak_series_abs = abs(peak_series)
                    if len(peak_series) > 1:
                        cor_max_peak = peak_series_abs.sort_values().index[
                            len(peak_series_abs)-1]
                        # cor_max = peak_series[cor_max_peak]
                    else:
                        cor_max_peak = peak_series.index[0]
                    #     cor_max = peak_series[0]
                    mean_count = self.peak_count[cor_max_peak][atac_cell_use].mean()
                    self.peak_gene_links[network][i]['peak'] = cor_max_peak
                    # self.peak_gene_links[network][i][
                    #     'peak_correlation'] = cor_max
                    self.peak_gene_links[network][i]['peak_mean_count'] = mean_count
                    # self.peak_gene_links[network][i][
                    #     'gene_mean_count'] = gene_mean_count
                    # self.peak_gene_links[network][i][
                    #     'related_tf'] = related_tf
                else:
                    self.peak_gene_links[network][i]['peak'] = None
                    # self.peak_gene_links[network][i][
                    #     'peak_correlation'] = 0
                    self.peak_gene_links[network][i]['peak_mean_count'] = 0
                    # self.peak_gene_links[network][i][
                    #     'gene_mean_count'] = gene_mean_count
                    # self.peak_gene_links[network][i][
                    #     'related_tf'] = related_tf

    def extract_motif_score(self, grps):
        gene_use = grps.columns
        tf_use = grps.index
        all_score = []
        for j in gene_use:
            target_gene_tf_score = self.tf_target_db[j]
            tf = target_gene_tf_score['motif']
            score = target_gene_tf_score['score']
            tf_zero = list(set(tf_use)-set(tf))
            tf_zero_pd = pd.Series([0]*len(tf_zero),index=tf_zero)
            #print(tf_zero)
            db_tf_score_seires = pd.Series(score, index=tf)
            db_tf_score_seires = pd.concat([tf_zero_pd,
                                            db_tf_score_seires], axis=0)
            score_use = db_tf_score_seires[tf_use].to_numpy()
            all_score.append(score_use)
        motif_score_mt = np.matrix(all_score).T.astype(float)
        ### scale to range 0 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        motif_score_mt = scaler.fit_transform(motif_score_mt)
        motif_score_mt = np.array(motif_score_mt)
        self.motif_enrich_score = motif_score_mt

    def generate_grp(self, expressed_cell_percent_thr:int = 0.05):
        ### This strategy is faster than using for
        ### loop to ilterate millions grps
        ### Basicaly, this strategy first using numpy
        ### to calculate correlation between all genes (3S running time
        ### for 1.5W genes),
        ### then filter the source and target genes
        pseudo_network_grp = {}
        ### merge all genes used in pseudo cell
        gene_all = []
        for i in list(self.peak_gene_links.keys()):
            gene_use = list(self.peak_gene_links[i].keys())
            gene_all = gene_all +gene_use
        gene_all = list(set(gene_all))
        ### add zero matrix
        gene_diff = list(np.setdiff1d(self.gff_gene, gene_all))
        zero_mt = np.zeros((len(gene_diff),len(self.rna_mt)))

        ### calculate correlation between genes
        ### then melt df to get all grps
        ### correlation calculation method in numpy
        mt_use = self.rna_mt[:, gene_all]
        mt_use = mt_use.X.todense().T
        mt_use = np.concatenate([mt_use,zero_mt],axis=0)
        mt_cor = np.corrcoef(mt_use)
        mt_cor_df = pd.DataFrame(mt_cor)
        mt_cor_df.index = gene_all + gene_diff
        mt_cor_df.columns = gene_all + gene_diff
        ### extract expressed tfs
        # expressed_tfs = self.__extract_expressed_tfs(
        #     expressed_cell_percent=expressed_cell_percent_thr)
        # expressed_tfs = np.intersect1d(expressed_tfs,gene_all)
        #
        # df_use = mt_cor_df.loc[expressed_tfs]
        path = resource_filename(
            __name__, '../data/' + 'mouse' + '/Transfac201803_MotifTFsF.txt.gz'
        )
        ### filter tfs
        tf_motifs = transfac.Reader(path=path).get_tfs()
        tf_all = list(tf_motifs.keys())
        tf_all = list(np.intersect1d(tf_all,self.gff_gene))
        df_use = mt_cor_df.loc[tf_all]
        df_use = df_use[self.gff_gene]
        df_use = df_use.loc[tf_all]

        return df_use

    def output_pseudo_samples(self):
        pseudo_sample_dict = {}
        gene_all = []
        for i in list(self.peak_gene_links.keys()):
            gene_use = list(self.peak_gene_links[i].keys())
            gene_all = gene_all + gene_use
        gene_all = list(set(gene_all))
        gene_diff = np.setdiff1d(self.gff_gene,gene_all)

        for i in list(self.peak_gene_links.keys()):
            ### pseudo sample
            pseudo_sample_dict[i] = {'rna': [], 'atac': []}
            rna_cell_use = self.pseudo_network[i]['rna']
            gene_use = np.array(list(self.peak_gene_links[i].keys()))
            pseudo_sample_dict[i] = {'rna': [], 'atac': []}
            rna_pseudo = self.rna_mt[rna_cell_use]
            rna_pseudo_network = pd.DataFrame(rna_pseudo.X.todense().T)
            rna_pseudo_network.index = list(rna_pseudo.var_names.values)
            rna_pseudo_network.columns = rna_pseudo.obs_names.values
            ### whether select invariable gene
            rna_pseudo_network = rna_pseudo_network.loc[gene_all]

            ### define all feature mt
            fea_mt = pd.DataFrame(
                {
                    'gene_mean_exp': [0]*len(self.gff_gene),
                    'peak_mean_count': [0]*len(self.gff_gene)
                },
                index=list(self.gff_gene),
                dtype='float64'
            )
            ### first feature mean expression
            gene_mean_exp = pd.Series(rna_pseudo_network.mean(axis=1),
                                      index=rna_pseudo_network.index)

            ### second feature  overlapped peak count
            for j in gene_all:
                fea_mt.loc[j][0] = gene_mean_exp[j]
                if j in gene_use:
                    fea_mt.loc[j][1] = self.peak_gene_links[i][j]['peak_mean_count']
            pseudo_sample_dict[i] = fea_mt
        return pseudo_sample_dict

    def __update_psNetwork(self, key, rna_cell_use, atac_cell_use):
        assert key not in self.pseudo_network
        self.pseudo_network[key] = {'rna': rna_cell_use, 'atac': atac_cell_use}

    def __extract_expressed_tfs(self,expressed_cell_percent:int = 0.05):
        path = resource_filename(
            __name__, '../data/' + 'mouse' + '/Transfac201803_MotifTFsF.txt.gz'
        )
        ### make TFs motifs dict
        expressed_cell_thr = int(
            expressed_cell_percent*len(self.rna_mt.obs_names))
        tf_motifs = transfac.Reader(path=path).get_tfs()
        tf_all = list(tf_motifs.keys())
        gene_all = list(self.rna_mt.var_names)
        tf_use = np.intersect1d(tf_all,gene_all)
        tf_exp = self.rna_mt[:,tf_use].to_df().T
        tf_use_idx = []
        for i in range(tf_exp.shape[0]):
            expressed_num = tf_exp.iloc[i][tf_exp.iloc[i] != 0].shape[0]
            if expressed_num >= expressed_cell_thr:
                tf_use_idx.append(i)
        tf_use = tf_use[tf_use_idx]
        return tf_use

    # R u using this?
    def __is_overlapping(self,x1, x2, y1, y2):
        return max(x1, y1) <= min(x2, y2)

    def __get_target_related_tf(self, motif_score_type:str = 'mean'):
        motif1 = pd.read_table(self.tf_target_db_path)
        target_tf_dict = {}

        for i in motif1.index:

            row = motif1.iloc[i,]
            target_tf_dict[row[4]] = {'motif': [], 'score': []}
            tf_str = row[0]
            tf_list = tf_str.split(';')
            target_tf_dict[row[4]]['motif'] = tf_list

            ### select motif score type
            if motif_score_type=='mean':
                motif_score = row[1]
            elif motif_score_type=='median':
                motif_score = row[2]
            motif_score = motif_score.split(';')

            target_tf_dict[row[4]]['score'] = motif_score

        self.tf_target_db = target_tf_dict


