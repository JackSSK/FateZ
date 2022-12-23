#!/usr/bin/env python3
"""
Objects

Note: Try to keep line length within 81 Chars

author: jjy
"""
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
from Bio import motifs
import fatez.tool.gff as gff1
import fatez.tool.transfac as transfac
import fatez.lib.template_grn as tgrn
import fatez.tool.sequence as seq
import itertools
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
        self.rna_path = rna_path
        self.atac_path = atac_path
        self.gff_path = gff_path
        self.data_type = data_type
        self.rna_mt = list()
        self.atac_mt = list()
        self.peak_annotations = dict()
        self.peak_region_df = pd.DataFrame()
        self.gene_region_df = pd.DataFrame()
        self.pseudo_network = dict()

    """
    :param rna_path: <class fatez.lib.preprocess.preprocess>
    	The object of prospective regulatory source gene.

    :param atac_path: <class fatez.lib.preprocess.preprocess>
    	The object of prospective regulatory target gene.
    """



    def load_data(self):

        ### 10x paired data
        if self.data_type == 'paired':
            sc.settings.verbosity = 3
            sc.logging.print_header()
            sc.settings.set_figure_params(dpi=80, facecolor='white')

            self.rna_mt = sc.read_10x_mtx(
                self.rna_path,
                var_names = 'gene_ids',
                cache = True
            )

            self.atac_mt = sc.read_10x_mtx(
                self.atac_path,
                var_names='gene_ids',
                cache=True,
                gex_only=False
            )

            self.atac_mt = self.atac_mt[:, len(self.rna_mt.var_names):(len(self.atac_mt.var_names) - 1)]


        ### load unparied data or ???
        elif self.data_type == 'unpaired':
            self.rna_mt = sc.read_10x_mtx(
                self.rna_path,
                var_names = 'gene_ids',
                cache = True
            )
            self.atac_mt = ad.read(self.atac_path)

        ### load gff
        gff = gff1.Reader(self.gff_path)
        gff_template = gff.get_genes_gencode()

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


    def rna_qc(self,rna_min_genes:int=3,rna_min_cells:int=200,
                  rna_max_cells:int=2000,rna_mt_counts:int=5):
        sc.pp.filter_cells(self.rna_mt, min_genes=rna_min_genes)
        sc.pp.filter_genes(self.rna_mt, min_cells=rna_min_cells)
        ###
        self.rna_mt.var['mt'] = self.rna_mt.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.rna_mt, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        self.rna_mt = self.rna_mt[self.rna_mt.obs.n_genes_by_counts < rna_max_cells, :]
        self.rna_mt = self.rna_mt[self.rna_mt.obs.pct_counts_mt < rna_mt_counts, :]

    def atac_qc(self,atac_min_features:int=3,atac_min_cells:int=100,
                atac_max_cells:int=2000):
        sc.pp.filter_cells(self.atac_mt, min_genes=atac_min_features)
        sc.pp.filter_genes(self.atac_mt, min_cells=atac_min_cells)
        self.atac_mt = self.atac_mt[self.atac_mt.obs.n_genes_by_counts < atac_max_cells, :]


    def merge_peak(self,width=250):
        """
        This function merge peaks and fix the peak length as 500bp,
        which is easier for downstream analysis.
        """
        peak_df = self.gene_region_df
        ### set query peak. The first peak is query peak
        chr1 = peak_df.iloc[0,0]
        start1 = peak_df.iloc[0,1]
        end1 = peak_df.iloc[0,2]
        peak_count1 = self.atac_mt[:,0]
        ###
        final_chr = []
        final_start = []
        final_end = []
        final_atac_mt = self.atac_mt[:,0]

        for i in peak_df.index:

            ### set target peak
            chr2 = peak_df.iloc[i,0]
            start2 = peak_df.iloc[i,1]
            end2 = peak_df.iloc[i,2]
            peak_count2 = self.atac_mt[:,i]

            ### merge peak
            if (chr1 == chr2) & (((start1>=start2) & (start1<=end2)) | ((end1>=start2) & (end1<=end2)) |
                               ((start1<=start2) & (end1 >end2)) | ((start2>end1) & ((end2-start1) < width*2) ) |
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
                final_chr.append(chr1)
                final_start.append(new_start)
                final_end.append(new_end)
                final_atac_mt = vstack(final_atac_mt,peak_count1)
                ### keep iterating
                chr1 = chr2
                start1 =start2
                end1 = end2

        ### add the final row
        final_chr.append(peak_df.iloc[len(peak_df.index)-1,0])
        final_start.append(peak_df.iloc[len(peak_df.index)-1,1])
        final_end.append(peak_df.iloc[len(peak_df.index)-1,2])
        final_atac_mt = vstack(final_atac_mt, (len(self.atac_mt.var_names)-1))

        self.gene_region_df = pd.DataFrame({'chr': final_chr, 'start': final_start,
                                            'end': final_end}, index=row_name_list)


    def annotate_peaks(self):
        template = tgrn.Template_GRN(id='gff')
        template.load_genes(gff_path=self.gff_path)
        template.load_cres()
        template.get_genetic_regions()
        cur_chr = None
        cur_index = 0
        skip_chr = False
        annotations = dict()
        for id in list(self.atac_mt.var_names):
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
                        ele['pos'][0], ele['pos'][1], closed = 'both',
                    )
                    # Load promoter
                    if ele['promoter']:
                        promoter = pd.Interval(
                            ele['promoter'][0],ele['promoter'][1],closed='both',
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
        self.peak_annotations = annotations


    def make_pseudo_networks(self,
        network_cell_size:int = 10,
        data_type:str = None,
        network_number:int = 10,
        method = 'sample_cells'
        ):
        """
        Because of the sparsity of single cell data and data asymmetry,
        pick up several cells to make pseudo cell help to increase model
        accuracy, and to significantly reduce running time. This function
        provides two strategies to make pseudo cell: 'sample_cells' and
        'slidewindow'
        """
        ### sample cells
        if method == 'sample_cells':
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
                    rna_pseudo = self.rna_mt[rna_cell_use]
                    atac_pseudo = self.atac_mt[atac_cell_use]

                    rna_pseudo_network = pd.DataFrame(rna_pseudo.X.todense().T)
                    rna_pseudo_network.index = list(rna_pseudo.var_names.values)
                    rna_pseudo_network.columns = rna_pseudo.obs_names.values

                    atac_pseudo_network = pd.DataFrame(atac_pseudo.X.todense().T)
                    atac_pseudo_network.index = list(atac_pseudo.var_names.values)
                    atac_pseudo_network.columns = atac_pseudo.obs_names.values

                    self.pseudo_network[i]['rna'] = rna_cell_use
                    self.pseudo_network[i]['atac'] = atac_cell_use
        elif method == 'slidewindow':
            print('under development')






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
                                     peak_start,peak_end):
                    ### calculate correlation between gene count and peak count
                        rna_count = self.pseudo_network['rna'].loc[row,]
                        atac_count = self.pseudo_network['atac'].loc[j,]

                        pg_cor = stats.pearsonr(rna_count,
                                 atac_count)
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
    def find_motifs_binding(self, specie:str = 'mouse',ref_seq_path:str='a'):
        """
        This function calculate binding score for all motifs
        in specific regions (overlap region between peak and
        gene promoter).
        """
        peak_list = list(self.peak_annotations.keys)
        ### load tf motif relationships
        path = resource_filename(
            __name__, '../data/' + specie +'/Transfac201803_MotifTFsF.txt.gz'
        )
        ### make TFs motifs dict
        tf_motifs = transfac.Reader(path = path).get_motifs()
        motifs_use = [tf_motifs.keys()]

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

        ### load reference seq
        ref_seq = seq.Reader(path=ref_seq_path).get_fasta()

        ### load TRANSFAC PWM
        pwm_path = resource_filename(
            __name__, '../data/' + specie +'/Transfac_PWM.txt'
        )
        handle = open(pwm_path)
        record = motifs.parse(handle, "TRANSFAC")
        handle.close()
        score_all = 0
        peak_motif_dict = {}
        ### motif discovering
        for peak in peak_list:
            if self.peak_annotations[peak] != None:
                peak2 = peak.split(':')
                ### extract peak sequence
                chr = peak2[0]
                start = int(peak2.split('-')[0])
                end = int(peak2.split('-')[1])
                peak_sequence = ref_seq[chr][start-1:end]

                motif_score_dict = {}
                ### motif discovering
                for i in motifs_use:

                    score_dict[i] = []
                    motif_use_name = i[1:5]
                    motif = record[int(motif_use_name)]
                    pwm = motif.counts.normalize()
                    pssm = pwm.log_odds()
                    for position, score in pssm.search(peak_sequence, threshold=0):
                        motif_score_dict[i] = score
                self.peak_annotations.keys[peak] = motif_score_dict


    def __get_tf_related_motif(self):
        print('under development')

    def generate_grp(self):
        for network in [self.pseudo_network.keys()]:
            all_gene = network['rna'].index

            ### filter useless gens
            rowmean = network.mean(axis=1)
            rowstd = network.std(axis=1)
            network = network[rowmean > 0.01]
            network = network[rowstd > 0.1]
            gene_use = network.index
            ### extract tf
            path = resource_filename(
                __name__, '../data/' + specie + '/Transfac201803_MotifTFsF.txt.gz'
            )
            tf_motifs = transfac.Reader(path=path).get_tfs()
            tfs = np.array(tf_motifs.keys())
            tf_use = [x in np.array(network.index) for x in tfs]
            tf_use = np.array(tfs)[tf_use]

            ### create dataframe with tf_number (row) x source_number (column)
            grp_df = pd.DataFrame(np.zeros(((len(tf_use)*len(gene_use)), 7)))

            ### fill the dataframe by expression, peak count, motif enrichment, and regulation type
            for tf_num in range(len(tf_use)):
                for gene_num in range(len(gene_use)):
                    grp_df.iloc[(tf_num + 1) * (gene_num + 1), 0] = tf_use[tf_num]
                    grp_df.iloc[(tf_num + 1) * (gene_num + 1), 1] = gene_use[gene_num]
                    grp_df.iloc[(tf_num + 1) * (gene_num + 1), 2] = rowmean[tf_use[tf_num]]
                    grp_df.iloc[(tf_num + 1) * (gene_num + 1), 3] = rowmean[gene_use[gene_num]]



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
