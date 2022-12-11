#!/usr/bin/env python3
"""
GRN inference.

Note: Developing~

author: jy
"""
import re
import pandas as pd
import fatez.lib.grn as grn
import fatez.tool.mex as mex



class Reconstruct(object):
    """
    Main object to reconstruct GRNs from given data.
    Note: this object is for GRNs belonging to same sample group (cell type)
    """
    def __init__(self, template_grn:grn.GRN = None):
        super(Reconstruct, self).__init__()
        self.template = template_grn

    def annotate_peaks(self, features:list = None,):
        """
        Annotate Peaks with Gene and CRE information from template GRN.

        Note: This version won't need features to be sorted by start pos.

        ToDo: During pre-process, make sure features are sorted by start pos
        and stored into DataFrame format as MEX feature.tsc.gz would lead to.
        Therefore, we can set a pointer while iterating region records and
        make sure there is no need to look back.

        :param features:<class pandas.DataFrame Default = None>
			DataFrame of features

        :return <class 'dict'>
        """
        annotations = dict()
        for index in range(len(features)):
            id = features.iloc[[index]][0].to_string(index = False)
            type = features.iloc[[index]][2].to_string(index = False)
            # Check peaks IDs
            if type == 'Peaks' and re.search(r'.*:\d*\-\d*', id):
                annotations[id] = None
                temp = id.split(':')
                chr = temp[0]
                if chr not in self.template.regions: continue

                # Get peak position
                start = int(temp[1].split('-')[0])
                end = int(temp[1].split('-')[1])
                assert start <= end
                peak = pd.Interval(start, end, closed = 'both')

                for i, ele in enumerate(self.template.regions[chr]):
                    # Check overlaps
                    if peak.overlaps(ele['pos']):
                        annotations[id] = {
                            'id':ele['id'],
                            'promoter':False,
                            'gene':False,
                        }
                        # Check whether there is promoter count
                        if ele['promoter_pos'] is not None:
                            annotations[id]['gene'] = True
                            if ele['pos'].left >= ele['promoter_pos']:
                                tss = ele['pos'].left
                            else:
                                tss = ele['pos'].right
                            if tss in peak:
                                annotations[id]['promoter'] = True
                        break
                    # What if peak only in promoter region
                    elif (ele['promoter_pos'] != -1 and
                            ele['promoter_pos'] in peak):
                        annotations[id] = {
                            'id':ele['id'],
                            'promoter':True,
                            'gene':False,
                        }
                        break
                    # No need to check others if filling into region gap
                    if i > 0:
                        prev_ele = self.template.regions[chr][i-1]
                        pre_max = max(
                            prev_ele['pos'].right, prev_ele['promoter_pos']
                        )
                        cur_min = min(ele['pos'].left, ele['promoter_pos'])
                        if peak.left >= pre_max and peak.right <= cur_min:
                            break
        return annotations

    def paired_multi_MEX(self,
        mex_reader:mex.Reader = None,
        group_barcodes:list = None,
        ):
        """
        Reconstruct sample GRNs with paired multiomic seq data in MEX format.

        :param mex_reader:<class fatez.tool.mex.Reader Default = None>
			A reader object that fed with MEX information.

        :param group_barcodes:<list Default = None>
			List of barcodes that representing cells belonging to a same
            sample group. (e.g. Cell types / Cell states)

        :return: dict of grn.GRN objects.

        ToDo:
        1. Infer GRPs with motif enrichment -> probably go by another object
        """
        answer = dict()
        peak_annotations = self.annotate_peaks(mex_reader.features)
        # Maybe take seperate_matrices() operation outside later
        matrices = mex_reader.seperate_matrices()

        # Process each sample
        for i in range(len(mex_reader.barcodes)):
            id = mex_reader.barcodes[i]
            if group_barcodes is None or id in group_barcodes:
                sample_grn = grn.GRN(id = id)
                exps = matrices['Gene Expression'][i]
                # Note: In the peak matrix, records must be ordered by start pos
                peaks = matrices['Peaks'][i]

                # Process genes
                for index in exps.index:
                    gene_rec = mex_reader.features.iloc[[index]]
                    gene_id = gene_rec[0].to_string(index = False)
                    if gene_id in self.template.genes:
                        sample_grn.add_gene(
                            grn.Gene(id = gene_id, rna_exp = float(exps[index]))
                        )
                sample_grn = self._add_missing_genes(sample_grn)

                # Process peaks
                for index in peaks.index:
                    count = float(peaks[index])
                    if count > 0.0:
                        peak_rec = mex_reader.features.iloc[[index]]
                        peak_id = peak_rec[0].to_string(index = False)
                        ann = peak_annotations[peak_id]
                        # leave peaks not having annotations
                        if ann is None: continue
                        if ann['gene']:
                            sample_grn.genes[ann['id']].peak_count += count
                        if ann['promoter']:
                            sample_grn.genes[ann['id']].promoter_peaks += count
                        if not ann['gene'] and not ann['promoter']:
                            print('CRE part still developing')

                # Add sample _grn
                answer[id] = sample_grn

        return answer

    def unpaired_sth(self,):
        """
        Take your time
        """
        return

    def _add_missing_genes(self, sample_grn:grn.GRN = None):
        """
        Add genes included in the template GRN but absent in a sample GRN.

        :param sample_grn:<class fatez.lib.grn.GRN Default = None>
			A GRN object need to be double-checked that every gene in the
            template GRN is included.
        """
        for gene in self.template.genes:
            if gene not in sample_grn.genes:
                sample_grn.add_gene(
                    grn.Gene(
                        id = gene,
                        symbol = self.template.genes[gene].symbol,
                        rna_exp = float(0.0)
                    )
                )
        return sample_grn



    # Populate CRE(), Gene(), GRP() with scRNA-seq, scATAC-seq and necessary DBs
    # Eventually, we aim to obtain a dictionary of reconstructed GRNs
    # If on single-cell level, it would be like:
    # self.grns = {
    #     'barcode1': GRN_Object,
    #     'barcode2': GRN_Object,
    # }
