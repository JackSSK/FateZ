#!/usr/bin/env python3
"""
GRN inference.

Note: Developing~

author: jy, nkmtmsys
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
        Note: This version WILL need features to be sorted by start pos.

        :param features:<class pandas.DataFrame Default = None>
            DataFrame of features

        :return <class 'dict'>
        """
        cur_chr = None
        cur_index = 0
        skip_chr = False
        annotations = dict()
        for index in range(len(features)):
            id = features.iloc[[index]][0].to_string(index = False)
            type = features.iloc[[index]][2].to_string(index = False)
            # Check peaks IDs
            if type == 'Peaks' and re.search(r'.*:\d*\-\d*', id):
                annotations[id] = None
                temp = id.split(':')
                chr = temp[0]

                # Skip weitd Chr
                if chr not in self.template.regions: continue
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
                while cur_index < len(self.template.regions[chr]):
                    ele = self.template.regions[chr][cur_index]

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
                    if cur_index == len(self.template.regions[chr]):
                        if peak.left >= max(position.right, promoter.right):
                            skip_chr = True
                        else:
                            cur_index -= 1
                            break
        return annotations

    def paired_multi_MEX(self,
        mex_reader:mex.Reader = None,
        peak_annotations:dict = None,
        group_barcodes:list = None,
        ):
        """
        Reconstruct sample GRNs with paired multiomic seq data in MEX format.

        :param mex_reader:<class fatez.tool.mex.Reader Default = None>
            A reader object that fed with MEX information.

        :param peak_annotations:<dict Default = None>
            The annotations for each peak.

        :param group_barcodes:<list Default = None>
            List of barcodes that representing cells belonging to a same
            sample group. (e.g. Cell types / Cell states)

        :return: dict of grn.GRN objects.

        ToDo:
        1. Infer GRPs with motif enrichment -> probably go by another object
        """
        answer = dict()
        if peak_annotations is None:
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
                            sample_grn.genes[ann['id']].peaks += count
                        if ann['promoter']:
                            sample_grn.genes[ann['id']].promoter_peaks += count
                        if ann['cre']:
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
