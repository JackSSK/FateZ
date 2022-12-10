#!/usr/bin/env python3
"""
GRN inference.

Note: Developing~

author: jy
"""
import fatez.lib.grn as grn
import fatez.tool.mex as mex



class Reconstruct(object):
    """
    Main object to reconstruct GRNs from given data.
    Note: this object is for GRNs belonging to same sample group (cell type)
    """
    def __init__(self, template_grn:grn.GRN = None):
        super(Reconstruct, self).__init__()
        self.template_grn = template_grn

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
        1. Assign peaks to genes or CREs. -> we can make a new obj in lib
        2. Infer GRPs with motif enrichment -> probably go by another object
        """
        answer = dict()
        # Maybe take seperate_matrices() operation outside later
        matrices = mex_reader.seperate_matrices()
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
                    if gene_id in self.template_grn.genes:
                        sample_grn.add_gene(
                            grn.Gene(id = gene_id, rna_exp = float(exps[index]))
                        )
                sample_grn = self._add_missing_genes(sample_grn)

                # Process peaks
                # ToDo: Intersection part not finished yet
                prev_chr = None
                prev_ind = None
                prev_start = None
                for index in peaks.index:
                    peak_count = float(peaks[index])
                    if peak_count > 0.0:
                        peak_rec = mex_reader.features.iloc[[index]]
                        peak_id = peak_rec[0].to_string(index = False)
                        temp = peak_id.split(':')
                        chr = temp[0]
                        start = int(temp[1].split('-')[0])
                        end = int(temp[1].split('-')[1])

                        if chr != prev_chr:
                            prev_chr = chr
                            prev_ind = 0
                            prev_start = 0

                        assert prev_start < start
                        prev_start = start

                        region_rec = self.template_grn.regions[chr][prev_ind]
                        # Overlapping
                        if self._check_pos_overlap(start, end, region_rec):
                            print(region_rec['id'], region_rec['type'])
                        # Not reaching current record
                        elif end <= region_rec['pos'][0]:
                            continue
                        # Passed current record
                        elif start >= region_rec['pos'][1]:
                            prev_ind += 1
                            # If there is no other records
                            if prev_ind >= len(self.template_grn.regions[chr]):
                                continue
                            region_rec = self.template_grn.regions[chr][prev_ind]

                            # ToDo: check whether overlapping with next rec


                        print(chr, start, end, peak_count)
                        # Find which gene or CRE the peak belongs to
                        # If we have something like atac_peak_annotation,
                        # this can be very easy.
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
        for gene in self.template_grn.genes:
            if gene not in sample_grn.genes:
                sample_grn.add_gene(
                    grn.Gene(
                        id = gene,
                        symbol = self.template_grn.genes[gene].symbol,
                        rna_exp = float(0.0)
                    )
                )
        return sample_grn

    def _check_pos_overlap(self, start, end, region_rec):
        """
        """
        if ((start <= region_rec['pos'][0] and end > region_rec['pos'][0]) or
            (start >= region_rec['pos'][0] and end <= region_rec['pos'][0]) or
            (start < region_rec['pos'][1] and end >= region_rec['pos'][1])):
            return True
        else:
            return False


    # Populate CRE(), Gene(), GRP() with scRNA-seq, scATAC-seq and necessary DBs
    # Eventually, we aim to obtain a dictionary of reconstructed GRNs
    # If on single-cell level, it would be like:
    # self.grns = {
    #     'barcode1': GRN_Object,
    #     'barcode2': GRN_Object,
    # }
