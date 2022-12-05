#!/usr/bin/env python3
"""
GRN inference.

Note: Developing~

author: jy
"""

class Reconstruct(object):
    """
    Main object to reconstruct GRNs from given data.
    Note: this object is for GRNs belonging to same sample group (cell type)
    """

    def __init__(self, arg):
        super(Reconstructor, self).__init__()
        self.arg = arg
        self.grns = dict()

    # Populate CRE(), Gene(), GRP() with scRNA-seq, scATAC-seq and necessary DBs
    # Eventually, we aim to obtain a dictionary of reconstructed GRNs
    # If on single-cell level, it would be like:
    # self.grns = {
    #     'barcode1': GRN_Object,
    #     'barcode2': GRN_Object,
    # }
