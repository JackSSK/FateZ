#!/usr/bin/env python3
"""
Calculate peak to gene linkages.

Note: Developing~

author: jjy
"""

import pyranges as pr
import pandas as pd

class linkages(Reader):
    """
    Preprocess to get corresponding
    relationships between peaks and genes
    """
    def __init__(self):




    def is_overlapping(x1, x2, y1, y2):
        return max(x1, y1) <= min(x2, y2)
    def overlap(self,peak_regions,gene_regions):




    def gp_correlation(self,):