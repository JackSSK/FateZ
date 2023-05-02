#!/usr/bin/env python3
"""
Test script to make sure FateZ working properly.

author: jy
"""
import fatez.tool.gff as gff
from fatez.test.faker import Faker

__all__ = [
    'Faker',
]

class Error(Exception):
    pass



# Still using this?
def make_template_grn_jjy():
    mm10_gff = gff.Reader('E:\\public/gencode.vM25.basic.annotation.gff3.gz')
    mm10_template = mm10_gff.get_genes_gencode(id = 'GRCm38_template')
    print(mm10_template.gene_regions)
