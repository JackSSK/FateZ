#!/usr/bin/env python3
"""
Test script to make sure FateZ working properly

author: jy
"""

import fatez
import fatez.tool.gff as gff
from pkg_resources import resource_filename

def Test(**kwargs):
	"""
	Function to test whether FateZ is performing properly or not.

	:param kwargs: arguments

	:return: what to return
	"""
	print('Start Test')
	return

def make_template_grn():
	a = gff.Reader('../data/mouse/gencode.vM25.basic.annotation.gff3.gz')
	template = a.get_genes_gencode()
	template.id = 'GRCm38_template'
	template.save('../data/mouse/template.grn.js')

def test_grn():
	gene_a = grn.Gene(id = 'a', symbol = 'AAA')
	gene_b = grn.Gene(id = 'b', symbol = 'BBB')
	grp_ab = grn.GRP(reg_source = gene_a, reg_target = gene_b)
	grn = grn.GRN(id = 'grn')
	grn.add_gene(gene_a)
	grn.add_gene(gene_b)
	grn.add_grp(grp_ab)
	grn.as_dict()
	grn.as_digraph()

if __name__ == '__main__':
	make_template_grn()
