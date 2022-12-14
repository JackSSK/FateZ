#!/usr/bin/env python3
"""
Test script to make sure FateZ working properly

author: jy
"""

import fatez
import fatez.tool.gff as gff
import fatez.lib.grn as grn
from pkg_resources import resource_filename

def Test(**kwargs):
	"""
	Function to test whether FateZ is performing properly or not.

	:param kwargs: arguments

	:return: what to return
	"""
	print('Start Test')
	return

def make_template_grn_jjy():
	mm10_gff = gff.Reader('E:\\public/gencode.vM25.basic.annotation.gff3.gz')
	mm10_template = mm10_gff.get_genes_gencode(id = 'GRCm38_template')
	print(mm10_template.gene_regions)


def test_grn():
	toy = grn.GRN(id = 'toy')
	# fake genes
	gene_list = [
		grn.Gene(id = 'a', symbol = 'AAA'),
		grn.Gene(id = 'b', symbol = 'BBB'),
		grn.Gene(id = 'c', symbol = 'CCC'),
	]
	# fake grps
	grp_list = [
		grn.GRP(reg_source = gene_list[0], reg_target = gene_list[1]),
		grn.GRP(reg_source = gene_list[0], reg_target = gene_list[2]),
		grn.GRP(reg_source = gene_list[1], reg_target = gene_list[2]),
	]
	# populate toy GRN
	for gene in gene_list:
		toy.add_gene(gene)
	for grp in grp_list:
		toy.add_grp(grp)

	toy.as_dict()
	toy.as_digraph()
	toy.save('../data/toy.grn.js.gz')
	# Load new GRN
	del toy
	toy_new = grn.GRN()
	toy_new.load('../data/toy.grn.js.gz')
	for id, rec in toy_new.grps.items():
		print(rec.reg_source.symbol)

if __name__ == '__main__':
	make_template_grn()
	test_grn()
