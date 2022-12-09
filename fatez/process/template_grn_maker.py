#!/usr/bin/env python3
"""
This file contains objects and functions to make template.

ToDo: load and associate CREs with Genes in template

author: jy
"""
import fatez.tool.gff as gff
import fatez.tool.JSON as json
from pkg_resources import resource_filename

def Make(path, template_id):
    gff_obj = gff.Reader(path)
    template_grn = gff_obj.get_genes_gencode(id = template_id)
    return gff_obj, template_grn

# Main function for testing purpose
if __name__ == '__main__':
    path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
    template_id = 'GRCm38_template'
    # path = '../data/human/gencode.v42.basic.annotation.gff3.gz'
    # template_id = 'GRCh38_template'
    gff_obj, template = Make(path, template_id)
    json.encode(template.as_dict(), '../data/mouse/template.js.gz')

    # Xkr4 = template.genes['ENSMUSG00000051951']
    # print(Xkr4.symbol)
    # gff_rec = gff_obj.get(Xkr4.gff_coordinates)
    # print(gff_rec)
