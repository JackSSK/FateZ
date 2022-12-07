#!/usr/bin/env python3
"""
General Feature Format (GFF) file processing tools.

author: jy, nkmtmsys
"""
import re
import gzip
import pandas as pd
from pkg_resources import resource_filename
import fatez.tool as tool
import fatez.lib.grn as grn



class Reader(tool.Reader_Template):
    """
	Object to read in GFF file.
	"""
    def get_genes_gencode(self,):
        """
        Pattern GFF file obtained from GENCODE, extract information of genes,
        and make a template GRN.

        :return: <class fatez.lib.grn.GRN>
        """
        template_grn = grn.GRN(ref_gff = self.path)
        while (True):
            line = self.file.readline()
            coordinate = self.file.tell() - len(line)
            # Skip comment line
            if line[0:1]=='#':
                continue
            # Stop when enconter empty line
            if line == '':
                break
            # Pattern contents in line
            line = line.split('\t')
            # Missing information
            if len(line) < 8:
                raise tool.Error('Bad GFF Format')

            if line[2] == 'gene':
                # chr = line[0]
                # source = line[1]
                # beg = int(line[3])
                # end = int(line[4])
                # score = line[5]
                # strand = line[6]
                # phase = line[7]
                info = line[8].split(';')
                # Skip pseudogene
                assert info[2].split('=')[0] == 'gene_type'
                type = info[2].split('=')[1]

                # Currently, we only consider protein_coding genes
                # We may increase this list later
                if not re.search(r'protein_coding', type):
                    continue

                assert info[0].split('=')[0] == 'ID'
                # Takes ENSMUSG00000064370, not ENSMUSG00000064370.1, as ID
                id = info[0].split('=')[1].split('.')[0]

                assert info[3].split('=')[0] == 'gene_name'
                symbol = info[3].split('=')[1]

                assert id not in template_grn.genes
                template_grn.add_gene(
                    grn.Gene(
                        id = id,
                        gff_coordinate = coordinate,
                        symbol = symbol
                    )
                )
        return template_grn

# Main function for tests
if __name__ == '__main__':
    a = Reader('../data/mouse/gencode.vM25.basic.annotation.gff3.gz')
    template = a.get_genes_gencode()
