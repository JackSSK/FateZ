#!/usr/bin/env python3
"""
General Feature Format (GFF) file processing tools.

author: jy, nkmtmsys
"""
import re
import gzip
import pandas as pd
import fatez.tool as tool
import fatez.lib.grn as grn



class Reader(tool.Reader_Template):
    """
	Object to read in GFF file.
	"""
    def get_genes_gencode(self,
        id:str = None,
        skip_chrM:bool = True,
        gene_types:dict = {'protein_coding':None, },
        ):
        """
        Pattern GFF file obtained from GENCODE, extract information of genes,
        and make a template GRN.

        :param id: <str Default = None>
			ID of the output GRN template.

        :param skip_chrM: <bool Default = True>
			Skip genes on mitochondria.

        :param gene_types: <dict Default = None>
			Dict of gene types being considered to make template GRN.
            Currently, we only consider protein coding genes.
            We may increase this list later.

        :return: <class fatez.lib.grn.GRN>
        """
        template_grn = grn.GRN(id = id, ref_gff = self.path, cres = dict())
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
                chr = line[0]
                # source = line[1]
                beg = int(line[3])
                end = int(line[4])
                # score = line[5]
                strand = line[6]
                # phase = line[7]
                info = line[8].split(';')

                # Skip mitochondria genes.
                if skip_chrM and chr == 'chrM': continue
                # Skip pseudogene
                assert info[2].split('=')[0] == 'gene_type'
                type = info[2].split('=')[1]
                if type not in gene_types: continue

                assert info[0].split('=')[0] == 'ID'
                # Takes ENSMUSG00000064370, not ENSMUSG00000064370.1, as ID
                id = info[0].split('=')[1].split('.')

                assert info[3].split('=')[0] == 'gene_name'
                symbol = info[3].split('=')[1]

                # populate template GRN
                if id[0] not in template_grn.genes:
                    template_grn.add_gene(
                        grn.Gene(
                            id = id[0],
                            symbol = symbol，
                            chr = chr，
                    		start_pos = beg，
                    		end_pos = end，
                            strand = strand,
                            gff_coordinates = [coordinate],
                            cre_regions = list(),
                        )
                    )
                else:
                    template_grn.genes[id[0]].gff_coordinates.append(coordinate)

        return template_grn
