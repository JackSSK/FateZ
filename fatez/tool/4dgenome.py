#!/usr/bin/env python3
"""
4DGenome Data processing tools.

author: jy
"""
import re
import gzip
import fatez.tool as tool
import fatez.lib.grn as grn
from pyliftover import LiftOver



class Reader(tool.Reader_Template):
    """
	Object to read in 4DGenome file and get region interaction information.
	"""
    def process_specie(self, refernece:str = None, liftover_data_path:str:None):
        converter = LiftOver(liftover_data_path)
        answer = dict()
        while (True):
            line = self.file.readline()
            # Skip comment line
            if line[0:1]=='#':
                continue
            # Stop when enconter empty line
            if line == '':
                break
            # Pattern contents in line
            line = line.split('\t')
            if line[8] == refernece:
                interactor_a_chr = line[0]
                interactor_a_pos = [int(line[1]) + 1, int(line[2])]
                interactor_a_gene = line[6].split(',')
                interactor_b_chr = line[3]
                interactor_b_pos = [int(line[4]) + 1, int(line[5])]
                interactor_b_gene = line[7].split(',')
                score_1 = line[-4]
                score_2 = line[-3]
                contact_freq = line[-2]

                b_pos = converter.convert_coordinate(
                    interactor_b_chr,
                    interactor_a_pos[0],
                    interactor_a_pos[1]
                )
                print(line)
