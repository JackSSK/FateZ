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
