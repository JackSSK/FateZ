#!/usr/bin/env python3
"""
Basic tools for handling files.

author: jy, nkmtmsys
"""
import re
import gzip


class Error(Exception):
    pass



class Reader_Template:
    """
    Template for Reader object.
    """
    def __init__(self, path:str = None):
        """
        :param path: <str Default = None>
            Path to the input file.
        """
        self.path = path
        self.file = self.load()

    # Load in file
    def load(self,):
        # Open as .gz file
        if re.search(r'\.gz$', self.path):
            return gzip.open(self.path, 'rt', encoding='utf-8')
        # Open directly
        else:
            return open(self.path, 'r')

    # Close file reading
    def close(self):
        self.file.close()

    # For iteration
    def __iter__(self):
        return self

    # Need to be override based on need
    def __next__(self):
        return self

    def get(self, coordinate:int = None, sep:str = '\t'):
        """
        Read in line at given coordinate.

        :param coordinate: <int Default = None>
            Path to the input file.

        :param sep: <str Default = '\t'>
            Seperate character within a line.

        :return: <class 'list'>
        """
        self.file.seek(coordinate)
        info = self.file.readline()
        if sep is not None:
            return info.split('\t')
        else:
            return list(info)
