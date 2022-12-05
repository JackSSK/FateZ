#!/usr/bin/env python3
"""
Basic tools for handling files.

author: jy, nkmtmsys
"""

class Error(Exception):
    pass



class Reader_Template:
    """
    Template for Reader object.
    """
    def __init__(self, path:str = None):
        """
        :param path:str = None
        """
        self.path = path
        # Open as .gz file
        if re.search(r'\.gz$', self.path):
            self.file = gzip.open(self.path, 'rt', encoding='utf-8')
        # Open directly
        else:
            self.file = open(self.path, 'r')

    # Close file reading
    def close(self):
        self.file.close()

    # For iteration
    def __iter__(self):
        return self

    # Need to be override based on need
    def __next__(self):
        return self
