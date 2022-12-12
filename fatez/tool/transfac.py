#!/usr/bin/env python3
"""
TRANSFAC file related tools

author: jy, nkmtmsys
"""
import re
import fatez.tool as tool



class Reader(tool.Reader_Template):
    """
    Load in Transfac database for determining whether a gene is Transcription
    Factor(TF).
    Will need to be override if TF database in different format.
    """
    def get_tfs(self):
        """
        Obtain TF information.
        """
        tfs = dict()
        while(True):
            line = self.file.readline().strip()
            if line == '':
                break
            elif line[:1] == '#':
                continue
            content = line.split('\t')
            symbols = content[3].split(';')
            ids = content[4].split(';')
            for id in ids:
                if id not in tfs:
                    tfs[id] = symbols
        self.close()
        return tfs

# Example
# if __name__ == '__main__':
#     reader = Reader('../data/mouse/Transfac201803_Mm_MotifTFsF.txt')
#     tfs = reader.get_tfs()
