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
            motif = content[0]
            symbols = content[3].split(';')
            ids = content[4].split(';')
            for id in ids:
                if id not in tfs:
                    tfs[id] = {'names':symbols, 'motif':[motif],}
                else:
                    assert motif not in tfs[id]['motif']
                    tfs[id]['names']=list(set(symbols) & set(tfs[id]['names']))
                    tfs[id]['motif'].append(motif)
        self.close()
        return tfs

    def get_motifs(self):
        """
        Obtain TF information.
        """
        motifs = dict()
        while(True):
            line = self.file.readline().strip()
            if line == '':
                break
            elif line[:1] == '#':
                continue
            content = line.split('\t')
            motif = content[0]
            symbols = content[3].split(';')
            ids = content[4].split(';')
            motifs[motif] = {'names':symbols,'id':ids}
        self.close()
        return motifs
# # Example
# if __name__ == '__main__':
#     tfs = Reader('../data/mouse/Transfac201803_MotifTFsF.txt.gz').get_tfs()
#     print(tfs)
