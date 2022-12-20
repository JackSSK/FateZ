import fatez.tool as tool
from Bio import SeqIO


class Reader(tool.Reader_Template):
    """
    Load sequence file (fasta or anything else)
    """
    def get_fasta(self):
        fa_seq = {}
        for seq_record in SeqIO.parse(self.path, "fasta"):
            fa_seq[seq_record.id] = seq_record.seq
        return fa_seq
