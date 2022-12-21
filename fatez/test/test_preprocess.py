import fatez.process.preprocessor as pre
import os

os.chdir("D:\\Westlake\\pwk lab\\fatez\\FateZ\\fatez\\test")
if __name__ == '__main__':
    peak_path = ('E:\\data_tutorial_buenrostro\\data_tutorial_buenrostro\\all_buenrostro_bulk_peaks.h5ad')
    rna_path = ('D:\\Westlake\\pwk lab\\fatez\\hg19')
    gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
    network=pre.Preprocessor(rna_path,peak_path,gff_path)
    network.load_data()
    print(network.rna_mt)