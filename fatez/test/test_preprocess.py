import fatez.process.preprocessor as pre
import os
import fatez.tool.sequence as seq
os.chdir("D:\\Westlake\\pwk lab\\fatez\\FateZ\\fatez\\test")
if __name__ == '__main__':
    peak_path = ('../data/human/filtered_PBMC/')
    rna_path = ('../data/human/filtered_PBMC/')
    gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'

    ### load data
    network = pre.Preprocessor(rna_path, peak_path, gff_path, data_type='paired')
    network.load_data()
    print(network.rna_mt)
    print(network.atac_mt)
    ### qc
    network.rna_qc()
    network.atac_qc()
    print(network.rna_mt)
    print(network.atac_mt)
    ### merge & fix peak regions
    atac_mt = network.merge_peak()
    ### pseudo cell
    ### load gene peak annotation
    ### calculate correlation between gene and peak
    ### select peak with top correlation
    ### construct grp with expressed genes and its tfs with motif enrichment in target promoter



