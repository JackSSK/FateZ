import fatez.process.preprocessor as pre
import os
import fatez.tool.sequence as seq
os.chdir("D:\\Westlake\\pwk lab\\fatez\\FateZ\\fatez\\test")
if __name__ == '__main__':
    peak_path = ('../data/human/filtered_PBMC/')
    rna_path = ('../data/human/filtered_PBMC/')
    gff_path = '../data/human/gencode.v42.basic.annotation.gff3.gz'

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
    #print(network.peak_count.keys())
    ### merge & fix peak regions
    network.merge_peak()
    #print(network.peak_count.keys())
    ### pseudo cell
    network.make_pseudo_networks(data_type='paired')
    print(network.pseudo_network)
    ### load gene peak annotation
    network.annotate_peaks()
    #print(network.peak_annotations.keys())
    ### calculate correlation between gene and peak
    ### select peak with top correlation
    network.cal_peak_gene_cor()
    print(network.peak_gene_links)
    ### construct grp with expressed genes and its tfs with motif enrichment in target promoter



