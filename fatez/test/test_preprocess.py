import fatez.process.preprocessor as pre
import os
import time
import fatez.tool.sequence as seq
os.chdir("D:\\Westlake\\pwk lab\\fatez\\FateZ\\fatez\\test")
if __name__ == '__main__':
    peak_path = ('../data/mouse/filtered_feature_bc_matrix/')
    rna_path = ('../data/mouse/filtered_feature_bc_matrix/')
    gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
    tf_db_path = 'E:\\public/TF_target_tss1500.txt'

    ### load data
    network = pre.Preprocessor(rna_path, peak_path, gff_path,tf_db_path, data_type='paired')
    network.load_data()
    #print(network.rna_mt)
    #print(network.atac_mt)
    #print(network.tf_target_db.keys())
    ### qc
    network.rna_qc(rna_min_genes=5,rna_min_cells=250,rna_max_cells=2500)
    network.atac_qc(atac_min_features=5,)
    #print(network.rna_mt)
    #print(network.atac_mt)
    #print(network.peak_count.keys())
    ### merge & fix peak regions

    #network.merge_peak() wtf????
    #print(network.peak_count.keys())

    ### pseudo cell

    network.make_pseudo_networks(data_type='paired')
    #print(network.pseudo_network)

    ### load gene peak annotation

    network.annotate_peaks()
    #print(network.peak_annotations.keys())

    ### calculate correlation between gene and peak
    ### select peak with top correlation
    t1 = time.time()
    network.cal_peak_gene_cor()
    t2 = time.time()
    print(t2-t1)
    a1=network.generate_grp()
    print(a1[1])
    ### construct grp with expressed genes and its tfs with motif enrichment in target promoter



