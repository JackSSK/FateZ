import fatez.lib.grn as grn
import fatez.lib.template_grn as tgrn
import fatez.tool.mex as mex
import fatez.tool.JSON as JSON
import fatez.process.grn_reconstructor as grn_recon
from pkg_resources import resource_filename
import os

os.chdir("D:\\FateZ\\fatez\\test")
if __name__ == '__main__':
    gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
    template_id = 'GRCm38_template'
    out_path = '../data/mouse/template.js.gz'

    template = tgrn.Template_GRN(id = template_id)
    template.load_genes(gff_path = gff_path)
    template.load_cres()
    template.get_genetic_regions()
    template.save(save_path = out_path)


    template_grn = grn.GRN()
    template_grn.load(load_path = '../data/mouse/template.js.gz')

    reconstructor = grn_recon.Reconstruct(template_grn = template_grn)
    data = mex.Reader(
        matrix_path = '../data/human/filtered_PBMC/matrix.mtx.gz',
        features_path = '../data/human/filtered_PBMC/features.tsv.gz',
        barcodes_path = '../data/human/filtered_PBMC/barcodes.tsv.gz'
    )
    grns = reconstructor.paired_multi_MEX(
        data, group_barcodes = ['AAACAGCCAAATATCC-1']
    )
    for k,v in grns.items():
        v.save('../data/sample_grn.js')
