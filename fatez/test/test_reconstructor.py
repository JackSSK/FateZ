import fatez.lib.grn as grn
import fatez.lib.template_grn as tgrn
import fatez.tool.mex as mex
import fatez.tool.JSON as JSON
import fatez.process.grn_reconstructor as grn_recon
from pkg_resources import resource_filename

# data can be downloaded here:
# https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-3-k-1-standard-2-0-0
# Filtered feature barcode matrix MEX (DIR)	<- download, gunzip it and then extract files


# Main function for testing purpose
if __name__ == '__main__':
    gff_path = '../data/mouse/gencode.vM25.basic.annotation.gff3.gz'
    template_id = 'GRCm38_template'
    out_path = '../data/mouse/template.js.gz'

    template = tgrn.Template_GRN(id = template_id)
    template.load_genes(gff_path = gff_path)
    template.load_cres()
    template.get_genetic_regions()
    template.save(save_path = out_path)

    gff_path = '../data/human/gencode.v42.basic.annotation.gff3.gz'
    template_id = 'GRCh38_template'
    out_path = '../data/human/template.js.gz'

    template = tgrn.Template_GRN(id = template_id)
    template.load_genes(gff_path = gff_path)
    template.load_cres()
    template.get_genetic_regions()
    template.save(save_path = out_path)


    template_grn = grn.GRN()
    template_grn.load(load_path = '../data/human/template.js.gz')

    reconstructor = grn_recon.Reconstruct(template_grn = template_grn)
    data = mex.Reader(
        matrix_path = '../data/human/filtered_PBMC/matrix.mtx.gz',
        features_path = '../data/human/filtered_PBMC/features.tsv.gz',
        barcodes_path = '../data/human/filtered_PBMC/barcodes.tsv.gz'
    )

    ####################################################################
    peak_annotations = reconstructor.annotate_peaks(data.features)
    # JSON.encode(peak_annotations, 'a.js')

    ####################################################################

    grns = reconstructor.paired_multi_MEX(
        data, group_barcodes = ['AAACAGCCAAATATCC-1']
    )
    for k,v in grns.items():
        v.save('../data/sample_grn.js.gz')
