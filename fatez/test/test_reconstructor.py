import fatez.lib.grn as grn
import fatez.tool.mex as mex
import fatez.tool.JSON as JSON
import fatez.process.grn_reconstructor as grn_recon
from pkg_resources import resource_filename

# data can be downloaded here:
# https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-3-k-1-standard-2-0-0
# Filtered feature barcode matrix MEX (DIR)	<- download, gunzip it and then extract files

template_grn = grn.GRN()
template_grn.load(load_path = '../data/human/template.js.gz')

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
    v.save('../data/sample_grn.js.gz')
