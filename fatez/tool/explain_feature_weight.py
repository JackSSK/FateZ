import pandas as pd
import numpy as np
import torch
import torch.nn.functional as torch_F

def rank_explain_weight(explain_weight):

    ### use softmax to normalize all features, then sum it
    all_fea_weight = []
    for i in range(explain_weight.shape[1]):
        fea = torch.from_numpy(explain_weight[:,i].astype(np.float32))
        scores = torch_F.softmax(fea, dim=-1)
        fea_weight = scores.numpy()
        all_fea_weight.append(fea_weight)
    all_fea_weight = np.array(all_fea_weight)
    all_fea_gene = all_fea_weight.sum(axis=0)

    ### rank gene
    gene_rank = pd.Series(all_fea_gene,
                          index= list(range(len(all_fea_gene))))
    gene_rank = gene_rank.sort_values(ascending=False)
    gene_rank = pd.Series(list(range(len(all_fea_gene))),
                          index= gene_rank.index)
    gene_rank = gene_rank.sort_index()

    return gene_rank