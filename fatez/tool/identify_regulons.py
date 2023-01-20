import pandas as pd
import numpy as np
import torch
import torch.nn.functional as torch_F
import fatez.process.explainer as explainer
class Regulon():

    def __init__(self,feature_mt):
        self.feature_mt = feature_mt
        self.gene_rank = {}


    def explain_model(self,model_use,batch_size:int):

        for i in range(len(self.feature_mt)):

            ### calculate shapley value
            explain = explainer.Gradient(model_use, self.feature_mt[i])
            shap_values = explain.shap_values(self.feature_mt[i],
                                              return_variances=True)

            self.gene_rank[i] = {}
            for j in range(batch_size):
                # print(shap_values[j])
                m1 = shap_values[j]
                explain_weight = np.matrix(m1[0][0][0])
                gene_rank = self.__rank_shapley_importance(
                    explain_weight)
                self.gene_rank[i][j] = gene_rank

    def sum_regulon_count(self):
        feature_num = len(self.gene_rank[0][0])
        regulon_count = pd.Series([0] * feature_num)
        for i in self.gene_rank:
            for j in self.gene_rank[i]:
                regulon_count += self.gene_rank[i][j]
        return regulon_count




    def get_top_regulon_count(self,top_regulon_num:int = 20):
        ### count top regulons in each sample, then summarize the freuqency
        top_regulon_count = {}
        for i in self.gene_rank:
            for j in self.gene_rank[i]:
                top_regulon = self.gene_rank[i][j].sort_values[0:top_regulon_num]
                for k in top_regulon.index:
                    if k in top_regulon_count.keys():
                        top_regulon_count[k] = top_regulon_count[k]\
                                               +top_regulon[k]
                    else:
                        top_regulon_count[k] = top_regulon[k]
        return top_regulon_count

    def __rank_shapley_importance(self,explain_weight):

        ### use softmax to normalize all features, then sum it
        all_fea_weight = []
        for i in range(explain_weight.shape[1]):
            fea = torch.from_numpy(explain_weight[:,i].astype(np.float32))
            scores = torch_F.softmax(fea.T, dim=-1)
            fea_weight = scores.numpy()
            all_fea_weight.append(fea_weight)
        all_fea_weight = np.array(all_fea_weight)
        all_fea_gene = all_fea_weight.sum(axis=0)
        all_fea_gene = all_fea_gene[0, :]

        ### rank gene
        gene_rank = pd.Series(all_fea_gene,
                              index= list(range(len(all_fea_gene))))
        gene_rank = gene_rank.sort_values(ascending=False)
        gene_rank = pd.Series(list(range(len(all_fea_gene))),
                              index= gene_rank.index)
        gene_rank = gene_rank.sort_index()
        ### index is gene, value is count

        return gene_rank

