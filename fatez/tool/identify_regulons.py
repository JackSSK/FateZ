import pandas as pd
import numpy as np
import torch
import torch.nn.functional as torch_F
import fatez.process.explainer as explainer


class Regulon():
    """
    ?
    """
    def __init__(self,feature_mt,grp):
        self.feature_mt = feature_mt
        self.gene_rank = {}
        self.tf_names = []
        self.gene_importance_values = []
        self.grp = grp

    def explain_model(self,bert_model,batch_size:int,filter_tf=True):
        gene_importance_values_all = np.array([0]*self.grp.shape[1])
        gene_importance_values_all = np.float64(gene_importance_values_all)
        for i in range(len(self.feature_mt)):

            ### calculate shapley value
            explain = explainer.Gradient(bert_model, self.feature_mt[i])
            shap_values = explain.shap_values(self.feature_mt[i],
                                              return_variances=True)

            self.gene_rank[i] = {}
            for j in range(len(shap_values)):
                # print(shap_values[j])
                m1 = shap_values[j]
                explain_weight = np.matrix(m1[0][0][0])
                gene_rank = self.__rank_shapley_importance(
                    explain_weight,self.grp,filter_tf)
                self.gene_rank[i][j] = gene_rank

                ### total importance
                gene_importance_values = shap_values[0][0][0].sum(2)
                gene_importance_values_all += gene_importance_values[0]
        ###

        self.gene_importance_values = gene_importance_values_all

    def explain_grp(self,gat_model,top_grp_num:int = 20,ignore_tf:str=''):
        ### explain grp for analyzing GRP importances
        grp_importance = gat_model.explain(
            torch.ones_like(self.feature_mt[0][0][0]),
            torch.ones_like(self.feature_mt[0][1][0])
        )
        for i in range(len(self.feature_mt)):
            importance = gat_model.explain(self.feature_mt[i][0][0],
                              self.feature_mt[i][1][0])
            grp_importance += importance


        ###
        grp_ones= gat_model.explain(
            torch.ones_like(self.feature_mt[0][0][0]),
            torch.ones_like(self.feature_mt[0][1][0])
        )
        tf_importance = np.array(torch.matmul(
            grp_ones, torch.Tensor(self.gene_importance_values)
        ))
        grp_importance = (grp_importance.T*tf_importance).T
        tf_all = []
        gene_all = []
        for i in range(top_grp_num):
            grp_idx = np.unravel_index(np.argmax(grp_importance),
                                    grp_importance.shape)
            grp_importance[grp_idx] = 0
            tf = self.grp.index[grp_idx[0]]
            gene = self.grp.columns[grp_idx[1]]
            if tf == ignore_tf:
                continue
            tf_all.append(tf)
            gene_all.append(gene)
        grp_out = pd.DataFrame([tf_all,gene_all],index=['source','target'])
        return grp_out.T

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
                top_regulon = self.gene_rank[i][j]
                top_regulon = top_regulon.sort_values()
                top_regulon = top_regulon[0:top_regulon_num]
                for k in top_regulon.index:
                    if k in top_regulon_count.keys():
                        top_regulon_count[k] = top_regulon_count[k]+1
                    else:
                        top_regulon_count[k] = 1
        return top_regulon_count

    def __rank_shapley_importance(self,explain_weight,grp,filter_tf = True):

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

        ### retina tfs
        if filter_tf:
            filter = grp.columns.isin(grp.index)
            self.tf_names = grp.columns[filter]
            all_fea_gene = all_fea_gene[filter]
        else:
            self.tf_names = grp.columns
        ### rank gene
        ### index is gene, value is count
        ### rank count, high count genes have low rank
        gene_rank = pd.Series(all_fea_gene,
                              index= list(range(len(all_fea_gene))))
        gene_rank = gene_rank.sort_values(ascending=False)
        gene_rank = pd.Series(list(range(len(all_fea_gene))),
                              index= gene_rank.index)
        gene_rank = gene_rank.sort_index()


        return gene_rank
