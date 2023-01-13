#!/usr/bin/env python3
"""
Note:

I suppose you want to explain BERT Fine Tune Model, not the decision model

"""

import fatez.tool.JSON as JSON
import fatez.model.mlp as mlp
import torch
import fatez.process.explainer as explainer
import fatez.process.fine_tuner as fine_tuner
import fatez.model.bert as bert
import fatez.model as model
import shap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp_param = {
    'd_model': 8,
    'n_hidden': 4,
    'n_class': 2,
    'device': device,
    'dtype': torch.float32,
}

# out_gat = JSON.decode('D:\\Westlake\\pwk lab\\fatez\\gat_gradient\\nhead0_nhidden1_weight_decay0\\out_gat\\99.pt')
out_gat = JSON.decode('../data/ignore/79.pt')
bert_encoder = torch.load('../data/ignore/bert_encoder.model')
gat_model = torch.load('../data/ignore/gat.model')
fine_tuning = fine_tuner.Model(
    gat = gat_model,
    bin_pro = model.Binning_Process(n_bin = 100),
    bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = 2)
)
fine_tuning = bert.Fine_Tune_Model(bert_encoder, n_class = 2)
out_gat = torch.Tensor([sample for batch in out_gat for sample in batch])
print(out_gat.size())

decision = mlp.Classifier(**mlp_param)
# Feed in all input data as background
explain = shap.GradientExplainer(decision, out_gat)
# Explain one specific input
shap_values = explain.shap_values(out_gat[:1])
print('Work Here')
# Feed in all input data as background
explain = explainer.Gradient(decision, out_gat)
# Explain one specific input
shap_values = explain.shap_values(out_gat[:1], return_variances=True)
print('Work Here 2')
