import fatez.lib as lib
from torch.utils.data import DataLoader
import fatez.tool.JSON as JSON
import fatez.model.mlp as mlp
import torch
import fatez.process.explainer as explainer
import fatez.process.fine_tuner as fine_tuner
import fatez.model.bert as bert
import fatez.model as model
import shap
import pandas as pd
import numpy as np
import torch.nn as nn
from fatez.tool import model_training
import fatez.process.position_embedder as pe
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp_param = {
    'd_model': 8,
    'n_hidden': 4,
    'n_class': 2,
    'device': device,
    'dtype': torch.float32,
}



out_gat = JSON.decode('D:\\Westlake\\pwk lab\\fatez\\gat_gradient\\nhead0_nhidden1_lr-3_epoch200\\out_gat\\199.js')
#out_gat = JSON.decode('../data/ignore/79.pt')
# bert_encoder = torch.load('../data/ignore/bert_encoder.model')
# gat_model = torch.load('../data/ignore/gat.model')
# fine_tuning = fine_tuner.Model(
#     gat = gat_model,
#     rep_embedder = pe.Skip(n_features = 100),
#     bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = 2)
# )
# fine_tuning = bert.Fine_Tune_Model(bert_encoder, n_class = 2)
#out_gat = torch.Tensor([sample for batch in out_gat for sample in batch])

out_gat = torch.tensor(out_gat)
print(out_gat.shape)
decision = mlp.Classifier(**mlp_param)
# Feed in all input data as background
explain = shap.GradientExplainer(decision, out_gat)
# Explain one specific input
shap_values = explain.shap_values(out_gat[:1])
print(shap_values)
print('Work Here')
# Feed in all input data as background
explain = explainer.Gradient(decision, out_gat)
# Explain one specific input
shap_values = explain.shap_values(out_gat[:1], return_variances=True)
print('Work Here 2')
