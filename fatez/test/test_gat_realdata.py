import numpy as np
import torch
import torch.nn as nn
import fatez.model as model
import fatez.model.gat as gat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner

# Ignoring warnings because of using LazyLinear
import warnings
warnings.filterwarnings('ignore')


k = 20000
top_k = 1000
n = 2
n_class = 2
nhead = None
d_model = 3
en_dim = 8


adj_mat = torch.randn(top_k, k)
sample = [torch.randn(k, d_model), adj_mat]
samples = [sample]*n
labels = torch.tensor([1]*n)
print('# Fake feat:', k)
print('# Sample:', len(samples))

# print('here')
print('Test plain GAT')

model_gat = gat.GAT(d_model = d_model, en_dim = en_dim, nhead = nhead,)

# Test GAT
out_gat = model_gat(samples)
out_gat = model_gat.activation(out_gat)
out_gat = model_gat.decision(out_gat)



# Need to make sure d_model is divisible by nhead
bert_encoder = bert.Encoder(
    d_model = en_dim,
    n_layer = 6,
    nhead = 8,
    dim_feedforward = 2,
)

test_model = fine_tuner.Model(
    gat = model_gat,
    bin_pro = model.Binning_Process(n_bin = 100),
    bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = n_class)
)
output = test_model(samples)


loss = nn.CrossEntropyLoss()(
    output, labels
)
loss.backward()
print('GAT CEL:', loss)

# loss = F.nll_loss(out_gat, labels)
# print(loss)



# print('Test sparse GAT')
# model_sgat = sgat.Spare_GAT(d_model = d_model, en_dim = en_dim, nhead = nhead,)
# out_sgat = model_sgat(samples)
#
# # Activation and Decision
# out_sgat = model_gat.activation(out_sgat)
# out_sgat = model_gat.decision(out_sgat)
#
# loss = nn.CrossEntropyLoss()(
#     out_sgat, labels
# )
# loss.backward()
# print('SGAT CEL:', loss)

# loss = F.nll_loss(out_sgat, labels)
# print(loss)
