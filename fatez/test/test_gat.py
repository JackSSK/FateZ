import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fatez.model.gat as gat
import fatez.model.sparse_gat as sgat

k = 20000
top_k = 1000
n = 70
n_class = 2
nhead = None
d_model = 3
en_dim = 2


adj_mat = torch.randn(top_k, k)
sample = [torch.randn(k, d_model), adj_mat]
samples = [sample]*n
labels = torch.tensor([1]*n)
print('# Fake feat:', k)
print('# Sample:', len(samples))

# print('here')
print('Test plain GAT')

model_gat = gat.GAT(d_model = d_model, en_dim = en_dim, nhead = nhead,)
out_gat = model_gat(samples)

# Activation and Decision
out_gat = model_gat.activation(out_gat)
out_gat = model_gat.decision(out_gat)


loss = nn.CrossEntropyLoss()(
    out_gat, labels
)
loss.backward()
print('GAT CEL:', loss)

# loss = F.nll_loss(out_gat, labels)
# print(loss)



print('Test sparse GAT')
model_sgat = sgat.Spare_GAT(d_model = d_model, en_dim = en_dim, nhead = nhead,)
out_sgat = model_sgat(samples)

# Activation and Decision
out_sgat = model_gat.activation(out_sgat)
out_sgat = model_gat.decision(out_sgat)

loss = nn.CrossEntropyLoss()(
    out_sgat, labels
)
loss.backward()
print('SGAT CEL:', loss)

# loss = F.nll_loss(out_sgat, labels)
# print(loss)
