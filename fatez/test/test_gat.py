import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fatez.model.gat as gat
import fatez.model.sparse_gat as sgat

k = 20000
n = 1

sample = [torch.randn(k, 1), torch.randn(k, k)]
samples = [sample]*n
print('# Fake feat:', k)
print('# Sample:', len(samples))

# print('here')
print('Test plain GAT')
labels = torch.tensor([[1]])
model_gat = gat.GAT(in_dim = 1,)
out_gat = model_gat(samples)
out_gat = F.softmax(out_gat, dim = -1)
print(out_gat)
print('\n')
print('Test sparse GAT')
model_sgat = sgat.Spare_GAT(in_dim = 1)
out_sgat = model_sgat(samples)
out_sgat = F.softmax(out_sgat, dim = -1)
print(out_sgat)
# linear = nn.Linear(8, 4)
# out = F.softmax(linear(out), dim = -1)

# loss = F.nll_loss(out, label)
# print(out, loss)
