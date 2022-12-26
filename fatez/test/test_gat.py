import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fatez.model.gat as gat
import fatez.model.sparse_gat as sgat

k = 4
n = 1
n_class = 2
en_dim = 2

sample = [torch.randn(k, 1), torch.randn(k, k)]
samples = [sample]*n
print('# Fake feat:', k)
print('# Sample:', len(samples))

# print('here')
print('Test plain GAT')
labels = torch.tensor([[1, 0]])
model_gat = gat.GAT(in_dim = 1, en_dim = en_dim)
out_gat = model_gat(samples)

# Activation and Decision
out_gat = model_gat.activation(out_gat)
out_gat = model_gat.decision(out_gat)


print(out_gat, len(out_gat))
print('\n')


loss = nn.CrossEntropyLoss()(
    out_gat, labels
)
print(loss)

loss = F.nll_loss(out_gat, labels)
print(loss)



# print('Test sparse GAT')
# model_sgat = sgat.Spare_GAT(in_dim = 1)
# out_sgat = model_sgat(samples)
# # Activation Layer
# out_sgat = F.softmax(F.log_softmax(out_sgat, dim = -1), dim = -1)
# print(out_sgat)

# linear = nn.Linear(8, 4)
# out = F.softmax(linear(out), dim = -1)
#
# loss = F.nll_loss(out, label)
# print(out, loss)
