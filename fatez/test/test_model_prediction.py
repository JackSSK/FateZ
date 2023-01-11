import numpy as np
import torch
import fatez.model as model
import fatez.model.gat as gat
import torch.nn as nn
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner
import fatez.process.preprocessor as pre
import pandas as pd
from torch.utils.data import DataLoader
import fatez.lib as lib
from fatez.tool import PreprocessIO
import fatez.model.mlp as mlp
from fatez.tool import model_testing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cluster_use =[1,4]
batch_size = 20
num_epoch = 5

matrix1 = PreprocessIO.input_csv_dict_df(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data/node/')
matrix2 = pd.read_csv(
    'D:\\Westlake\\pwk lab\\fatez\\hsc_unpaired_testing_data/edge_matrix.csv'
    ,index_col=0)
samples = []
for i in range(len(matrix1)):
    m1 = matrix1[list(matrix1.keys())[i]]
    m1 = torch.from_numpy(m1.to_numpy())
    m2 = torch.from_numpy(matrix2.to_numpy())
    m1 = m1.to(torch.float32)
    m2 = m2.to(torch.float32)
    m1 = m1.to(device)
    m2 = m2.to(device)
    samples.append([m1, m2])
labels = torch.from_numpy(np.repeat(range(len(cluster_use))
                                    ,len(matrix1)/len(cluster_use)))
labels = labels.long()
labels = labels.to(device)
test_model = model.Load('../data/ignore/unpaired_bert_encoder_4head_0111.model')
test_dataloader = DataLoader(
    lib.FateZ_Dataset(samples=samples, labels=labels),
    batch_size=batch_size,
    shuffle=True
)
for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    test_loss, test_acc = model_testing.testing(test_dataloader,
                                                test_model,
                                                nn.CrossEntropyLoss()
                                                , device=device)
    print(
        f"epoch: {epoch+1}, test_loss: {test_loss}, test accuracy: {test_acc}")