# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:57:09 2023

@author: peiweike
"""
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# Assume X and Y are your data
X = pd.read_table('D:\\Westlake\\pwk lab\\fatez\\rebuild_baseline\\rebuild_x.txt',header=0,index_col=0)
Y = pd.read_table('D:\\Westlake\\pwk lab\\fatez\\rebuild_baseline\\rebuild_y.txt',header=0,index_col=0)
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).float()
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Create DataLoader
batch_size = 10
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=21820, out_features=5000),
            nn.ReLU(),
            nn.Linear(in_features=5000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=500, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=5000),
            nn.ReLU(),
            nn.Linear(in_features=5000, out_features=21820),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.LogSoftmax(dim=-1)(x)
        return x
    


model = Autoencoder()
model = Autoencoder().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # initial learning rate
scheduler = CosineAnnealingLR(optimizer, T_max=100)  # 100 epochs

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).float()
def correlation(tensor1, tensor2):
    mean_tensor1 = torch.mean(tensor1, dim=1, keepdim=True)
    mean_tensor2 = torch.mean(tensor2, dim=1, keepdim=True)
    std_tensor1 = torch.std(tensor1, dim=1, keepdim=True)
    std_tensor2 = torch.std(tensor2, dim=1, keepdim=True)

    # Calculate correlation coefficient
    correlation = torch.mean(
        (tensor1 - mean_tensor1) * (tensor2 - mean_tensor2), dim=1,
        keepdim=True) / (std_tensor1 * std_tensor2)

    return torch.mean((correlation))
epochs = 3000
for epoch in range(epochs):
    loss_epoch = []
    cor_epoch = []
    for batch in train_dataloader:
        X_batch, Y_batch = batch
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to device
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, nn.LogSoftmax(dim=-1)(Y_batch))
        cor = correlation(outputs,nn.LogSoftmax(dim=-1)(Y_batch))
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.item())
        cor_epoch.append(cor.item())
        #print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, cor: {cor}')
    scheduler.step()  # update learning rate

    print(f'Epoch: {epoch + 1}, Loss_epoch: {np.array(loss_epoch).mean()}, cor: {np.array(cor_epoch).mean()}')
torch.save(Autoencoder,'D:\\Westlake\\pwk lab\\fatez/autoencoder.pt')