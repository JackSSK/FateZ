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
only_tf = True

X = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_hep/gene_cell.txt',header=0,index_col=0)
Y = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_hep/peak_cell.txt',header=0,index_col=0)
p1 = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_endo/gene_cell.txt',header=0,index_col=0)
p2 = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE205117_endo/gene_cell.txt',header=0,index_col=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 10


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=1103, out_features=250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=50, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=1103),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.LogSoftmax(dim=-1)(x)
        return x

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        x = nn.LogSoftmax(dim=-1)(x)
        return x
    




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

class rebuild_baseline():
    def __init__(self,
                train_X = None,
                train_Y = None,
                predict_X = None,
                predict_Y = None,
                model_use = 'AE'):
        self.train_X = train_X
        self.train_Y = train_Y
        self.predict_X = predict_X
        self.predict_Y = predict_Y
        self.model_use = model_use
        self.train_los = []
        self.train_cor = []
        self.pred_cor = []
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.pred_loader = None
    def pp(self):
        ### training data
        X_train, X_test, Y_train, Y_test = train_test_split(self.train_X.T,
                                                            self.train_Y .T,
                                                            test_size=0.2,
                                                            random_state=42)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_train_tensor = torch.from_numpy(X_train).float()
        Y_train_tensor = torch.from_numpy(Y_train).float()
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        X_test_tensor = torch.from_numpy(X_test).float()
        Y_test_tensor = torch.from_numpy(Y_test).float()

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False)
        ### testing data
        predict_X = np.array(self.predict_X.T)
        predict_Y = np.array(self.predict_Y.T)
        predict_X_tensor = torch.from_numpy(predict_X).float()
        predict_Y_tensor = torch.from_numpy(predict_Y).float()
        pred_dataset = TensorDataset(predict_X_tensor, predict_Y_tensor)

        self.pred_loader = DataLoader(pred_dataset, batch_size=batch_size,
                                      shuffle=False)

    def train(self,epochs=100):
        if  self.model_use == 'AE':
            self.model = Autoencoder().to(device)
        elif self.model_use == 'DNN':
            self.model = RegressionModel(1103,128,1103).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(),
                                   lr=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)


        for epoch in range(epochs):
            loss_epoch = []
            cor_epoch = []
            for batch in self.train_loader:
                X_batch, Y_batch = batch
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to device
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, nn.LogSoftmax(dim=-1)(Y_batch))
                cor = correlation(outputs,nn.LogSoftmax(dim=-1)(Y_batch))
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.item())
                cor_epoch.append(cor.item())
                #print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, cor: {cor}')
            scheduler.step()  # update learning rate
            self.train_los.append(np.array(loss_epoch).mean())
            self.train_cor.append(np.array(cor_epoch).mean())
            print(f'Epoch: {epoch + 1}, Loss_epoch: {np.array(loss_epoch).mean()}, cor: {np.array(cor_epoch).mean()}')
    def test(self):
        self.model.eval()
        loss_epoch = []
        cor_epoch = []
        for batch in self.test_loader:
            X_batch, Y_batch = batch
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(
                device)  # Move data to device
            outputs = self.model(X_batch)
            criterion = nn.L1Loss()
            cor = correlation(outputs, nn.LogSoftmax(dim=-1)(Y_batch))
            loss = criterion(outputs, nn.LogSoftmax(dim=-1)(Y_batch))
            loss_epoch.append(loss.item())
            cor_epoch.append(cor.item())
        print('----testing----')
        print(
            f'Loss_epoch: {np.array(loss_epoch).mean()}, cor: {np.array(cor_epoch).mean()}')
    def predict(self):
        cor_epoch = []
        print('predition')
        for batch in self.pred_loader:
            X_batch, Y_batch = batch
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(
                device)
            outputs = self.model(X_batch)
            cor = correlation(outputs, nn.LogSoftmax(dim=-1)(Y_batch))
            cor_epoch.append(cor.item())
        print(
            f'cor: {np.array(cor_epoch).mean()}')

ae = rebuild_baseline(X,Y,p1,p2)
ae.pp()
ae.train(epochs=500)
ae.test()



dnn = rebuild_baseline(X,Y,X,Y,model_use='DNN')
dnn.pp()
dnn.train(epochs=500)
dnn.test()

print('--------------dnn result!!!')
print('training cor all mean:',np.array(dnn.train_cor).mean())
dnn.predict()
print('--------------ae result!!!')
print('training cor all mean:',np.array(ae.train_cor).mean())
ae.predict()