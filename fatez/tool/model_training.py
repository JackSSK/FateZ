"""
Depreciated

import torch
from torch.nn import L1Loss

def training(dataloader, model_gat, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_num = 1
    model.train()
    train_loss, correct = 0, 0
    out_gat_data = list()
    for x,y in dataloader:
        optimizer.zero_grad()
        out_gat = model_gat(x[0], x[1])
        output = model(x[0].to(device), x[1].to(device))
        for ele in out_gat.detach().tolist(): out_gat_data.append(ele)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        acc = (output.argmax(1)==y).type(torch.float).sum().item()
        print(f"batch: {batch_num} loss: {loss} accuracy:{acc/num_batches}")
        batch_num += 1
        train_loss += loss
        correct += acc
    train_loss /= num_batches
    correct /= size
    return out_gat_data,train_loss,correct

def pre_training(dataloader, model, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_num = 1
    model.train()
    train_loss = 0
    for x, _ in dataloader:
        optimizer.zero_grad()
        node=x[0].to(device)
        edge=x[1].to(device)
        output, output_adj = model(node, edge)
        loss = L1Loss()(
            output, torch.split(node, output.shape[1], dim=1)[0]
        )
        loss.backward()
        optimizer.step()
        print(f"batch: {batch_num} loss: {loss}")
        batch_num += 1
        train_loss += loss
    train_loss /= num_batches
    return train_loss

def testing(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(x[0].to(device), x[1].to(device))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss,correct
"""
