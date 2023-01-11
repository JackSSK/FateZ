import os
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import fatez.tool.JSON as JSON
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gat as gat
import fatez.model.sparse_gat as sgat
import fatez.model.bert as bert
import fatez.lib as lib
import fatez.process.fine_tuner as fine_tuner
import fatez.process.explainer as explainer

# Ignoring warnings because of using LazyLinear
import warnings
warnings.filterwarnings('ignore')


def test_gat(train_dataloader, gat_param, mlp_param):
    print('Testing plain GAT')
    model_gat = gat.GAT(**gat_param)
    decision = mlp.Classifier(**mlp_param)
    # Using data loader now
    for input, label in train_dataloader:
        out_gat = model_gat(input[0], input[1])
        output = decision(out_gat)
        loss = nn.CrossEntropyLoss()(
            output, label
        )
        loss.backward()

    model_gat.explain(input[0][0], input[1][0])

    explain = shap.GradientExplainer(decision, out_gat)
    shap_values = explain.shap_values(out_gat)
    print(shap_values)
    explain = explainer.Gradient(decision, out_gat)
    shap_values = explain.shap_values(out_gat, return_variances = True)
    print(shap_values)
    print('Last GAT CEL:', loss, '\n')
    return model_gat


def test_sparse_gat(train_dataloader, gat_param, mlp_param):
    print('Testing sparse GAT')
    model_sgat = sgat.Spare_GAT(**gat_param)
    decision = mlp.Classifier(**mlp_param)
    # Using data loader now
    for input, label in train_dataloader:
        output = model_sgat(input[0], input[1])
        output = decision(output)
        loss = nn.CrossEntropyLoss()(
            output, label
        )
        loss.backward()

    model_sgat.explain(input[0][0], input[1][0])
    print('Last SGAT CEL:', loss, '\n')
    # model_sgat.explain()
    return model_sgat


def test_fine_tune(train_dataloader, n_bin, n_class, gat_model, bert_encoder):
    print('Testing Fine-Tune Model')
    fine_tuning = fine_tuner.Model(
        gat = gat_model,
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = n_class)
    )
    # Using data loader now
    for input, label in train_dataloader:
        output = fine_tuning(input[0], input[1])
        loss = nn.CrossEntropyLoss()(
            output, label
        )
        loss.backward()
    explain = explainer.Gradient(fine_tuning, input)
    shap_values = explain.shap_values(input, return_variances = True)
    print(shap_values)
    print('Last Fine Tuner CEL:', loss, '\n')
    return fine_tuning


def test_pre_train(input, mask, n_bin, n_class, gat_model, bert_encoder):
    print('Testing Pre-Train Model')
    fine_tuning = fine_tuner.Model(
        gat = gat_model,
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = n_class)
    )
    # Using data loader now
    for input, label in train_dataloader:
        output = fine_tuning(input[0], input[1])
        loss = nn.CrossEntropyLoss()(
            output, label
        )
        loss.backward()
    print('Last Pre-Trainer CEL:', loss, '\n')
    return fine_tuning


if __name__ == '__main__':
    # Create the cache directory if not present
    if not os.path.exists('../data/ignore'):
        os.makedirs('../data/ignore')

    # Parameters
    k = 10
    top_k = 4
    n_sample = 10
    batch_size = 1
    n_class = 4
    gat_param = {
        'd_model': 2,   # Feature dim
        'en_dim': 8,
        'n_hidden': 4,
        'nhead': 2,
        'device':'cpu',
        'dtype': torch.float32,
    }
    mlp_param = {
        'd_model':gat_param['en_dim'],
        'n_hidden': 4,
        'n_class':n_class,
        'device':gat_param['device'],
        'dtype':gat_param['dtype'],
    }

    # Need to make sure d_model is divisible by nhead
    bert_encoder_param = {
        'd_model': gat_param['en_dim'],
        'n_layer': 6,
        'nhead': 8,
        'dim_feedforward': 3,
        'dtype': torch.float32,
    }
    n_bin = 100

    # Generate Fake data
    sample = [
        torch.randn(k, gat_param['d_model'], dtype = torch.float32),
        torch.randn(top_k, k, dtype = torch.float32)
    ]
    # To test data loader not messing up exp data and adj mats
    one_sample = [
        torch.ones(k, gat_param['d_model'], dtype = torch.float32),
        torch.ones(top_k, k, dtype = torch.float32)
    ]
    samples = [sample] * (n_sample - 1)
    samples.append(one_sample)
    labels = torch.tensor([1] * n_sample)

    train_dataloader = DataLoader(
        lib.FateZ_Dataset(samples = samples, labels = labels),
        batch_size = batch_size,
        shuffle = True
    )

    print('Fake gene num:', k)
    print('Fake TF num:', top_k)
    print('Fake Sample Number:', n_sample)
    print('Batch Size:', batch_size)
    print('Class Number:', n_class, '\n')

    temp = test_sparse_gat(
        train_dataloader,
        gat_param = gat_param,
        mlp_param = mlp_param
    )
    model.Save(temp, '../data/ignore/gat.model')

    temp = test_gat(
        train_dataloader,
        gat_param = gat_param,
        mlp_param = mlp_param
    )
    model.Save(temp, '../data/ignore/gat.model')

    temp = test_fine_tune(
        train_dataloader, n_bin, n_class,
        gat_model = gat.GAT(**gat_param),
        bert_encoder = bert.Encoder(**bert_encoder_param),
    )
    model.Save(temp.bert_model.encoder, '../data/ignore/a.model')

    # Test Loading Model
    test_fine_tune(
        train_dataloader, n_bin, n_class,
        gat_model = model.Load('../data/ignore/gat.model'),
        bert_encoder = model.Load('../data/ignore/a.model'),
    )
