import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fatez.model as model
import fatez.model.gat as gat
import fatez.model.sparse_gat as sgat
import fatez.model.bert as bert
import fatez.process.fine_tuner as fine_tuner

# Ignoring warnings because of using LazyLinear
import warnings
warnings.filterwarnings('ignore')


def test_gat(input, label, gat_param):
    print('Testing plain GAT')
    model_gat = gat.GAT(**gat_param)
    out_gat = model_gat(input)
    out_gat = model_gat.activation(out_gat)
    out_gat = model_gat.decision(out_gat)
    loss = nn.CrossEntropyLoss()(
        out_gat, label
    )
    loss.backward()
    print('GAT CEL:', loss, '\n')
    return model_gat


def test_sparse_gat(input, label, gat_param):
    print('Testing sparse GAT')
    model_sgat = sgat.Spare_GAT(**gat_param)
    out_sgat = model_sgat(input)
    out_sgat = model_sgat.activation(out_sgat)
    out_sgat = model_sgat.decision(out_sgat)
    loss = nn.CrossEntropyLoss()(
        out_sgat, label
    )
    loss.backward()
    print('SGAT CEL:', loss, '\n')
    return model_sgat


def test_fine_tune(input, label, n_bin, n_class, gat_model, bert_encoder):
    print('Testing Fine-Tune Model')
    fine_tuning = fine_tuner.Model(
        gat = gat_model,
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = n_class)
    )
    output = fine_tuning(input)
    loss = nn.CrossEntropyLoss()(
        output, label
    )
    loss.backward()
    print('Fine Tuner CEL:', loss, '\n')
    return fine_tuning


def test_pre_train(input, mask, n_bin, n_class, gat_model, bert_encoder):
    print('Testing Pre-Train Model')
    fine_tuning = fine_tuner.Model(
        gat = gat_model,
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = bert.Fine_Tune_Model(bert_encoder, n_class = n_class)
    )
    output = fine_tuning(input)
    loss = nn.CrossEntropyLoss()(
        output, label
    )
    loss.backward()
    print('Pre-Trainer CEL:', loss, '\n')
    return fine_tuning


if __name__ == '__main__':
    # Create the cache directory if not present
    if not os.path.exists('../data/ignore'):
        os.makedirs('../data/ignore')

    # Parameters
    k = 5000
    top_k = 200
    n_sample = 4
    n_class = 4
    gat_param = {
        'd_model': 2,   # Feature dim
        'en_dim': 8,
        'nhead': None,
        'n_class': n_class,
        'dtype': torch.float64,
    }
    # Need to make sure d_model is divisible by nhead
    bert_encoder_param = {
        'd_model': gat_param['en_dim'],
        'n_layer': 6,
        'nhead': 8,
        'dim_feedforward': 3,
        'dtype': torch.float64,
    }
    n_bin = 100

    # Generate Fake data
    adj_mat = torch.randn(top_k, k, dtype = torch.float64)
    sample = [torch.randn(k, gat_param['d_model'], dtype = torch.float64), adj_mat]

    # FYI
    print(sample[0].size(), sample[1].size())

    input = [sample] * n_sample
    label = torch.tensor([1] * n_sample)
    print('Fake gene num:', k)
    print('Fake TF num:', top_k)
    print('Fake Sample Number:', len(input))
    print('Class Number:', n_class, '\n')

    temp = test_sparse_gat(input, label, gat_param = gat_param)
    model.Save(temp, '../data/ignore/gat.model')

    temp = test_gat(input, label, gat_param = gat_param)
    model.Save(temp, '../data/ignore/gat.model')

    temp = test_fine_tune(
        input, label, n_bin, n_class,
        gat_model = gat.GAT(**gat_param),
        bert_encoder = bert.Encoder(**bert_encoder_param),
    )
    model.Save(temp.bert_model.encoder, '../data/ignore/a.model')

    # Test Loading Model
    test_fine_tune(
        input, label, n_bin, n_class,
        gat_model = model.Load('../data/ignore/gat.model'),
        bert_encoder = model.Load('../data/ignore/a.model'),
    )
