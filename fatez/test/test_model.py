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
import fatez.process.pre_trainer as pre_trainer
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
    print('Last GAT CEL:', loss, '\n')

    gat_explain = model_gat.explain(input[0][0], input[1][0])
    # print(gat_explain)
    explain = shap.GradientExplainer(decision, out_gat)
    shap_values = explain.shap_values(out_gat)
    # print(shap_values)
    explain = explainer.Gradient(decision, out_gat)
    shap_values = explain.shap_values(out_gat, return_variances = True)
    # print(shap_values)

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

def test_model(train_dataloader, n_bin, n_class, gat_model, masker, bert_encoder):
    print('Testing Model')
    pre_training = pre_trainer.Model(
        gat = gat_model,
        masker = masker,
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = bert.Pre_Train_Model(bert_encoder, n_bin = n_bin)
    )
    for input, _ in train_dataloader:
        output = pre_training(input[0], input[1])
        gat_out = pre_training.get_gat_output(input[0], input[1])
        loss = nn.CrossEntropyLoss()(
            output, gat_out
        )
        loss.backward()
    print('Last Pre Trainer CEL:', loss, '\n')

    fine_tuning = fine_tuner.Model(
        gat = gat_model,
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = bert.Fine_Tune_Model(
            bert_encoder,
            n_hidden = 2,
            n_class = n_class
        )
    )
    for input, label in train_dataloader:
        output = fine_tuning(input[0], input[1])
        loss = nn.CrossEntropyLoss()(
            output, label
        )
        loss.backward()

    gat_explain = gat_model.explain(input[0][0], input[1][0])
    explain = explainer.Gradient(fine_tuning, input)
    shap_values = explain.shap_values(input, return_variances = True)
    print(shap_values)
    print('Last Fine Tuner CEL:', loss, '\n')
    return bert_encoder

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
    masker_ratio = 0.5
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
        'dim_feedforward': gat_param['en_dim'],
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

    # temp = test_sparse_gat(
    #     train_dataloader,
    #     gat_param = gat_param,
    #     mlp_param = mlp_param
    # )
    # model.Save(temp, '../data/ignore/gat.model')
    #
    # temp = test_gat(
    #     train_dataloader,
    #     gat_param = gat_param,
    #     mlp_param = mlp_param
    # )
    # model.Save(temp, '../data/ignore/gat.model')

    encoder = test_model(
        train_dataloader, n_bin, n_class,
        gat_model = gat.GAT(**gat_param),
        masker = model.Masker(ratio = masker_ratio),
        bert_encoder = bert.Encoder(**bert_encoder_param),
    )
    model.Save(encoder, '../data/ignore/bert_encoder.model')


    # Test Loading Model
    # Saving Fine Tune Model should be fine
    test = bert.Fine_Tune_Model(encoder, n_class = 2)
    model.Save(test, '../data/ignore/b.model')
    fine_tuning = fine_tuner.Model(
        gat = model.Load('../data/ignore/gat.model'),
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = model.Load('../data/ignore/b.model')
    )
    # Using data loader now
    for input, label in train_dataloader:
        output = fine_tuning(input[0], input[1])
        loss = nn.CrossEntropyLoss()(
            output, label
        )
        loss.backward()
