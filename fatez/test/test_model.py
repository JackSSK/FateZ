import os
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fatez.lib as lib
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.bert as bert
import fatez.model.gat as gat
import fatez.model.sparse_gat as sgat
import fatez.process.explainer as explainer
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer


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

def test_model(
    train_dataloader,
    n_bin,
    n_class,
    n_gene,
    gat_model,
    masker,
    bert_encoder
    ):
    print('Testing Model')
    pre_train_criteria = nn.L1Loss()
    fine_tune_criteria = nn.CrossEntropyLoss()

    # Pre-train part
    pre_training = pre_trainer.Model(
        gat = gat_model,
        masker = masker,
        bin_pro = model.Binning_Process(n_bin = n_bin),
        bert_model = bert.Pre_Train_Model(
            bert_encoder,
            n_dim_node = gat_model.d_model,
            n_dim_adj = n_gene,
        )
    )
    for input, _ in train_dataloader:
        print(f'Shape of fea mat:{input[0].shape}',
            f'Shape of adj mat:{input[1].shape}')
        output_node, output_adj = pre_training(input[0], input[1])
        # gat_out = pre_training.get_gat_output(input[0], input[1])
        loss_node = pre_train_criteria(
            output_node,
            torch.split(input[0], output_node.shape[1] , dim = 1)[0]
        )

        if output_adj is not None:
            loss_adj = pre_train_criteria(output_adj, input[1])
            loss = loss_node + loss_adj
        else:
            loss = loss_node

        loss.backward()
    print('Last Pre Trainer CEL:', loss, '\n')

    # Fine tune part
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
        loss = fine_tune_criteria(output, label)
        loss.backward()

    gat_explain = gat_model.explain(input[0][0], input[1][0])
    explain = explainer.Gradient(fine_tuning, input)
    shap_values = explain.shap_values(input, return_variances = True)
    # print(shap_values)
    print('Last Fine Tuner CEL:', loss, '\n')
    return bert_encoder

if __name__ == '__main__':
    # Create the cache directory if not present
    if not os.path.exists('../data/ignore'):
        os.makedirs('../data/ignore')

    # Parameters
    k = 1051
    top_k = 452
    n_sample = 10
    batch_size = 4
    n_class = 4
    n_bin = 100
    masker_ratio = 0.5
    gat_param = {
        'd_model': 2,   # Feature dim
        'en_dim': 3,
        'n_hidden': 2,
        'nhead': 0,
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
        'nhead': 3,
        'dim_feedforward': gat_param['en_dim'],
        'dtype': gat_param['dtype'],
    }

    # Generate Fake data
    samples = [
        [
            torch.randn(k, gat_param['d_model'], dtype = torch.float32),
            torch.randn(top_k, k, dtype = torch.float32)
        ] for i in range(n_sample - 1)
    ]
    # To test data loader not messing up exp data and adj mats
    samples.append(
        [
            torch.ones(k, gat_param['d_model'], dtype = torch.float32),
            torch.ones(top_k, k, dtype = torch.float32)
        ]
    )
    labels = torch.empty(n_sample, dtype = torch.long).random_(n_class)

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
    temp = test_gat(
        train_dataloader,
        gat_param = gat_param,
        mlp_param = mlp_param
    )
    model.Save(temp, '../data/ignore/gat.model')

    encoder = test_model(
        train_dataloader = train_dataloader,
        n_bin = n_bin,
        n_class = n_class,
        # Gene number, if not None then reconstructing adj mat.
        n_gene = None,
        # n_gene = k,
        gat_model = gat.GAT(**gat_param),
        masker = model.Masker(ratio = masker_ratio),
        bert_encoder = bert.Encoder(**bert_encoder_param),
    )
    model.Save(encoder, '../data/ignore/bert_encoder.model')

    #
    # # Test Loading Model
    # # Saving Fine Tune Model should be fine
    # test = bert.Fine_Tune_Model(encoder, n_class = 2)
    # model.Save(test, '../data/ignore/b.model')
    # fine_tuning = fine_tuner.Model(
    #     gat = model.Load('../data/ignore/gat.model'),
    #     bin_pro = model.Binning_Process(n_bin = n_bin),
    #     bert_model = model.Load('../data/ignore/b.model')
    # )
    # # Using data loader now
    # for input, label in train_dataloader:
    #     output = fine_tuning(input[0], input[1])
    #     loss = nn.CrossEntropyLoss()(
    #         output, label
    #     )
    #     loss.backward()
