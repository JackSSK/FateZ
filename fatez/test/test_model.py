import os
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fatez.lib as lib
import fatez.tool.JSON as JSON
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.process.explainer as explainer
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
from pkg_resources import resource_filename

# Ignoring warnings because of using LazyLinear
import warnings
warnings.filterwarnings('ignore')


def test_gat(train_dataloader, config, factory_kwargs, mlp_param):
    # Build GAT Model accordingly
    gat_model = model.Set_GAT(config, factory_kwargs)
    decision = mlp.Classifier(**mlp_param, **factory_kwargs)

    # Using data loader to train
    for input, label in train_dataloader:
        out_gat = gat_model(input[0], input[1])
        output = decision(out_gat)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
    print(f'CEL:{loss}\n')

    gat_explain = gat_model.explain(input[0][0], input[1][0])
    # print(gat_explain)
    explain = shap.GradientExplainer(decision, out_gat)
    shap_values = explain.shap_values(out_gat)
    # print(shap_values)
    explain = explainer.Gradient(decision, out_gat)
    shap_values = explain.shap_values(out_gat, return_variances = True)
    # print(shap_values)
    return gat_model


def test_bert(train_dataloader, config, factory_kwargs,):
    print('Testing Full Model')

    # Pre-train part
    trainer = pre_trainer.Set_Trainer(config, factory_kwargs)
    pt_loss = trainer.train(train_dataloader)
    print(f'Pre Trainer CEL:{pt_loss}\n')

    # Fine tune part
    tuner = fine_tuner.Tuner(
        gat = trainer.model.gat,
        encoder = trainer.model.encoder,
        bin_pro = trainer.model.bin_pro,
        n_hidden = config['fine_tuner']['n_hidden'],
        n_class = config['fine_tuner']['n_class'],
        **factory_kwargs,
    )
    for input, label in train_dataloader:
        output = tuner.model(input[0], input[1])
        loss = tuner.criterion(output, label)
        loss.backward()

    gat_explain = tuner.model.gat.explain(input[0][0], input[1][0])
    explain = explainer.Gradient(tuner.model, input)
    shap_values = explain.shap_values(input, return_variances = True)
    # print(shap_values)
    print('Last Fine Tuner CEL:', loss, '\n')
    return trainer.model

if __name__ == '__main__':
    # Create the cache directory if not present
    if not os.path.exists('../data/ignore'):
        os.makedirs('../data/ignore')

    # Parameters to fake data
    k = 10
    top_k = 4
    n_sample = 10
    batch_size = 4
    n_class = 4

    config = JSON.decode(resource_filename(
        __name__, '../data/config/test_config.json'
        )
    )
    config['fine_tuner']['n_class'] = n_class
    # JSON.encode(config, 'test_config.json')
    factory_kwargs = {'device': 'cpu', 'dtype': torch.float32,}

    # Generate Fake data
    samples = [
        [
            torch.randn(
                k,
                config['gat']['params']['d_model'],
                dtype = factory_kwargs['dtype']
            ),
            torch.randn(top_k, k, dtype = factory_kwargs['dtype'])
        ] for i in range(n_sample - 1)
    ]
    # To test data loader not messing up exp data and adj mats
    samples.append(
        [
            torch.ones(
                k,
                config['gat']['params']['d_model'],
                dtype = factory_kwargs['dtype']
            ),
            torch.ones(top_k, k, dtype = factory_kwargs['dtype'])
        ]
    )

    train_dataloader = DataLoader(
        lib.FateZ_Dataset(
            samples = samples,
            labels = torch.empty(n_sample, dtype = torch.long).random_(n_class)
        ),
        batch_size = batch_size,
        shuffle = True
    )

    print('Fake gene num:', k)
    print('Fake TF num:', top_k)
    print('Fake Sample Number:', n_sample)
    print('Batch Size:', batch_size)
    print('Class Number:', n_class, '\n')

    mlp_param = {
        'd_model':config['gat']['params']['en_dim'],
        'n_hidden': 4,
        'n_class':n_class,
    }
    temp = test_gat(train_dataloader, config, factory_kwargs, mlp_param)
    model.Save(temp, '../data/ignore/gat.model')

    temp = test_bert(train_dataloader, config, factory_kwargs)
    model.Save(temp, '../data/ignore/bert_encoder.model')


    from sklearn import cluster
    eps = 0.5
    # Set model
    dbscan = cluster.DBSCAN(eps = eps)
    kmeans = cluster.KMeans(n_clusters = 2)
    # Flatten feature matrices for clustering
    data = [torch.reshape(x[0], (-1,)).tolist() for x in samples]

    dbscan.fit(data)
    kmeans.fit(data)
    print(dbscan.labels_.astype(int))
    print(kmeans.labels_.astype(int))
