#!/usr/bin/env python3
"""
Test script to make sure FateZ working properly

ToDo:
Need to revise explainary mechanism if using graph embedder

author: jy
"""
import sys
import shap
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import fatez
import fatez.lib as lib
import fatez.lib.grn as grn
import fatez.tool.gff as gff
import fatez.tool.JSON as JSON
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gat as gat
import fatez.process as process
import fatez.process.explainer as explainer
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
import fatez.model.criterion as crit
import fatez.model.position_embedder as pe
from pkg_resources import resource_filename
from collections import OrderedDict


def make_template_grn_jjy():
    mm10_gff = gff.Reader('E:\\public/gencode.vM25.basic.annotation.gff3.gz')
    mm10_template = mm10_gff.get_genes_gencode(id = 'GRCm38_template')
    print(mm10_template.gene_regions)


def test_grn():
    toy = grn.GRN(id = 'toy')
    # fake genes
    gene_list = [
        grn.Gene(id = 'a', symbol = 'AAA'),
        grn.Gene(id = 'b', symbol = 'BBB'),
        grn.Gene(id = 'c', symbol = 'CCC'),
    ]
    # fake grps
    grp_list = [
        grn.GRP(reg_source = gene_list[0], reg_target = gene_list[1]),
        grn.GRP(reg_source = gene_list[0], reg_target = gene_list[2]),
        grn.GRP(reg_source = gene_list[1], reg_target = gene_list[2]),
    ]
    # populate toy GRN
    for gene in gene_list:
        toy.add_gene(gene)
    for grp in grp_list:
        toy.add_grp(grp)

    toy.as_dict()
    toy.as_digraph()
    toy.save('../data/toy.grn.js.gz')
    # Load new GRN
    del toy
    toy_new = grn.GRN()
    toy_new.load('../data/toy.grn.js.gz')
    for id, rec in toy_new.grps.items():
        print(rec.reg_source.symbol)

# if __name__ == '__main__':
#     make_template_grn()
#     test_grn()




class Faker(object):
    """
    Test modules with fake data
    """

    def __init__(self,
        model_config:dict = None,
        n_sample:int = 10,
        batch_size:int = 5,
        simpler_samples:bool = False,
        device:str = 'cpu',
        dtype:type = torch.float32,
        ):
        super(Faker, self).__init__()
        if model_config is None:
            path = '../data/config/gat_bert_config.json'
            # path = '../data/config/gat_bert_cnn1d_config.json'
            # path = '../data/config/gat_bert_rnn_config.json'
            self.config = JSON.decode(resource_filename(__name__, path))
        else:
            self.config = model_config

        n_features = self.config['input_sizes'][0][-1]

        self.n_sample = n_sample
        self.batch_size = batch_size
        self.simpler_samples = simpler_samples
        self.factory_kwargs = {'device': device, 'dtype': dtype,}

    def make_data_loader(self, sparse_data = True):
        """
        Generate random data with given parameters and
        set up a PyTorch DataLoader.

        :return:
            torch.utils.data.DataLoader
        """
        assert self.config['fine_tuner']['n_class'] == 2

        samples = list()
        # Prepare labels
        t0_labels = torch.zeros(int(self.n_sample/2), dtype=torch.long)
        t1_labels = torch.ones(self.n_sample - len(t0_labels), dtype=torch.long)
        labels = torch.cat((t0_labels, t1_labels), -1)

        def rand_sample():
            dtype = self.factory_kwargs['dtype']
            fea_m = torch.randn(self.config['input_sizes'][0][1:], dtype=dtype)
            adj_m = torch.randn(self.config['input_sizes'][1][1:], dtype=dtype)
            if self.simpler_samples:
                fea_m = fea_m * 0 + 1
                adj_m = adj_m * 0 + 1
            fea_m[-2:] *= 0
            adj_m[:,-2:] *= 0
            return fea_m, adj_m

        def append_sample(samples, fea_m, adj_m):
            if sparse_data:
                samples.append([fea_m.to_sparse(), lib.Adj_Mat(adj_m).sparse])
            else:
                # Dense version
                samples.append([fea_m, adj_m])

        # Prepare type_0 samples
        for i in range(len(t0_labels)):
            fea_m, adj_m = rand_sample()
            fea_m[-1] += 9
            adj_m[:,-1] += 9
            append_sample(samples, fea_m, adj_m)

        # Prepare type_1 samples
        for i in range(len(t1_labels)):
            fea_m, adj_m = rand_sample()
            fea_m[-1] += 1
            adj_m[:,-1] += 1
            append_sample(samples, fea_m, adj_m)

        return DataLoader(
            lib.FateZ_Dataset(samples = samples, labels = labels),
            batch_size = self.batch_size,
            shuffle = True
        )

    def test_gat(self, config:dict = None, decision = None):
        """
        Function to test whether GAT in FateZ is performing properly or not.

        :param config:
            The configuration of testing model.

        :param decision:
            The decision model.

        :return:
            GAT model
        """
        print('Testing GAT.\n')
        suppressor = process.Quiet_Mode()
        # Initialize
        device = self.factory_kwargs['device']
        data_loader = self.make_data_loader()
        if config is None: config = self.config
        graph_embedder = pe.Skip()
        gat_model = gat.Set(
            config['gnn'],
            config['input_sizes'],
            self.factory_kwargs
        )
        if decision is None:
            mlp_param = {
                'd_model': self.config['gnn']['params']['en_dim'],
                'n_hidden': 4,
                'n_class': self.config['fine_tuner']['n_class'],
            }
            decision = mlp.Model(**mlp_param, **self.factory_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion = crit.Accuracy(requires_grad = True)

        # Using data loader to train
        for input, label in data_loader:
            output = graph_embedder(input[0], adj = input[1])
            out_gat = gat_model(output, input[1])
            output = decision(out_gat)
            loss = criterion(output, label.to(device))
            loss.backward()
        print(f'\tGAT Green.\n')


        """
        Need to revise explainary mechanism if using graph embedder
        """
        gat_explain = gat_model.explain(input[0][0], input[1][0])
        # print(gat_explain)
        explain = shap.GradientExplainer(decision, out_gat)
        shap_values = explain.shap_values(out_gat)
        # print(shap_values)
        explain = explainer.Gradient(decision, out_gat)
        shap_values = explain.shap_values(out_gat, return_variances = True)
        # print(shap_values)
        print(f'\tExplainer Green.\n')
        return gat_model

    def test_full_model(self,
        config:dict = None,
        epoch:int = 50,
        quiet:bool = True
        ):
        """
        Function to test whether FateZ is performing properly or not.

        :param config:
            The configuration of testing model.

        :return:
            Pre-trainer model
        """
        print('Testing Full Model.\n')
        suppressor = process.Quiet_Mode()
        # Initialize
        device = self.factory_kwargs['device']
        data_loader = self.make_data_loader()
        if config is None: config = self.config

        # Pre-train part
        trainer = pre_trainer.Set(config, self.factory_kwargs)
        for i in range(epoch):
            report = trainer.train(data_loader)
        print(f'\tPre-Trainer Green.\n')

        # Fine tune part
        if quiet: suppressor.on()
        tuner = fine_tuner.Set(
            config, self.factory_kwargs, prev_model = trainer.model
        )
        for i in range(epoch):
            report = tuner.train(data_loader, report_batch = False)
            print(f'Epoch {i} Loss: {report.iloc[0,0]}')
        # Test fine tune model
        report = tuner.test(data_loader, report_batch = True)
        print('Tuner Test Report')
        print(report)
        if quiet: suppressor.off()
        print(f'\tFine-Tuner Green.\n')

        # Test explain
        suppressor.on()
        adj_explain = torch.zeros(self.config['input_sizes'][1][1:])
        node_explain = torch.zeros(self.config['input_sizes'][0][1:])

        bg_data = DataLoader(data_loader.dataset, batch_size = self.n_sample)
        bg_data = [a for a, _ in bg_data][0]
        bg_data = [a.to(self.factory_kwargs['device']) for a in bg_data]
        explain = explainer.Gradient(tuner.model, bg_data)

        for x, y in data_loader:
            # Explain GAT to obtain adj explanations
            for i in range(len(x[0])):
                adj_explain+=tuner.model.gat.explain(x[0][i], x[1][i]).to('cpu')

            node_exp, vars = explain.shap_values(
                [i.to(self.factory_kwargs['device']).to_dense() for i in x],
                return_variances = True
            )
            for exp in node_exp[0][0]: node_explain += abs(exp)
            break
        suppressor.off()

        print(adj_explain)
        print(torch.sum(node_explain, dim = -1))
        print(f'\tExplainer Green.\n')

        return trainer.model, tuner.model

if __name__ == '__main__':
    a = Faker(batch_size = 4, simpler_samples = False, device = 'cuda')
    # models = a.test_gat()
    models = a.test_full_model(quiet = False)
