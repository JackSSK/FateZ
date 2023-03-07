#!/usr/bin/env python3
"""
Test script to make sure FateZ working properly

ToDo:
Need to revise explainary mechanism if using graph embedder

author: jy
"""
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fatez
import fatez.lib as lib
import fatez.lib.grn as grn
import fatez.tool.gff as gff
import fatez.tool.JSON as JSON
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gat as gat
import fatez.process.explainer as explainer
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
import fatez.process.position_embedder as pe
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

if __name__ == '__main__':
    make_template_grn()
    test_grn()




class Faker(object):
    """
    Test modules with fake data
    """

    def __init__(self,
        model_config:dict = None,
        n_sample:int = 10,
        batch_size:int = 4,
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
        self.k = self.config['input_sizes'][1][-1]
        self.top_k = self.config['input_sizes'][1][-2]
        self.n_class = self.config['fine_tuner']['n_class']
        self.n_features = self.config['input_sizes'][0][-1]
        assert self.n_features == self.config['gat']['params']['d_model']

        self.n_sample = n_sample
        self.batch_size = batch_size
        self.factory_kwargs = {'device': device, 'dtype': dtype,}

    def make_data_loader(self, test_consist:bool = False):
        """
        Generate random data with given parameters and
        set up a PyTorch DataLoader.

        :return:
            torch.utils.data.DataLoader
        """
        samples = [
            [
                # Fake node feature matrices
                torch.randn(
                    self.k,
                    self.n_features,
                    dtype = self.factory_kwargs['dtype'],
                    device = self.factory_kwargs['device']
                ),
                # Fake adjacency matrices
                torch.randn(
                    self.top_k,
                    self.k,
                    dtype = self.factory_kwargs['dtype'],
                    device = self.factory_kwargs['device']
                )
            ] for i in range(self.n_sample)
        ]

        # To test data loader not messing up exp data and adj mats
        if test_consist:
            samples.pop(-1)
            samples.append(
                [
                    torch.ones(
                        self.k,
                        self.n_features,
                        dtype = self.factory_kwargs['dtype'],
                        device = self.factory_kwargs['device']
                    ),
                    torch.ones(
                        self.top_k,
                        self.k,
                        dtype = self.factory_kwargs['dtype'],
                        device = self.factory_kwargs['device']
                    )
                ]
            )

        data_loader = DataLoader(
            lib.FateZ_Dataset(
                samples = samples,
                labels = torch.empty(
                    self.n_sample,
                    dtype = torch.long,
                    device = self.factory_kwargs['device']
                ).random_(self.n_class)
            ),
            batch_size = self.batch_size,
            shuffle = True
        )

        if test_consist:
            print('Under construction: previously examinated by human eyes')

        return data_loader

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
        # Initialize
        data_loader = self.make_data_loader()
        if config is None: config = self.config
        graph_embedder = pe.Skip()
        gat_model = gat.Set(config['gat'], self.factory_kwargs)
        if decision is None:
            mlp_param = {
                'd_model': self.config['gat']['params']['en_dim'],
                'n_hidden': 4,
                'n_class': self.n_class,
            }
            decision = mlp.Model(**mlp_param, **self.factory_kwargs)

        # Using data loader to train
        for input, label in data_loader:
            output = graph_embedder(input[0], adj = input[1])
            out_gat = gat_model(output, input[1])
            output = decision(out_gat)
            loss = nn.CrossEntropyLoss()(output, label)
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

    def test_full_model(self, config:dict = None,):
        """
        Function to test whether FateZ is performing properly or not.

        :param config:
            The configuration of testing model.

        :return:
            Pre-trainer model
        """
        print('Testing Full Model.\n')
        # Initialize
        epoch = 1
        data_loader = self.make_data_loader()
        if config is None: config = self.config

        # Pre-train part
        trainer = pre_trainer.Set(config, self.factory_kwargs)
        for i in range(epoch):
            report = trainer.train(data_loader)
        print(f'\tPre-Trainer Green.\n')

        # Fine tune part
        tuner = fine_tuner.Set(
            config, self.factory_kwargs, prev_model = trainer.model
        )
        for i in range(epoch):
            report = tuner.train(data_loader, report_batch = True)
        # Test fine tune model
        report = tuner.test(data_loader, report_batch = True)
        print(f'\tFine-Tuner Green.\n')

        # Test explain
        for x, y in data_loader:
            gat_explain = tuner.model.gat.explain(x[0][0], x[1][0])
            # print(gat_explain)
            explain = explainer.Gradient(tuner.model, x)
            shap_values = explain.shap_values(x, return_variances = True)
            # print(shap_values)
            break
        print(f'\tExplainer Green.\n')

        return trainer.model, tuner.model
